from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import importlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from scipy.signal import fftconvolve
except ImportError:  # pragma: no cover
    fftconvolve = None

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover
    register_heif_opener = None


SOURCE_DIR = Path(__file__).resolve().parent
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_DIR


def resolve_resource_path(*parts: str) -> Path:
    for base_dir in (APP_DIR, SOURCE_DIR):
        candidate = base_dir.joinpath(*parts)
        if candidate.exists():
            return candidate
    return APP_DIR.joinpath(*parts)


ROOT_DIR = APP_DIR
LOCAL_SHARP_SRC = SOURCE_DIR / "ml-sharp" / "src"
LOCAL_DA360_ROOT = resolve_resource_path("third_party", "DA360")
LOCAL_IMAGEMAGICK_ROOT = APP_DIR / "third_party" / "ImageMagick"
DEFAULT_CONFIG_PATH = resolve_resource_path("insp_settings.json")
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_DA360_CHECKPOINT_PATH = resolve_resource_path("checkpoints", "DA360_large.pth")
GUI_SETTINGS_PATH = resolve_resource_path("easy_360_sharp_gui_settings.json")
SEEDVR2_SETTINGS_PATH = resolve_resource_path("seedvr2_settings.json")
SEEDVR2_CLI_PATH = resolve_resource_path("seedvr2_videoupscaler", "inference_cli.py")
IMAGEMAGICK_PRESET_SETTINGS: dict[str, dict[str, Any]] = {
    "blur_safe": {
        "imagemagick_auto_level": True,
        "imagemagick_auto_gamma": True,
        "imagemagick_normalize": True,
        "imagemagick_enhance": False,
        "imagemagick_despeckle": False,
        "imagemagick_unsharp_enabled": True,
        "imagemagick_unsharp_value": "0x1.6+1.2+0.02",
        "imagemagick_extra_args": "",
    },
    "classic": {
        "imagemagick_auto_level": True,
        "imagemagick_auto_gamma": True,
        "imagemagick_normalize": True,
        "imagemagick_enhance": True,
        "imagemagick_despeckle": True,
        "imagemagick_unsharp_enabled": True,
        "imagemagick_unsharp_value": "0x1.2+0.8+0.02",
        "imagemagick_extra_args": "",
    },
}
DEFAULT_IMAGEMAGICK_PRESET = "blur_safe"
DEFAULT_IMAGEMAGICK_COMMANDS = "-auto-level -auto-gamma -normalize -unsharp 0x1.6+1.2+0.02"
IMAGEMAGICK_OPTION_FLAGS = (
    ("imagemagick_auto_level", "-auto-level"),
    ("imagemagick_auto_gamma", "-auto-gamma"),
    ("imagemagick_normalize", "-normalize"),
    ("imagemagick_enhance", "-enhance"),
    ("imagemagick_despeckle", "-despeckle"),
)
DEFAULT_IMAGEMAGICK_GUI_SETTINGS = {
    "imagemagick_preset": DEFAULT_IMAGEMAGICK_PRESET,
    **IMAGEMAGICK_PRESET_SETTINGS[DEFAULT_IMAGEMAGICK_PRESET],
}
DEFAULT_DEBLUR_GUI_SETTINGS = {
    "enable_deblur_preprocessing": False,
    "deblur_strength": "medium",
}
DEBLUR_PROFILES: dict[str, dict[str, Any]] = {
    "low": {"kernel_length": 7, "iterations": 5, "blend": 0.35, "preview_iterations": 2},
    "medium": {"kernel_length": 11, "iterations": 7, "blend": 0.5, "preview_iterations": 3},
    "high": {"kernel_length": 15, "iterations": 9, "blend": 0.65, "preview_iterations": 4},
}
DEBLUR_ANGLE_CANDIDATES = (0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0)
SUPPORTED_INPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}
COMPRESSED_OUTPUT_SUFFIXES = {".spx", ".spz", ".sog"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


if str(LOCAL_SHARP_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SHARP_SRC))

if TYPE_CHECKING:
    from sharp.utils.gaussians import Gaussians3D


LOGGER = logging.getLogger("insp_to_splat")
SUPPORTED_ALIGNMENT_MODES = ("none", "da360", "overlap")
OVERLAP_ALIGNMENT_YAW_BINS = 12
OVERLAP_ALIGNMENT_PITCH_BINS = 10
OVERLAP_ALIGNMENT_MIN_BIN_POINTS = 4
OVERLAP_ALIGNMENT_MIN_SHARED_BINS = 6
OVERLAP_DEPTH_LAYER_BINS = 6
OVERLAP_DEPTH_LAYER_MIN_SAMPLES = 16


def format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, remaining = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remaining:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {remaining:.1f}s"


@dataclass(frozen=True)
class StepTiming:
    name: str
    duration_seconds: float
    depth: int
    succeeded: bool


CURRENT_STEP_TIMINGS: ContextVar[list[StepTiming] | None] = ContextVar("current_step_timings", default=None)
CURRENT_STEP_TIMING_DEPTH: ContextVar[int] = ContextVar("current_step_timing_depth", default=0)


def log_step_timing_summary(step_timings: Sequence[StepTiming], total_elapsed: float) -> None:
    top_level_steps = [step for step in step_timings if step.depth == 0]
    if not top_level_steps:
        return

    LOGGER.info("Step timing summary (top-level steps, slowest first):")
    LOGGER.info("  %9s  %6s  %s", "Duration", "Share", "Step")
    for step in sorted(top_level_steps, key=lambda item: item.duration_seconds, reverse=True):
        share = (step.duration_seconds / total_elapsed * 100.0) if total_elapsed > 1e-9 else 0.0
        status_suffix = "" if step.succeeded else " [failed]"
        LOGGER.info("  %9s  %5.1f%%  %s%s", format_duration(step.duration_seconds), share, step.name, status_suffix)

    timed_total = sum(step.duration_seconds for step in top_level_steps)
    overhead = max(0.0, total_elapsed - timed_total)
    overhead_share = (overhead / total_elapsed * 100.0) if total_elapsed > 1e-9 else 0.0
    LOGGER.info("  %9s  %5.1f%%  %s", format_duration(overhead), overhead_share, "Untimed / overhead")


@contextmanager
def log_timed_step(step_name: str) -> Iterator[None]:
    step_timings = CURRENT_STEP_TIMINGS.get()
    depth = CURRENT_STEP_TIMING_DEPTH.get()
    depth_token = CURRENT_STEP_TIMING_DEPTH.set(depth + 1)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        if step_timings is not None:
            step_timings.append(StepTiming(step_name, elapsed, depth, False))
        LOGGER.info("Step failed: %s after %s", step_name, format_duration(elapsed))
        raise
    else:
        elapsed = time.perf_counter() - start
        if step_timings is not None:
            step_timings.append(StepTiming(step_name, elapsed, depth, True))
        LOGGER.info("Step complete: %s in %s", step_name, format_duration(elapsed))
    finally:
        CURRENT_STEP_TIMING_DEPTH.reset(depth_token)


@dataclass(frozen=True)
class FaceOrientation:
    name: str
    right: tuple[float, float, float]
    down: tuple[float, float, float]
    forward: tuple[float, float, float]

    @property
    def rotation_matrix(self) -> np.ndarray:
        return np.column_stack((self.right, self.down, self.forward)).astype(np.float32)


@dataclass(frozen=True)
class ExtractionLayout:
    name: str
    views: tuple[FaceOrientation, ...]
    focal_px: float
    focal_y_px: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class OverlapBoundaryField:
    boundary_yaw_deg: float
    half_overlap_degrees: float
    yaw_edges: np.ndarray
    pitch_edges: np.ndarray
    log_scale_grid: np.ndarray
    fallback_log_scale: float
    peer_name: str


@dataclass(frozen=True)
class DepthLayerCurve:
    log_radius_knots: np.ndarray
    log_scale_knots: np.ndarray
    fallback_log_scale: float


@dataclass(frozen=True)
class DA360Predictor:
    model: torch.nn.Module
    input_height: int
    input_width: int
    model_name: str


@dataclass
class PipelineResult:
    output_path: Path
    depth_map_path: Path | None = None
    generated_outputs: list[Path] | None = None
    display_path: Path | None = None


FACE_ORIENTATIONS = (
    FaceOrientation("front", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    FaceOrientation("right", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    FaceOrientation("back", (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    FaceOrientation("left", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    FaceOrientation("top", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    FaceOrientation("bottom", (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a 2:1 equirectangular panorama (.jpg/.png/.heic) into stitched "
            "perspective views, run Apple's SHARP predictor on each view, rotate the view splats "
            "into a common frame, and merge them into one output splat file."
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input panorama file.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output splat file.")
    parser.add_argument(
        "--side-count",
        type=int,
        default=0,
        help="Number of horizon views to extract around the panorama. Use 0 for the config default.",
    )
    parser.add_argument(
        "--face-size",
        type=int,
        default=0,
        help="Perspective view size in pixels. Use 0 to derive it from the panorama width.",
    )
    parser.add_argument(
        "--format",
        choices=("ply", "spx", "spz", "sog"),
        default=None,
        help="Output file format. Defaults to the output suffix or config value.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=None,
        help="Compression quality 1-9 for .spx/.spz/.sog output.",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=None,
        help="Spherical harmonics degree to request when gsbox conversion is used.",
    )
    parser.add_argument(
        "--device",
        choices=("default", "cuda", "cpu", "mps"),
        default="default",
        help="Inference device.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional SHARP checkpoint path. Defaults to Apple's published checkpoint.",
    )
    parser.add_argument(
        "--da360-checkpoint",
        type=Path,
        default=None,
        help="Optional DA360 checkpoint path. Defaults to checkpoints/DA360_large.pth.",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=SUPPORTED_ALIGNMENT_MODES,
        default=None,
        help="Slice alignment mode: none, da360, or overlap.",
    )
    parser.add_argument(
        "--disable-da360-alignment",
        action="store_true",
        help="Disable DA360 depth alignment and use the raw SHARP per-view scales.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep extracted face images and per-face PLY files next to the output.",
    )
    parser.add_argument(
        "--delete-temp-files",
        action="store_true",
        help="Delete the Temp workspace after processing finishes successfully.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Optional directory for intermediate faces and per-face splats.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON config file with default options.",
    )
    parser.add_argument(
        "--gsbox",
        type=Path,
        default=None,
        help="Path to gsbox.exe for compressed output conversion.",
    )
    parser.add_argument(
        "--enable-seedvr2-upscale",
        action="store_true",
        help="Sharpen the panorama and upscale extracted face images with SeedVR2 before SHARP prediction.",
    )
    parser.add_argument(
        "--enable-imagemagick-optimization",
        action="store_true",
        help="Optimize the panorama with ImageMagick before slicing it into SHARP views.",
    )
    parser.add_argument(
        "--enable-deblur-preprocessing",
        action="store_true",
        help="Apply a motion-deblur stage on extracted faces before SeedVR2 and SHARP.",
    )
    parser.add_argument(
        "--deblur-strength",
        choices=("low", "medium", "high"),
        default="medium",
        help="Strength of the motion-deblur stage applied to extracted faces.",
    )
    parser.add_argument(
        "--imagemagick",
        type=Path,
        default=None,
        help="Optional path to magick.exe. Uses PATH when omitted.",
    )
    parser.add_argument(
        "--imagemagick-commands",
        type=str,
        default=DEFAULT_IMAGEMAGICK_COMMANDS,
        help="ImageMagick command string applied before slicing.",
    )
    parser.add_argument(
        "--alignment-grid-resolution",
        type=int,
        default=8,
        help="NxN grid size for DA360 depth alignment (1-64). Higher values preserve finer spatial detail.",
    )
    parser.add_argument(
        "--alignment-detail-weight",
        type=float,
        default=0.0,
        help="Blend between smooth grid scale (0.0) and per-point raw scale (1.0) for depth alignment detail.",
    )
    parser.add_argument(
        "--cutoff-height-percent",
        type=float,
        default=0.0,
        help="Crop away the bottom part of each extracted slice before SHARP inference (0-40 percent).",
    )
    parser.add_argument(
        "--enable-alignment-sweep",
        action="store_true",
        help="Generate a 5x5 sweep of grid-resolution/detail-preservation variants after SHARP inference has run once.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain an object: {path}")
    return data


def load_seedvr2_settings() -> dict:
    settings = load_config(SEEDVR2_SETTINGS_PATH)
    gui_settings = load_config(GUI_SETTINGS_PATH)
    if not gui_settings:
        return settings

    seedvr2_keys = {
        "model_name": "seedvr2_model_name",
        "output_format": "seedvr2_output_format",
        "color_correction": "seedvr2_color_correction",
        "attention_mode": "seedvr2_attention_mode",
        "cuda_device": "seedvr2_cuda_device",
        "dit_offload_device": "seedvr2_dit_offload_device",
        "vae_offload_device": "seedvr2_vae_offload_device",
        "tensor_offload_device": "seedvr2_tensor_offload_device",
        "resolution_factor": "seedvr2_resolution_factor",
        "stretch_proportions": "seedvr2_stretch_proportions",
        "target_short_side": "seedvr2_target_short_side",
        "min_resolution": "seedvr2_min_resolution",
        "max_resolution": "seedvr2_max_resolution",
        "batch_size": "seedvr2_batch_size",
        "seed": "seedvr2_seed",
        "skip_first_frames": "seedvr2_skip_first_frames",
        "blocks_to_swap": "seedvr2_blocks_to_swap",
        "vae_encode_tile_size": "seedvr2_vae_encode_tile_size",
        "vae_encode_tile_overlap": "seedvr2_vae_encode_tile_overlap",
        "vae_decode_tile_size": "seedvr2_vae_decode_tile_size",
        "vae_decode_tile_overlap": "seedvr2_vae_decode_tile_overlap",
        "compile_backend": "seedvr2_compile_backend",
        "compile_mode": "seedvr2_compile_mode",
        "swap_io_components": "seedvr2_swap_io_components",
        "vae_encode_tiled": "seedvr2_vae_encode_tiled",
        "vae_decode_tiled": "seedvr2_vae_decode_tiled",
        "cache_dit": "seedvr2_cache_dit",
        "cache_vae": "seedvr2_cache_vae",
        "debug_enabled": "seedvr2_debug_enabled",
    }
    for target_key, gui_key in seedvr2_keys.items():
        if gui_key in gui_settings:
            settings[target_key] = gui_settings[gui_key]
    return settings


def resolve_seedvr2_stretch_dimensions(
    image_width: int,
    image_height: int,
    settings: Mapping[str, Any],
) -> tuple[int, int, bool]:
    if not bool(settings.get("stretch_proportions", False)):
        return image_width, image_height, False

    short_side = min(image_width, image_height)
    long_side = max(image_width, image_height)
    min_res = int(settings.get("min_resolution", 0) or 0)

    # When no minimum output resolution floor is set, stretching defaults to a
    # square input that matches the current long side.
    if min_res <= 0:
        if image_width >= image_height:
            target_width, target_height = long_side, long_side
        else:
            target_width, target_height = long_side, long_side
        return target_width, target_height, (target_width, target_height) != (image_width, image_height)

    target_short_side_raw = settings.get("target_short_side", 0)
    try:
        target_short_side = int(target_short_side_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("SeedVR2 target short side must be an integer when stretch proportions is enabled.") from exc

    if target_short_side <= 0:
        raise ValueError("SeedVR2 target short side must be greater than 0 when stretch proportions is enabled.")

    if target_short_side < short_side:
        raise ValueError(
            f"SeedVR2 target short side {target_short_side} is smaller than the current short side {short_side}."
        )
    if target_short_side > long_side:
        raise ValueError(
            f"SeedVR2 target short side {target_short_side} exceeds the current long side {long_side}."
        )

    if image_width >= image_height:
        target_width, target_height = long_side, target_short_side
    else:
        target_width, target_height = target_short_side, long_side
    return target_width, target_height, (target_width, target_height) != (image_width, image_height)


def normalize_imagemagick_preset_name(name: str | None) -> str:
    normalized = str(name or DEFAULT_IMAGEMAGICK_PRESET).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in IMAGEMAGICK_PRESET_SETTINGS:
        return normalized
    if normalized == "default":
        return DEFAULT_IMAGEMAGICK_PRESET
    return "custom"


def get_imagemagick_preset_settings(name: str | None) -> dict[str, Any]:
    preset_name = normalize_imagemagick_preset_name(name)
    if preset_name in IMAGEMAGICK_PRESET_SETTINGS:
        return {
            "imagemagick_preset": preset_name,
            **IMAGEMAGICK_PRESET_SETTINGS[preset_name],
        }
    return dict(DEFAULT_IMAGEMAGICK_GUI_SETTINGS)


def infer_imagemagick_preset_name(settings: Mapping[str, Any] | None) -> str:
    if not settings:
        return DEFAULT_IMAGEMAGICK_PRESET
    for preset_name, preset_settings in IMAGEMAGICK_PRESET_SETTINGS.items():
        matches = True
        for key, expected_value in preset_settings.items():
            if settings.get(key) != expected_value:
                matches = False
                break
        if matches:
            return preset_name
    return "custom"


def normalize_deblur_strength(strength: str | None) -> str:
    normalized = str(strength or "medium").strip().lower()
    if normalized not in DEBLUR_PROFILES:
        raise ValueError(f"Unsupported deblur strength: {strength}")
    return normalized


def resolve_deblur_profile(strength: str | None) -> dict[str, Any]:
    profile_name = normalize_deblur_strength(strength)
    return {"strength": profile_name, **DEBLUR_PROFILES[profile_name]}


def build_motion_psf(length: int, angle_deg: float) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for motion deblur but is not installed in the current Python environment.")
    size = max(3, int(length))
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    cv2.line(kernel, (0, center), (size - 1, center), 1.0, 1, lineType=cv2.LINE_AA)
    rotation = cv2.getRotationMatrix2D((center, center), float(angle_deg), 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size), flags=cv2.INTER_CUBIC)
    kernel = np.clip(kernel, 0.0, None)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 1e-8:
        kernel[center, center] = 1.0
        kernel_sum = 1.0
    return kernel / kernel_sum


def richardson_lucy_deblur(image: np.ndarray, psf: np.ndarray, iterations: int) -> np.ndarray:
    if fftconvolve is None:
        raise RuntimeError("SciPy is required for deblur preprocessing but is not installed in the current Python environment.")
    estimate = np.clip(image.astype(np.float32, copy=True), 0.0, 1.0)
    observed = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    psf_mirror = psf[::-1, ::-1]
    for _ in range(max(1, int(iterations))):
        conv = fftconvolve(estimate, psf, mode="same")
        relative = observed / np.maximum(conv, 1e-5)
        estimate *= fftconvolve(relative, psf_mirror, mode="same")
        estimate = np.clip(estimate, 0.0, 1.0)
    return estimate


def score_sharpness_laplacian(image: np.ndarray) -> float:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for deblur preprocessing but is not installed in the current Python environment.")
    laplacian = cv2.Laplacian(image.astype(np.float32, copy=False), cv2.CV_32F)
    return float(laplacian.var())


def estimate_motion_angle(luma: np.ndarray, profile: Mapping[str, Any]) -> float:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for deblur preprocessing but is not installed in the current Python environment.")
    preview = luma.astype(np.float32, copy=False)
    preview_height, preview_width = preview.shape[:2]
    preview_longest = max(preview_height, preview_width)
    if preview_longest > 768:
        scale = 768.0 / float(preview_longest)
        preview = cv2.resize(preview, (max(1, int(round(preview_width * scale))), max(1, int(round(preview_height * scale)))), interpolation=cv2.INTER_AREA)

    best_angle = 0.0
    best_score = float("-inf")
    preview_iterations = int(profile.get("preview_iterations", 2))
    kernel_length = int(profile.get("kernel_length", 11))
    for angle_deg in DEBLUR_ANGLE_CANDIDATES:
        psf = build_motion_psf(kernel_length, angle_deg)
        restored = richardson_lucy_deblur(preview, psf, preview_iterations)
        score = score_sharpness_laplacian(restored)
        if score > best_score:
            best_score = score
            best_angle = float(angle_deg)
    return best_angle


def deblur_face_image(image_array: np.ndarray, profile: Mapping[str, Any]) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for deblur preprocessing but is not installed in the current Python environment.")
    rgb = np.clip(image_array.astype(np.float32) / 255.0, 0.0, 1.0)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    luma = ycrcb[..., 0]
    angle_deg = estimate_motion_angle(luma, profile)
    psf = build_motion_psf(int(profile["kernel_length"]), angle_deg)
    restored_luma = richardson_lucy_deblur(luma, psf, int(profile["iterations"]))
    blend = float(profile["blend"])
    ycrcb[..., 0] = np.clip((1.0 - blend) * luma + blend * restored_luma, 0.0, 1.0)
    restored_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    LOGGER.info(
        "Deblurred face using motion angle %.0f deg (kernel=%d, iterations=%d, blend=%.2f)",
        angle_deg,
        int(profile["kernel_length"]),
        int(profile["iterations"]),
        blend,
    )
    return np.clip(np.round(restored_rgb * 255.0), 0, 255).astype(np.uint8)


def deblur_faces_with_motion_rl(
    faces: Mapping[str, np.ndarray],
    strength: str,
) -> dict[str, np.ndarray]:
    profile = resolve_deblur_profile(strength)
    LOGGER.info(
        "Running motion deblur on %d faces (strength=%s, kernel=%d, iterations=%d, blend=%.2f)",
        len(faces),
        profile["strength"],
        int(profile["kernel_length"]),
        int(profile["iterations"]),
        float(profile["blend"]),
    )
    return {
        name: deblur_face_image(image_array, profile)
        for name, image_array in faces.items()
    }


def resolve_seedvr2_target_dimensions(
    image_width: int,
    image_height: int,
    settings: Mapping[str, Any],
) -> tuple[int, int, int]:
    factor = int(settings.get("resolution_factor", 2))
    min_res = int(settings.get("min_resolution", 0))
    max_res = int(settings.get("max_resolution", 0))
    longest_side = max(image_width, image_height)
    target_resolution = longest_side * factor
    if min_res > 0 and target_resolution < min_res:
        target_resolution = min_res
    if max_res > 0 and target_resolution > max_res:
        target_resolution = max_res
    scale_factor = target_resolution / float(max(1, longest_side))
    expected_image_width = max(1, int(round(image_width * scale_factor)))
    expected_image_height = max(1, int(round(image_height * scale_factor)))
    return target_resolution, expected_image_width, expected_image_height


def parse_imagemagick_gui_settings(raw_settings: Mapping[str, Any] | None) -> dict[str, Any]:
    settings = dict(DEFAULT_IMAGEMAGICK_GUI_SETTINGS)
    if not raw_settings:
        return settings

    structured_keys = set(DEFAULT_IMAGEMAGICK_GUI_SETTINGS)
    has_structured_values = any(key in raw_settings for key in structured_keys)
    for key in structured_keys:
        if key in raw_settings:
            settings[key] = raw_settings[key]

    legacy_commands = str(raw_settings.get("imagemagick_commands", "") or "").strip()
    if not legacy_commands or has_structured_values:
        return settings

    tokens = shlex.split(legacy_commands)
    extras: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        matched_flag = False
        for key, flag in IMAGEMAGICK_OPTION_FLAGS:
            if token == flag:
                settings[key] = True
                matched_flag = True
                break
        if matched_flag:
            index += 1
            continue
        if token == "-unsharp":
            settings["imagemagick_unsharp_enabled"] = True
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                settings["imagemagick_unsharp_value"] = tokens[index + 1]
                index += 2
            else:
                index += 1
            continue
        extras.append(token)
        index += 1

    settings["imagemagick_extra_args"] = shlex.join(extras) if extras else ""
    return settings


def build_imagemagick_command_tokens(
    command_string: str | None = None,
    option_values: Mapping[str, Any] | None = None,
) -> list[str]:
    if option_values is not None:
        settings = parse_imagemagick_gui_settings(option_values)
        tokens: list[str] = []
        for key, flag in IMAGEMAGICK_OPTION_FLAGS:
            if bool(settings.get(key)):
                tokens.append(flag)
        if bool(settings.get("imagemagick_unsharp_enabled")):
            tokens.append("-unsharp")
            unsharp_value = str(settings.get("imagemagick_unsharp_value", "") or "").strip()
            if unsharp_value:
                tokens.append(unsharp_value)
        extra_args = str(settings.get("imagemagick_extra_args", "") or "").strip()
        if extra_args:
            tokens.extend(shlex.split(extra_args))
        return tokens

    resolved_string = (command_string or DEFAULT_IMAGEMAGICK_COMMANDS).strip() or DEFAULT_IMAGEMAGICK_COMMANDS
    return shlex.split(resolved_string)


def build_imagemagick_command_string(
    command_string: str | None = None,
    option_values: Mapping[str, Any] | None = None,
) -> str:
    tokens = build_imagemagick_command_tokens(command_string=command_string, option_values=option_values)
    return shlex.join(tokens) if tokens else ""


def resolve_imagemagick_command_tokens(args: argparse.Namespace | Any) -> list[str]:
    option_values = {key: getattr(args, key) for key in DEFAULT_IMAGEMAGICK_GUI_SETTINGS if hasattr(args, key)}
    if option_values:
        return build_imagemagick_command_tokens(option_values=option_values)
    return build_imagemagick_command_tokens(command_string=getattr(args, "imagemagick_commands", None))


def _iter_imagemagick_install_candidates() -> Iterable[Path]:
    local_candidates = [
        LOCAL_IMAGEMAGICK_ROOT / "magick.exe",
        LOCAL_IMAGEMAGICK_ROOT / "bin" / "magick.exe",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            yield candidate

    env_roots = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramW6432"),
        os.environ.get("ProgramFiles(x86)"),
    ]
    local_app_data = os.environ.get("LocalAppData")
    if local_app_data:
        env_roots.append(str(Path(local_app_data) / "Programs"))

    seen: set[Path] = set()
    for root in env_roots:
        if not root:
            continue
        root_path = Path(root)
        if not root_path.exists():
            continue
        for install_dir in sorted(root_path.glob("ImageMagick-*"), reverse=True):
            candidate = install_dir / "magick.exe"
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def find_imagemagick_executable(path_value: str | Path | None) -> Path | None:
    resolved = resolve_optional_path(path_value)
    if resolved is not None:
        if resolved.exists():
            return resolved
        return None
    magick_on_path = shutil.which("magick")
    if magick_on_path:
        return Path(magick_on_path)
    for candidate in _iter_imagemagick_install_candidates():
        if candidate.exists():
            return candidate
    return None


def optimize_panorama_with_imagemagick(
    panorama: np.ndarray,
    temp_root: Path,
    magick_path: Path,
    command_tokens: Sequence[str],
) -> np.ndarray:
    preprocess_dir = temp_root / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    input_path = preprocess_dir / "panorama_input.png"
    output_path = preprocess_dir / "panorama_optimized.png"
    Image.fromarray(panorama).save(input_path)

    command = [str(magick_path), str(input_path), *command_tokens, str(output_path)]
    LOGGER.info("Running ImageMagick optimization: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        output = output.strip()
        if output:
            raise RuntimeError(f"ImageMagick optimization failed with exit code {result.returncode}:\n{output}")
        raise RuntimeError(f"ImageMagick optimization failed with exit code {result.returncode}.")

    with Image.open(output_path) as image:
        optimized = image.convert("RGB")
        return np.asarray(optimized).copy()


def sharpen_panorama(image: np.ndarray) -> np.ndarray:
    from PIL import ImageFilter

    pil_img = Image.fromarray(image)
    sharpened = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=2))
    return np.asarray(sharpened)


def upscale_faces_with_seedvr2(
    faces: dict[str, np.ndarray],
    image_width: int,
    image_height: int,
    temp_root: Path,
) -> tuple[dict[str, np.ndarray], int, int]:
    settings = load_seedvr2_settings()
    seedvr2_input_width, seedvr2_input_height, stretch_applied = resolve_seedvr2_stretch_dimensions(
        image_width,
        image_height,
        settings,
    )
    factor = int(settings.get("resolution_factor", 2))
    min_res = int(settings.get("min_resolution", 0))
    max_res = int(settings.get("max_resolution", 0))
    longest_side = max(seedvr2_input_width, seedvr2_input_height)
    target_resolution, expected_image_width, expected_image_height = resolve_seedvr2_target_dimensions(
        seedvr2_input_width,
        seedvr2_input_height,
        settings,
    )

    input_dir = temp_root / "seedvr2_input"
    output_dir = temp_root / "seedvr2_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stretch_applied:
        stretch_target_label = (
            f"auto-square from long side {max(image_width, image_height)}"
            if int(settings.get("min_resolution", 0) or 0) <= 0
            else str(settings.get("target_short_side", ""))
        )
        LOGGER.info(
            "Stretching SeedVR2 inputs before upscale: %dx%d -> %dx%d (target short side %s)",
            image_width,
            image_height,
            seedvr2_input_width,
            seedvr2_input_height,
            stretch_target_label,
        )

    for name, image_array in faces.items():
        image = Image.fromarray(image_array)
        if stretch_applied:
            image = image.resize((seedvr2_input_width, seedvr2_input_height), Image.Resampling.LANCZOS)
        image.save(input_dir / f"{name}.png")

    python_exe = sys.executable
    cmd: list[str] = [
        python_exe, str(SEEDVR2_CLI_PATH),
        str(input_dir),
        "--output", str(output_dir),
        "--output_format", "png",
        "--resolution", str(target_resolution),
        "--dit_model", settings.get("model_name", "seedvr2_ema_3b_fp8_e4m3fn.safetensors"),
        "--color_correction", settings.get("color_correction", "lab"),
        "--attention_mode", settings.get("attention_mode", "sdpa"),
        "--batch_size", str(settings.get("batch_size", "1")),
        "--seed", str(settings.get("seed", "42")),
        "--blocks_to_swap", str(settings.get("blocks_to_swap", "0")),
        "--dit_offload_device", settings.get("dit_offload_device", "none"),
        "--vae_offload_device", settings.get("vae_offload_device", "none"),
        "--tensor_offload_device", settings.get("tensor_offload_device", "cpu"),
        "--compile_backend", settings.get("compile_backend", "inductor"),
        "--compile_mode", settings.get("compile_mode", "default"),
    ]

    if max_res > 0:
        cmd += ["--max_resolution", str(max_res)]

    cuda_device = settings.get("cuda_device")
    if cuda_device:
        cmd += ["--cuda_device", str(cuda_device)]

    if settings.get("swap_io_components", False):
        cmd.append("--swap_io_components")
    if settings.get("vae_encode_tiled", False):
        cmd += [
            "--vae_encode_tiled",
            "--vae_encode_tile_size", str(settings.get("vae_encode_tile_size", "1024")),
            "--vae_encode_tile_overlap", str(settings.get("vae_encode_tile_overlap", "128")),
        ]
    if settings.get("vae_decode_tiled", False):
        cmd += [
            "--vae_decode_tiled",
            "--vae_decode_tile_size", str(settings.get("vae_decode_tile_size", "1024")),
            "--vae_decode_tile_overlap", str(settings.get("vae_decode_tile_overlap", "128")),
        ]
    if settings.get("cache_dit", False):
        cmd.append("--cache_dit")
    if settings.get("cache_vae", False):
        cmd.append("--cache_vae")
    if settings.get("debug_enabled", False):
        cmd.append("--debug")

    LOGGER.info(
        "Running SeedVR2 upscale: %d faces, %dx%d input (%dx%d after stretch) -> longest side %d, target %d (factor %d, min %d, max %d)",
        len(faces), image_width, image_height, seedvr2_input_width, seedvr2_input_height, longest_side, target_resolution, factor, min_res, max_res,
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-2000:]
        raise RuntimeError(f"SeedVR2 upscale failed (exit {result.returncode}):\n{stderr_tail}")

    upscaled: dict[str, np.ndarray] = {}
    for name in faces:
        out_path = output_dir / f"{name}.png"
        if not out_path.exists():
            raise FileNotFoundError(f"SeedVR2 did not produce expected output: {out_path}")
        with Image.open(out_path) as img:
            rgb = img.convert("RGB")
            if rgb.size != (expected_image_width, expected_image_height):
                LOGGER.warning(
                    "SeedVR2 returned %dx%d for %s; resizing to expected %dx%d to preserve face aspect ratio.",
                    rgb.width,
                    rgb.height,
                    name,
                    expected_image_width,
                    expected_image_height,
                )
                rgb = rgb.resize((expected_image_width, expected_image_height), Image.Resampling.LANCZOS)
            upscaled[name] = np.asarray(rgb).copy()

    first_upscaled = upscaled[next(iter(upscaled))]
    new_image_height, new_image_width = first_upscaled.shape[:2]
    LOGGER.info("SeedVR2 upscale complete. New face size: %dx%d", new_image_width, new_image_height)
    return upscaled, new_image_width, new_image_height


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def load_torch_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dictionary: {path}")
    return checkpoint


def resolve_optional_path(path_value: str | Path | None) -> Path | None:
    if path_value is None or path_value == "" or path_value is False:
        return None
    path_input: str | Path = path_value if isinstance(path_value, Path) else str(path_value)
    path = Path(path_input)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def resolve_da360_alignment_enabled(args: argparse.Namespace, config: dict) -> bool:
    explicit = getattr(args, "enable_da360_alignment", None)
    if explicit is not None:
        return bool(explicit)
    if getattr(args, "disable_da360_alignment", False):
        return False
    return bool(config.get("default_enable_da360_alignment", True))


def normalize_alignment_mode(mode: str | None) -> str:
    normalized = str(mode or "").strip().lower()
    if not normalized:
        return "da360"
    aliases = {
        "off": "none",
        "disabled": "none",
        "false": "none",
        "depth": "da360",
        "da360_depth": "da360",
        "overlap_scale": "overlap",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_ALIGNMENT_MODES:
        raise ValueError(f"Unsupported alignment mode: {mode}")
    return normalized


def resolve_alignment_mode(args: argparse.Namespace, config: dict) -> str:
    explicit_mode = getattr(args, "alignment_mode", None)
    if explicit_mode not in {None, ""}:
        return normalize_alignment_mode(explicit_mode)
    explicit_legacy = getattr(args, "enable_da360_alignment", None)
    if explicit_legacy is not None:
        return "da360" if bool(explicit_legacy) else "none"
    if getattr(args, "disable_da360_alignment", False):
        return "none"
    config_mode = config.get("default_alignment_mode")
    if config_mode not in {None, ""}:
        return normalize_alignment_mode(config_mode)
    return "da360" if bool(config.get("default_enable_da360_alignment", True)) else "none"


def resolve_da360_checkpoint_path(args: argparse.Namespace, config: dict) -> Path:
    configured = getattr(args, "da360_checkpoint", None)
    if configured is not None:
        resolved = resolve_optional_path(configured)
        if resolved is None:
            raise ValueError("DA360 alignment is enabled but no DA360 checkpoint path was provided.")
        return resolved
    config_value = config.get("default_da360_checkpoint")
    resolved = resolve_optional_path(config_value)
    if resolved is not None:
        return resolved
    return DEFAULT_DA360_CHECKPOINT_PATH


def ensure_da360_import_path() -> None:
    if not LOCAL_DA360_ROOT.exists():
        raise FileNotFoundError(
            f"DA360 source directory not found: {LOCAL_DA360_ROOT}. Clone or vendor the DA360 repo first."
        )
    da360_path = str(LOCAL_DA360_ROOT)
    if da360_path not in sys.path:
        sys.path.insert(0, da360_path)


def register_optional_image_plugins() -> None:
    if register_heif_opener is not None:
        register_heif_opener()


def resolve_device(requested: str) -> torch.device:
    if requested == "default":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    return torch.device(requested)


def configure_cuda_inference() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def warmup_sharp_predictor(predictor: torch.nn.Module, device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 1536, 1536, device=device, dtype=torch.float32)
            dummy_disparity = torch.ones(1, device=device, dtype=torch.float32)
            predictor(dummy_image, dummy_disparity)
        torch.cuda.synchronize(device)
    except Exception as exc:
        LOGGER.debug("Skipping SHARP CUDA warmup after failure: %s", exc)


def load_panorama(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(f"Unsupported input format: {suffix}")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        rgb = image.convert("RGB")
        array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an RGB panorama, got shape {array.shape}")
    return array


def load_input_panorama(path: Path) -> np.ndarray:
    panorama = load_panorama(path)
    return np.ascontiguousarray(panorama)


def validate_equirectangular_shape(image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    if width != height * 2:
        raise ValueError(
            "This implementation currently expects a stitched equirectangular panorama with "
            f"2:1 aspect ratio. Received {width}x{height}."
        )
    return width, height


def resolve_side_count(requested: int, config: dict) -> int:
    if requested and requested > 0:
        side_count = requested
    else:
        side_count = int(config.get("default_side_count", 4))
    if side_count < 2:
        raise ValueError("side_count must be at least 2.")
    return side_count


def resolve_face_size(requested: int, panorama_width: int, side_count: int, config: dict) -> int:
    if requested and requested > 0:
        return requested
    mode = str(config.get("default_face_size_mode", "width_div_sides")).lower()
    if mode == "width_div_sides":
        return max(256, panorama_width // side_count)
    configured_size = int(config.get("default_face_size", 1536))
    return max(256, configured_size)


def make_horizon_view(index: int, side_count: int) -> FaceOrientation:
    yaw = (2.0 * np.pi * index) / side_count
    forward = (float(np.sin(yaw)), 0.0, float(np.cos(yaw)))
    right = (float(np.cos(yaw)), 0.0, float(-np.sin(yaw)))
    down = (0.0, 1.0, 0.0)

    if side_count == 2:
        names = ("front", "back")
        name = names[index]
    elif side_count == 4:
        names = ("front", "right", "back", "left")
        name = names[index]
    else:
        name = f"side_{index + 1:02d}"

    return FaceOrientation(name, right, down, forward)


def resolve_view_fov_degrees(side_count: int, config: dict) -> float:
    overlap_degrees = float(config.get("horizon_overlap_degrees", 10.0))
    span_degrees = 360.0 / side_count
    target_fov = span_degrees + overlap_degrees
    if side_count == 2 and target_fov >= 180.0:
        LOGGER.warning(
            "Two-view mode requested. Exact 180-degree pinhole views are not possible, so the view FOV is clamped below 180 degrees."
        )
    return min(170.0, target_fov)


def resolve_cutoff_height_percent(requested: float, config: dict) -> float:
    if requested not in {None, "", False}:
        cutoff_percent = float(requested)
    else:
        cutoff_percent = float(config.get("default_cutoff_height_percent", 0.0))
    if not (0.0 <= cutoff_percent <= 40.0):
        raise ValueError("cutoff_height_percent must be between 0 and 40.")
    return cutoff_percent


def resolve_extraction_image_height(panorama_height: int, cutoff_height_percent: float) -> int:
    base_height = max(64, int(round(panorama_height)))
    return max(64, int(round(base_height * (1.0 - (cutoff_height_percent / 100.0)))))


def resolve_extraction_focal_y(focal_x_px: float, image_width: int, image_height: int) -> float:
    # Taller rectangular slices keep the same angular density vertically by scaling fy with the output aspect ratio.
    return float(focal_x_px) * (float(image_height) / float(max(1, image_width)))


def build_extraction_layout(
    face_size: int,
    panorama_height: int,
    side_count: int,
    cutoff_height_percent: float,
    config: dict,
) -> ExtractionLayout:
    span_degrees = 360.0 / side_count
    view_fov_degrees = resolve_view_fov_degrees(side_count, config)
    if not (45.0 <= view_fov_degrees < 179.0):
        raise ValueError("Resolved view FOV must be between 45 and 179 degrees.")
    image_width = max(face_size, int(round(face_size * (view_fov_degrees / span_degrees))))
    focal_px = (image_width / 2.0) / np.tan(np.deg2rad(view_fov_degrees) / 2.0)
    image_height = resolve_extraction_image_height(panorama_height, cutoff_height_percent)
    focal_y_px = resolve_extraction_focal_y(focal_px, image_width, image_height)
    views = tuple(make_horizon_view(index, side_count) for index in range(side_count))
    return ExtractionLayout(f"horizon{side_count}", views, focal_px, focal_y_px, image_width, image_height)


def filter_gaussians_by_view_border(
    gaussians: Gaussians3D,
    horizontal_border_degrees: float,
    vertical_border_degrees: float | None = None,
) -> Gaussians3D:
    from sharp.utils.gaussians import Gaussians3D

    if horizontal_border_degrees >= 179.0:
        horizontal_limit = float("inf")
    else:
        half_horizontal = np.deg2rad(horizontal_border_degrees / 2.0)
        horizontal_limit = float(np.tan(half_horizontal))

    mean_vectors = gaussians.mean_vectors
    depth = mean_vectors[..., 2]
    horizontal_ratio = torch.abs(mean_vectors[..., 0]) / torch.clamp(depth, min=1e-6)
    keep_mask = (depth > 0.0) & (horizontal_ratio <= horizontal_limit)

    if vertical_border_degrees is not None and vertical_border_degrees > 0.0:
        if vertical_border_degrees >= 179.0:
            vertical_limit = float("inf")
        else:
            half_vertical = np.deg2rad(vertical_border_degrees / 2.0)
            vertical_limit = float(np.tan(half_vertical))
        vertical_ratio = torch.abs(mean_vectors[..., 1]) / torch.clamp(depth, min=1e-6)
        keep_mask = keep_mask & (vertical_ratio <= vertical_limit)

    kept_count = int(keep_mask.sum().item())
    total_count = int(keep_mask.numel())
    if kept_count == 0:
        raise ValueError("View-border clipping removed all predicted Gaussians for one view.")
    if kept_count == total_count:
        return gaussians

    LOGGER.info("Clipped %d of %d Gaussians outside the configured per-view border.", total_count - kept_count, total_count)
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, keep_mask[0], :],
        singular_values=gaussians.singular_values[:, keep_mask[0], :],
        quaternions=gaussians.quaternions[:, keep_mask[0], :],
        colors=gaussians.colors[:, keep_mask[0], ...],
        opacities=gaussians.opacities[:, keep_mask[0], ...],
    )


def scale_gaussians(gaussians: Gaussians3D, scale_factor: float) -> Gaussians3D:
    from sharp.utils.gaussians import Gaussians3D

    if abs(scale_factor - 1.0) < 1e-5:
        return gaussians
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors * scale_factor,
        singular_values=gaussians.singular_values * scale_factor,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def apply_per_point_scales(gaussians: Gaussians3D, per_point_scale: np.ndarray) -> Gaussians3D:
    from sharp.utils.gaussians import Gaussians3D

    scale_values = np.asarray(per_point_scale, dtype=np.float32).reshape(-1)
    if scale_values.shape[0] != int(gaussians.mean_vectors.shape[1]):
        raise ValueError("Per-point scale count does not match the Gaussian count.")
    scale_t = torch.from_numpy(scale_values).to(device=gaussians.mean_vectors.device, dtype=gaussians.mean_vectors.dtype).unsqueeze(0).unsqueeze(-1)
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors * scale_t,
        singular_values=gaussians.singular_values * scale_t,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def wrap_angle_degrees(angle_degrees: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle_degrees) + 180.0) % 360.0 - 180.0


def compute_face_world_polar_arrays(
    gaussians: Gaussians3D,
    view: FaceOrientation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_vectors = gaussians.mean_vectors[0].detach().cpu().numpy().astype(np.float32)
    world_vectors = mean_vectors @ view.rotation_matrix.T
    radial = np.linalg.norm(world_vectors, axis=1)
    horizontal = np.linalg.norm(world_vectors[:, (0, 2)], axis=1)
    yaw_deg = np.rad2deg(np.arctan2(world_vectors[:, 0], world_vectors[:, 2])).astype(np.float32)
    pitch_deg = np.rad2deg(np.arctan2(world_vectors[:, 1], np.clip(horizontal, 1e-6, None))).astype(np.float32)
    valid = np.isfinite(radial) & (radial > 1e-6) & np.isfinite(yaw_deg) & np.isfinite(pitch_deg)
    return yaw_deg, pitch_deg, radial.astype(np.float32), valid


def compute_face_world_polar_stats(
    gaussians: Gaussians3D,
    view: FaceOrientation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw_deg, pitch_deg, radial, valid = compute_face_world_polar_arrays(gaussians, view)
    return yaw_deg[valid], pitch_deg[valid], radial[valid]


def smooth_overlap_grid(grid: np.ndarray, passes: int = 1) -> np.ndarray:
    result = np.asarray(grid, dtype=np.float32)
    for _ in range(max(0, int(passes))):
        padded = np.pad(result, ((1, 1), (1, 1)), mode="edge")
        result = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ).astype(np.float32) / 9.0
    return result


def compute_overlap_pair_bins(
    left_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    right_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    boundary_yaw_deg: float,
    overlap_degrees: float,
    yaw_bin_count: int = OVERLAP_ALIGNMENT_YAW_BINS,
    pitch_bin_count: int = OVERLAP_ALIGNMENT_PITCH_BINS,
    min_points_per_bin: int = OVERLAP_ALIGNMENT_MIN_BIN_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    half_overlap = overlap_degrees / 2.0
    left_yaw, left_pitch, left_radial = left_stats
    right_yaw, right_pitch, right_radial = right_stats
    left_delta = np.asarray(wrap_angle_degrees(left_yaw - boundary_yaw_deg), dtype=np.float32)
    right_delta = np.asarray(wrap_angle_degrees(right_yaw - boundary_yaw_deg), dtype=np.float32)
    left_keep = np.abs(left_delta) <= half_overlap
    right_keep = np.abs(right_delta) <= half_overlap
    if int(np.count_nonzero(left_keep)) < min_points_per_bin or int(np.count_nonzero(right_keep)) < min_points_per_bin:
        return None

    left_delta = left_delta[left_keep]
    right_delta = right_delta[right_keep]
    left_pitch = np.asarray(left_pitch[left_keep], dtype=np.float32)
    right_pitch = np.asarray(right_pitch[right_keep], dtype=np.float32)
    left_radial = np.asarray(left_radial[left_keep], dtype=np.float32)
    right_radial = np.asarray(right_radial[right_keep], dtype=np.float32)
    combined_pitch = np.concatenate((left_pitch, right_pitch))
    if combined_pitch.size == 0:
        return None
    pitch_low, pitch_high = np.quantile(combined_pitch, [0.05, 0.95])
    if float(pitch_high - pitch_low) < 1e-3:
        pitch_low = float(np.min(combined_pitch)) - 0.5
        pitch_high = float(np.max(combined_pitch)) + 0.5

    yaw_edges = np.linspace(-half_overlap, half_overlap, yaw_bin_count + 1, dtype=np.float32)
    pitch_edges = np.linspace(float(pitch_low), float(pitch_high), pitch_bin_count + 1, dtype=np.float32)
    left_medians, left_counts = build_overlap_bin_medians(left_delta, left_pitch, left_radial, yaw_edges, pitch_edges, min_points_per_bin)
    right_medians, right_counts = build_overlap_bin_medians(right_delta, right_pitch, right_radial, yaw_edges, pitch_edges, min_points_per_bin)
    return left_medians, left_counts, right_medians, right_counts, yaw_edges, pitch_edges


def compute_clipped_edge_pair_bins(
    left_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    right_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    boundary_yaw_deg: float,
    edge_width_degrees: float,
    yaw_bin_count: int = OVERLAP_ALIGNMENT_YAW_BINS,
    pitch_bin_count: int = OVERLAP_ALIGNMENT_PITCH_BINS,
    min_points_per_bin: int = OVERLAP_ALIGNMENT_MIN_BIN_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    left_yaw, left_pitch, left_radial = left_stats
    right_yaw, right_pitch, right_radial = right_stats
    left_delta = np.asarray(wrap_angle_degrees(left_yaw - boundary_yaw_deg), dtype=np.float32)
    right_delta = np.asarray(wrap_angle_degrees(right_yaw - boundary_yaw_deg), dtype=np.float32)
    left_keep = (left_delta >= -edge_width_degrees) & (left_delta <= 0.0)
    right_keep = (right_delta >= 0.0) & (right_delta <= edge_width_degrees)
    if int(np.count_nonzero(left_keep)) < min_points_per_bin or int(np.count_nonzero(right_keep)) < min_points_per_bin:
        return None

    left_edge_distance = np.asarray(-left_delta[left_keep], dtype=np.float32)
    right_edge_distance = np.asarray(right_delta[right_keep], dtype=np.float32)
    left_pitch = np.asarray(left_pitch[left_keep], dtype=np.float32)
    right_pitch = np.asarray(right_pitch[right_keep], dtype=np.float32)
    left_radial = np.asarray(left_radial[left_keep], dtype=np.float32)
    right_radial = np.asarray(right_radial[right_keep], dtype=np.float32)
    combined_pitch = np.concatenate((left_pitch, right_pitch))
    if combined_pitch.size == 0:
        return None
    pitch_low, pitch_high = np.quantile(combined_pitch, [0.05, 0.95])
    if float(pitch_high - pitch_low) < 1e-3:
        pitch_low = float(np.min(combined_pitch)) - 0.5
        pitch_high = float(np.max(combined_pitch)) + 0.5

    yaw_edges = np.linspace(0.0, edge_width_degrees, yaw_bin_count + 1, dtype=np.float32)
    pitch_edges = np.linspace(float(pitch_low), float(pitch_high), pitch_bin_count + 1, dtype=np.float32)
    left_medians, left_counts = build_overlap_bin_medians(left_edge_distance, left_pitch, left_radial, yaw_edges, pitch_edges, min_points_per_bin)
    right_medians, right_counts = build_overlap_bin_medians(right_edge_distance, right_pitch, right_radial, yaw_edges, pitch_edges, min_points_per_bin)
    return left_medians, left_counts, right_medians, right_counts, yaw_edges, pitch_edges


def build_overlap_bin_medians(
    yaw_delta_deg: np.ndarray,
    pitch_deg: np.ndarray,
    radial: np.ndarray,
    yaw_edges: np.ndarray,
    pitch_edges: np.ndarray,
    min_points_per_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    yaw_bin_count = len(yaw_edges) - 1
    pitch_bin_count = len(pitch_edges) - 1
    medians = np.full((pitch_bin_count, yaw_bin_count), np.nan, dtype=np.float32)
    counts = np.zeros((pitch_bin_count, yaw_bin_count), dtype=np.int32)
    yaw_indices = np.digitize(yaw_delta_deg, yaw_edges) - 1
    pitch_indices = np.digitize(pitch_deg, pitch_edges) - 1
    valid = (
        (yaw_indices >= 0) & (yaw_indices < yaw_bin_count)
        & (pitch_indices >= 0) & (pitch_indices < pitch_bin_count)
        & np.isfinite(radial) & (radial > 1e-6)
    )
    if not np.any(valid):
        return medians, counts

    flat_indices = pitch_indices[valid] * yaw_bin_count + yaw_indices[valid]
    flat_values = radial[valid]
    for flat_index in np.unique(flat_indices):
        cell_values = flat_values[flat_indices == flat_index]
        pitch_index = int(flat_index // yaw_bin_count)
        yaw_index = int(flat_index % yaw_bin_count)
        counts[pitch_index, yaw_index] = int(cell_values.size)
        if cell_values.size >= min_points_per_bin:
            medians[pitch_index, yaw_index] = float(np.median(cell_values))
    return medians, counts


def estimate_overlap_pair_scale_ratio(
    left_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    right_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    boundary_yaw_deg: float,
    overlap_degrees: float,
    yaw_bin_count: int = OVERLAP_ALIGNMENT_YAW_BINS,
    pitch_bin_count: int = OVERLAP_ALIGNMENT_PITCH_BINS,
    min_points_per_bin: int = OVERLAP_ALIGNMENT_MIN_BIN_POINTS,
    min_shared_bins: int = OVERLAP_ALIGNMENT_MIN_SHARED_BINS,
) -> tuple[float, int] | None:
    pair_bins = compute_overlap_pair_bins(
        left_stats,
        right_stats,
        boundary_yaw_deg,
        overlap_degrees,
        yaw_bin_count=yaw_bin_count,
        pitch_bin_count=pitch_bin_count,
        min_points_per_bin=min_points_per_bin,
    )
    if pair_bins is None:
        return None
    left_medians, left_counts, right_medians, right_counts, _yaw_edges, _pitch_edges = pair_bins
    shared = np.isfinite(left_medians) & np.isfinite(right_medians)
    shared_count = int(shared.sum())
    if shared_count < min_shared_bins:
        return None

    ratios = np.clip(left_medians[shared] / np.maximum(right_medians[shared], 1e-6), 1e-6, None)
    log_ratios = np.log(ratios.astype(np.float64))
    weights = np.minimum(left_counts[shared], right_counts[shared]).astype(np.float64)
    if log_ratios.size >= 6:
        low, high = np.quantile(log_ratios, [0.1, 0.9])
        trimmed = (log_ratios >= low) & (log_ratios <= high)
        if np.any(trimmed):
            log_ratios = log_ratios[trimmed]
            weights = weights[trimmed]
    if log_ratios.size == 0:
        return None
    log_ratio = weighted_median(log_ratios.astype(np.float32), np.maximum(weights, 1e-3).astype(np.float32))
    return float(np.exp(log_ratio)), shared_count


def build_overlap_pair_boundary_fields(
    left_name: str,
    right_name: str,
    left_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    right_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    boundary_yaw_deg: float,
    overlap_degrees: float,
    left_global_scale: float,
    right_global_scale: float,
    yaw_bin_count: int = OVERLAP_ALIGNMENT_YAW_BINS,
    pitch_bin_count: int = OVERLAP_ALIGNMENT_PITCH_BINS,
    min_points_per_bin: int = OVERLAP_ALIGNMENT_MIN_BIN_POINTS,
    min_shared_bins: int = OVERLAP_ALIGNMENT_MIN_SHARED_BINS,
) -> tuple[OverlapBoundaryField, OverlapBoundaryField, float, int] | None:
    pair_bins = compute_overlap_pair_bins(
        left_stats,
        right_stats,
        boundary_yaw_deg,
        overlap_degrees,
        yaw_bin_count=yaw_bin_count,
        pitch_bin_count=pitch_bin_count,
        min_points_per_bin=min_points_per_bin,
    )
    if pair_bins is None:
        return None
    left_medians, left_counts, right_medians, right_counts, yaw_edges, pitch_edges = pair_bins
    return build_boundary_fields_from_pair_bins(
        left_name,
        right_name,
        left_medians,
        left_counts,
        right_medians,
        right_counts,
        yaw_edges,
        pitch_edges,
        overlap_degrees / 2.0,
        boundary_yaw_deg,
        left_global_scale,
        right_global_scale,
        min_shared_bins=min_shared_bins,
    )


def build_boundary_fields_from_pair_bins(
    left_name: str,
    right_name: str,
    left_medians: np.ndarray,
    left_counts: np.ndarray,
    right_medians: np.ndarray,
    right_counts: np.ndarray,
    yaw_edges: np.ndarray,
    pitch_edges: np.ndarray,
    half_overlap: float,
    boundary_yaw_deg: float,
    left_global_scale: float,
    right_global_scale: float,
    min_shared_bins: int = OVERLAP_ALIGNMENT_MIN_SHARED_BINS,
) -> tuple[OverlapBoundaryField, OverlapBoundaryField, float, int] | None:
    shared = np.isfinite(left_medians) & np.isfinite(right_medians)
    shared_count = int(shared.sum())
    if shared_count < min_shared_bins:
        return None

    scaled_left = left_medians * float(left_global_scale)
    scaled_right = right_medians * float(right_global_scale)
    shared_log_ratios = np.log(np.clip(scaled_left[shared] / np.maximum(scaled_right[shared], 1e-6), 1e-6, None).astype(np.float64))
    weights = np.minimum(left_counts[shared], right_counts[shared]).astype(np.float64)
    trimmed_log_ratios = shared_log_ratios
    trimmed_weights = weights
    if shared_log_ratios.size >= 6:
        low, high = np.quantile(shared_log_ratios, [0.1, 0.9])
        trimmed = (shared_log_ratios >= low) & (shared_log_ratios <= high)
        if np.any(trimmed):
            trimmed_log_ratios = shared_log_ratios[trimmed]
            trimmed_weights = weights[trimmed]
    if trimmed_log_ratios.size == 0:
        return None

    fallback_log_ratio = weighted_median(
        trimmed_log_ratios.astype(np.float32),
        np.maximum(trimmed_weights, 1e-3).astype(np.float32),
    )
    ratio_log_grid = np.full_like(left_medians, fallback_log_ratio, dtype=np.float32)
    ratio_log_grid[shared] = shared_log_ratios.astype(np.float32)
    ratio_log_grid = np.clip(smooth_overlap_grid(ratio_log_grid, passes=1), np.log(0.25), np.log(4.0))

    left_log_grid = -0.5 * ratio_log_grid
    right_log_grid = 0.5 * ratio_log_grid
    return (
        OverlapBoundaryField(boundary_yaw_deg, half_overlap, yaw_edges, pitch_edges, left_log_grid, float(-0.5 * fallback_log_ratio), right_name),
        OverlapBoundaryField(boundary_yaw_deg, half_overlap, yaw_edges, pitch_edges, right_log_grid, float(0.5 * fallback_log_ratio), left_name),
        float(np.exp(fallback_log_ratio)),
        shared_count,
    )


def solve_overlap_alignment_scales(
    face_count: int,
    constraints: Sequence[tuple[int, int, float, int]],
) -> np.ndarray:
    if face_count <= 0:
        return np.ones(0, dtype=np.float32)
    if not constraints:
        return np.ones(face_count, dtype=np.float32)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for left_index, right_index, ratio, weight in constraints:
        row = np.zeros(face_count, dtype=np.float64)
        scaled_weight = float(np.sqrt(max(1.0, float(weight))))
        row[left_index] = -scaled_weight
        row[right_index] = scaled_weight
        rows.append(row)
        rhs.append(np.log(max(ratio, 1e-6)) * scaled_weight)
    rows.append(np.ones(face_count, dtype=np.float64))
    rhs.append(0.0)
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(rhs, dtype=np.float64), rcond=None)
    solution -= float(np.mean(solution))
    return np.exp(solution).astype(np.float32)


def resolve_overlap_alignment_scales(
    face_gaussians: Mapping[str, Gaussians3D],
    layout: ExtractionLayout,
    config: Mapping[str, Any],
) -> tuple[dict[str, float], list[tuple[str, str, float, int]]]:
    side_count = max(1, len(layout.views))
    if side_count == 1:
        return {layout.views[0].name: 1.0}, []

    span_degrees = 360.0 / side_count
    overlap_degrees = max(0.0, resolve_view_fov_degrees(side_count, dict(config)) - span_degrees)
    if overlap_degrees <= 0.25:
        LOGGER.warning("Overlap alignment requested, but the configured overlap is too small (%.2f deg).", overlap_degrees)
        return {view.name: 1.0 for view in layout.views}, []

    polar_stats = {
        view.name: compute_face_world_polar_stats(face_gaussians[view.name], view)
        for view in layout.views
    }
    constraints: list[tuple[int, int, float, int]] = []
    diagnostics: list[tuple[str, str, float, int]] = []
    for index, view in enumerate(layout.views):
        next_index = (index + 1) % side_count
        next_view = layout.views[next_index]
        center_yaw_deg = float(np.rad2deg(np.arctan2(view.forward[0], view.forward[2])))
        boundary_yaw_deg = float(wrap_angle_degrees(center_yaw_deg + (span_degrees / 2.0)))
        estimate = estimate_overlap_pair_scale_ratio(
            polar_stats[view.name],
            polar_stats[next_view.name],
            boundary_yaw_deg,
            overlap_degrees,
        )
        if estimate is None:
            LOGGER.warning("Overlap alignment could not derive a reliable constraint between %s and %s.", view.name, next_view.name)
            continue
        ratio, shared_bins = estimate
        constraints.append((index, next_index, ratio, shared_bins))
        diagnostics.append((view.name, next_view.name, ratio, shared_bins))

    scales = solve_overlap_alignment_scales(side_count, constraints)
    return {view.name: float(scales[index]) for index, view in enumerate(layout.views)}, diagnostics


def resolve_overlap_alignment_fields(
    face_gaussians: Mapping[str, Gaussians3D],
    layout: ExtractionLayout,
    config: Mapping[str, Any],
    global_scales: Mapping[str, float],
) -> tuple[dict[str, dict[str, OverlapBoundaryField]], list[tuple[str, str, float, int]]]:
    boundary_fields = {view.name: {} for view in layout.views}
    side_count = max(1, len(layout.views))
    if side_count == 1:
        return boundary_fields, []

    span_degrees = 360.0 / side_count
    overlap_degrees = max(0.0, resolve_view_fov_degrees(side_count, dict(config)) - span_degrees)
    if overlap_degrees <= 0.25:
        return boundary_fields, []

    polar_stats = {
        view.name: compute_face_world_polar_stats(face_gaussians[view.name], view)
        for view in layout.views
    }
    diagnostics: list[tuple[str, str, float, int]] = []
    for index, view in enumerate(layout.views):
        next_index = (index + 1) % side_count
        next_view = layout.views[next_index]
        center_yaw_deg = float(np.rad2deg(np.arctan2(view.forward[0], view.forward[2])))
        boundary_yaw_deg = float(wrap_angle_degrees(center_yaw_deg + (span_degrees / 2.0)))
        pair_fields = build_overlap_pair_boundary_fields(
            view.name,
            next_view.name,
            polar_stats[view.name],
            polar_stats[next_view.name],
            boundary_yaw_deg,
            overlap_degrees,
            float(global_scales.get(view.name, 1.0)),
            float(global_scales.get(next_view.name, 1.0)),
        )
        if pair_fields is None:
            LOGGER.warning("Overlap seam field could not derive a reliable local correction between %s and %s.", view.name, next_view.name)
            continue
        left_field, right_field, residual_ratio, shared_bins = pair_fields
        boundary_fields[view.name]["right"] = left_field
        boundary_fields[next_view.name]["left"] = right_field
        diagnostics.append((view.name, next_view.name, residual_ratio, shared_bins))
    return boundary_fields, diagnostics


def resolve_clipped_edge_alignment_fields(
    face_gaussians: Mapping[str, Gaussians3D],
    layout: ExtractionLayout,
    config: Mapping[str, Any],
    global_scales: Mapping[str, float],
) -> tuple[dict[str, dict[str, OverlapBoundaryField]], list[tuple[str, str, float, int]]]:
    boundary_fields = {view.name: {} for view in layout.views}
    side_count = max(1, len(layout.views))
    if side_count == 1:
        return boundary_fields, []

    span_degrees = 360.0 / side_count
    edge_width_degrees = max(0.0, (resolve_view_fov_degrees(side_count, dict(config)) - span_degrees) / 2.0)
    if edge_width_degrees <= 0.125:
        return boundary_fields, []

    polar_stats = {
        view.name: compute_face_world_polar_stats(face_gaussians[view.name], view)
        for view in layout.views
    }
    diagnostics: list[tuple[str, str, float, int]] = []
    for index, view in enumerate(layout.views):
        next_index = (index + 1) % side_count
        next_view = layout.views[next_index]
        center_yaw_deg = float(np.rad2deg(np.arctan2(view.forward[0], view.forward[2])))
        boundary_yaw_deg = float(wrap_angle_degrees(center_yaw_deg + (span_degrees / 2.0)))
        pair_bins = compute_clipped_edge_pair_bins(
            polar_stats[view.name],
            polar_stats[next_view.name],
            boundary_yaw_deg,
            edge_width_degrees,
        )
        if pair_bins is None:
            LOGGER.warning("Clipped-edge refinement could not derive a reliable local correction between %s and %s.", view.name, next_view.name)
            continue
        left_medians, left_counts, right_medians, right_counts, yaw_edges, pitch_edges = pair_bins
        pair_fields = build_boundary_fields_from_pair_bins(
            view.name,
            next_view.name,
            left_medians,
            left_counts,
            right_medians,
            right_counts,
            yaw_edges,
            pitch_edges,
            edge_width_degrees,
            boundary_yaw_deg,
            float(global_scales.get(view.name, 1.0)),
            float(global_scales.get(next_view.name, 1.0)),
        )
        if pair_fields is None:
            LOGGER.warning("Clipped-edge refinement rejected sparse correction data between %s and %s.", view.name, next_view.name)
            continue
        left_field, right_field, residual_ratio, shared_bins = pair_fields
        boundary_fields[view.name]["right"] = left_field
        boundary_fields[next_view.name]["left"] = right_field
        diagnostics.append((view.name, next_view.name, residual_ratio, shared_bins))
    return boundary_fields, diagnostics


def sample_overlap_boundary_field(
    field: OverlapBoundaryField,
    yaw_delta_deg: np.ndarray,
    pitch_deg: np.ndarray,
) -> np.ndarray:
    yaw_cell_width = float(field.yaw_edges[-1] - field.yaw_edges[0]) / float(max(1, len(field.yaw_edges) - 1))
    pitch_cell_width = float(field.pitch_edges[-1] - field.pitch_edges[0]) / float(max(1, len(field.pitch_edges) - 1))
    yaw_cont = (np.asarray(yaw_delta_deg, dtype=np.float32) - float(field.yaw_edges[0])) / max(yaw_cell_width, 1e-6) - 0.5
    pitch_cont = (np.asarray(pitch_deg, dtype=np.float32) - float(field.pitch_edges[0])) / max(pitch_cell_width, 1e-6) - 0.5

    grid_height, grid_width = field.log_scale_grid.shape
    x0 = np.clip(np.floor(yaw_cont).astype(np.int32), 0, grid_width - 1)
    y0 = np.clip(np.floor(pitch_cont).astype(np.int32), 0, grid_height - 1)
    x1 = np.clip(x0 + 1, 0, grid_width - 1)
    y1 = np.clip(y0 + 1, 0, grid_height - 1)
    wx = np.clip(yaw_cont - x0, 0.0, 1.0).astype(np.float32)
    wy = np.clip(pitch_cont - y0, 0.0, 1.0).astype(np.float32)

    s00 = field.log_scale_grid[y0, x0]
    s01 = field.log_scale_grid[y0, x1]
    s10 = field.log_scale_grid[y1, x0]
    s11 = field.log_scale_grid[y1, x1]
    return (
        s00 * (1.0 - wx) * (1.0 - wy)
        + s01 * wx * (1.0 - wy)
        + s10 * (1.0 - wx) * wy
        + s11 * wx * wy
    ).astype(np.float32)


def apply_overlap_alignment_fields(
    gaussians: Gaussians3D,
    view: FaceOrientation,
    boundary_fields: Mapping[str, OverlapBoundaryField],
) -> tuple[Gaussians3D, int]:
    if not boundary_fields:
        return gaussians, 0

    yaw_deg, pitch_deg, _radial, valid = compute_face_world_polar_arrays(gaussians, view)
    point_count = yaw_deg.shape[0]
    log_scale_sum = np.zeros(point_count, dtype=np.float32)
    weight_sum = np.zeros(point_count, dtype=np.float32)
    touched_mask = np.zeros(point_count, dtype=bool)

    for field in boundary_fields.values():
        yaw_delta = np.asarray(wrap_angle_degrees(yaw_deg - field.boundary_yaw_deg), dtype=np.float32)
        seam_mask = valid & np.isfinite(yaw_delta) & np.isfinite(pitch_deg) & (np.abs(yaw_delta) <= field.half_overlap_degrees)
        if not np.any(seam_mask):
            continue
        sampled_log_scale = sample_overlap_boundary_field(field, yaw_delta[seam_mask], pitch_deg[seam_mask])
        seam_weight = np.clip(1.0 - (np.abs(yaw_delta[seam_mask]) / max(field.half_overlap_degrees, 1e-6)), 0.0, 1.0).astype(np.float32)
        seam_weight = np.maximum(seam_weight * seam_weight, 1e-3)
        log_scale_sum[seam_mask] += sampled_log_scale * seam_weight
        weight_sum[seam_mask] += 1.0
        touched_mask[seam_mask] = True

    if not np.any(touched_mask):
        return gaussians, 0

    per_point_scale = np.ones(point_count, dtype=np.float32)
    per_point_scale[touched_mask] = np.exp(np.clip(log_scale_sum[touched_mask] / np.maximum(weight_sum[touched_mask], 1e-6), np.log(0.25), np.log(4.0)))
    per_point_scale = np.clip(per_point_scale, 0.25, 4.0)
    return apply_per_point_scales(gaussians, per_point_scale), int(touched_mask.sum())


def apply_overlap_global_and_local_alignment(
    gaussians: Gaussians3D,
    view: FaceOrientation,
    global_scale: float,
    boundary_fields: Mapping[str, OverlapBoundaryField],
) -> tuple[Gaussians3D, int]:
    gaussians = scale_gaussians(gaussians, global_scale)
    return apply_overlap_alignment_fields(gaussians, view, boundary_fields)


def smooth_curve_values(values: np.ndarray, passes: int = 1) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    for _ in range(max(0, int(passes))):
        padded = np.pad(result, (1, 1), mode="edge")
        result = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]).astype(np.float32) / 4.0
    return result


def build_depth_layer_standardization_curve(
    log_radius_samples: np.ndarray,
    log_scale_samples: np.ndarray,
    weights: np.ndarray,
    bin_count: int = OVERLAP_DEPTH_LAYER_BINS,
    min_samples_per_bin: int = OVERLAP_DEPTH_LAYER_MIN_SAMPLES,
) -> DepthLayerCurve | None:
    if log_radius_samples.size < min_samples_per_bin or log_scale_samples.size != log_radius_samples.size:
        return None

    sample_weights = np.maximum(np.asarray(weights, dtype=np.float32).reshape(-1), 1e-3)
    log_radius_samples = np.asarray(log_radius_samples, dtype=np.float32).reshape(-1)
    log_scale_samples = np.asarray(log_scale_samples, dtype=np.float32).reshape(-1)
    radius_low, radius_high = np.quantile(log_radius_samples, [0.05, 0.95])
    if not np.isfinite(radius_low) or not np.isfinite(radius_high) or float(radius_high - radius_low) < 1e-3:
        return None

    edges = np.linspace(float(radius_low), float(radius_high), max(2, int(bin_count)) + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    baseline = weighted_median(log_scale_samples, sample_weights)
    knot_values = np.full(centers.shape, np.nan, dtype=np.float32)
    for index in range(centers.shape[0]):
        if index == centers.shape[0] - 1:
            in_bin = (log_radius_samples >= edges[index]) & (log_radius_samples <= edges[index + 1])
        else:
            in_bin = (log_radius_samples >= edges[index]) & (log_radius_samples < edges[index + 1])
        if int(in_bin.sum()) < min_samples_per_bin:
            continue
        knot_values[index] = weighted_median(log_scale_samples[in_bin], sample_weights[in_bin]) - baseline

    known = np.isfinite(knot_values)
    if not np.any(known):
        return None
    if int(known.sum()) == 1:
        knot_values[:] = float(knot_values[known][0])
    else:
        knot_values = np.interp(centers, centers[known], knot_values[known]).astype(np.float32)
    knot_values = np.clip(smooth_curve_values(knot_values, passes=1), np.log(0.5), np.log(2.0))
    return DepthLayerCurve(centers.astype(np.float32), knot_values.astype(np.float32), 0.0)


def sample_depth_layer_curve(curve: DepthLayerCurve, log_radius: np.ndarray) -> np.ndarray:
    return np.interp(
        np.asarray(log_radius, dtype=np.float32),
        curve.log_radius_knots,
        curve.log_scale_knots,
        left=float(curve.log_scale_knots[0]),
        right=float(curve.log_scale_knots[-1]),
    ).astype(np.float32)


def resolve_overlap_depth_layer_standardization(
    face_gaussians: Mapping[str, Gaussians3D],
    layout: ExtractionLayout,
    config: Mapping[str, Any],
) -> tuple[dict[str, DepthLayerCurve | None], list[tuple[str, int, float]]]:
    curves: dict[str, DepthLayerCurve | None] = {view.name: None for view in layout.views}
    diagnostics: list[tuple[str, int, float]] = []
    side_count = max(1, len(layout.views))
    if side_count == 1:
        return curves, diagnostics

    span_degrees = 360.0 / side_count
    edge_width_degrees = max(0.0, (resolve_view_fov_degrees(side_count, dict(config)) - span_degrees) / 2.0)
    if edge_width_degrees <= 0.125:
        return curves, diagnostics

    polar_stats = {
        view.name: compute_face_world_polar_stats(face_gaussians[view.name], view)
        for view in layout.views
    }
    per_face_log_radius: dict[str, list[np.ndarray]] = {view.name: [] for view in layout.views}
    per_face_log_scale: dict[str, list[np.ndarray]] = {view.name: [] for view in layout.views}
    per_face_weights: dict[str, list[np.ndarray]] = {view.name: [] for view in layout.views}

    for index, view in enumerate(layout.views):
        next_index = (index + 1) % side_count
        next_view = layout.views[next_index]
        center_yaw_deg = float(np.rad2deg(np.arctan2(view.forward[0], view.forward[2])))
        boundary_yaw_deg = float(wrap_angle_degrees(center_yaw_deg + (span_degrees / 2.0)))
        pair_bins = compute_clipped_edge_pair_bins(
            polar_stats[view.name],
            polar_stats[next_view.name],
            boundary_yaw_deg,
            edge_width_degrees,
        )
        if pair_bins is None:
            continue
        left_medians, left_counts, right_medians, right_counts, _yaw_edges, _pitch_edges = pair_bins
        shared = np.isfinite(left_medians) & np.isfinite(right_medians)
        if int(shared.sum()) < OVERLAP_ALIGNMENT_MIN_SHARED_BINS:
            continue

        left_shared = np.clip(left_medians[shared], 1e-6, None).astype(np.float32)
        right_shared = np.clip(right_medians[shared], 1e-6, None).astype(np.float32)
        weight_shared = np.minimum(left_counts[shared], right_counts[shared]).astype(np.float32)
        log_ratio = np.log(np.clip(left_shared / right_shared, 1e-6, None).astype(np.float64))
        if log_ratio.size >= 6:
            low, high = np.quantile(log_ratio, [0.1, 0.9])
            trimmed = (log_ratio >= low) & (log_ratio <= high)
            if np.any(trimmed):
                left_shared = left_shared[trimmed]
                right_shared = right_shared[trimmed]
                weight_shared = weight_shared[trimmed]
                log_ratio = log_ratio[trimmed]
        if log_ratio.size == 0:
            continue

        per_face_log_radius[view.name].append(np.log(left_shared).astype(np.float32))
        per_face_log_scale[view.name].append((-0.5 * log_ratio).astype(np.float32))
        per_face_weights[view.name].append(weight_shared.astype(np.float32))

        per_face_log_radius[next_view.name].append(np.log(right_shared).astype(np.float32))
        per_face_log_scale[next_view.name].append((0.5 * log_ratio).astype(np.float32))
        per_face_weights[next_view.name].append(weight_shared.astype(np.float32))

    for view in layout.views:
        name = view.name
        if not per_face_log_radius[name]:
            continue
        curve = build_depth_layer_standardization_curve(
            np.concatenate(per_face_log_radius[name]),
            np.concatenate(per_face_log_scale[name]),
            np.concatenate(per_face_weights[name]),
        )
        if curve is None:
            continue
        curves[name] = curve
        diagnostics.append((name, int(np.concatenate(per_face_log_radius[name]).size), float(np.exp(np.max(np.abs(curve.log_scale_knots))))))
    return curves, diagnostics


def apply_depth_layer_standardization_curve(
    gaussians: Gaussians3D,
    curve: DepthLayerCurve | None,
) -> tuple[Gaussians3D, int, float]:
    if curve is None:
        return gaussians, 0, 1.0

    radial = torch.norm(gaussians.mean_vectors, dim=-1)[0].detach().cpu().numpy().astype(np.float32)
    valid = np.isfinite(radial) & (radial > 1e-6)
    if not np.any(valid):
        return gaussians, 0, 1.0

    per_point_scale = np.ones(radial.shape[0], dtype=np.float32)
    per_point_scale[valid] = np.exp(sample_depth_layer_curve(curve, np.log(radial[valid]).astype(np.float32)))
    per_point_scale = np.clip(per_point_scale, 0.5, 2.0)
    valid_scales = per_point_scale[valid]
    median_scale = float(np.median(valid_scales)) if valid_scales.size else 1.0
    return apply_per_point_scales(gaussians, per_point_scale), int(valid.sum()), median_scale


def apply_overlap_alignment_pipeline(
    gaussians: Gaussians3D,
    view: FaceOrientation,
    global_scale: float,
    primary_boundary_fields: Mapping[str, OverlapBoundaryField],
    depth_layer_curve: DepthLayerCurve | None,
    residual_boundary_fields: Mapping[str, OverlapBoundaryField],
) -> tuple[Gaussians3D, dict[str, float | int]]:
    gaussians, primary_points = apply_overlap_global_and_local_alignment(
        gaussians,
        view,
        global_scale,
        primary_boundary_fields,
    )
    gaussians, depth_points, depth_median_scale = apply_depth_layer_standardization_curve(gaussians, depth_layer_curve)
    gaussians, residual_points = apply_overlap_alignment_fields(gaussians, view, residual_boundary_fields)
    return gaussians, {
        "primary_points": int(primary_points),
        "depth_points": int(depth_points),
        "depth_median_scale": float(depth_median_scale),
        "residual_points": int(residual_points),
    }


def align_gaussians_to_reference(
    gaussians: Gaussians3D,
    reference_disparity_view: np.ndarray,
    focal_x_px: float,
    focal_y_px: float,
    image_width: int,
    image_height: int,
    grid_resolution: int = 8,
    detail_weight: float = 0.0,
) -> tuple[Gaussians3D, float, int]:
    """Align Gaussian depths to DA360 using a smooth low-frequency scale field.

    Instead of overriding every Gaussian's depth with DA360 (which destroys
    SHARP's fine detail), we:
      1. Compute per-Gaussian ``da360_depth / sharp_depth`` ratios.
      2. Bin those ratios into a coarse spatial grid and take the
         robust median per cell.
      3. Bilinearly interpolate the coarse grid back to each Gaussian.
      4. Optionally blend the smooth grid scale with the per-point raw
         scale according to *detail_weight* (0 = fully smooth, 1 = fully
         per-point).

    *grid_resolution* controls the NxN grid size (higher = finer spatial
    alignment).  *detail_weight* 0-1 blends smooth vs per-point scales.

    Returns ``(aligned_gaussians, median_scale, sample_count)``.
    """
    from sharp.utils.gaussians import Gaussians3D

    grid_cells_x = max(1, int(grid_resolution))
    grid_cells_y = max(1, int(round(grid_cells_x * (image_height / max(1, image_width)))))

    mean_vectors = gaussians.mean_vectors  # (1, N, 3)
    mv_np = mean_vectors[0].detach().cpu().numpy().astype(np.float32)
    depth_z = mv_np[:, 2]
    radial = np.linalg.norm(mv_np, axis=1)

    valid = depth_z > 1e-6
    pixel_x = (mv_np[:, 0] / np.clip(depth_z, 1e-6, None)) * focal_x_px + (image_width / 2.0) - 0.5
    pixel_y = (mv_np[:, 1] / np.clip(depth_z, 1e-6, None)) * focal_y_px + (image_height / 2.0) - 0.5
    valid &= (pixel_x >= 0) & (pixel_x <= image_width - 1)
    valid &= (pixel_y >= 0) & (pixel_y <= image_height - 1)

    per_point_scale = np.ones(mv_np.shape[0], dtype=np.float32)
    median_scale = 1.0
    count = 0

    if int(valid.sum()) >= 64:
        ref_disp = bilinear_sample_scalar(
            reference_disparity_view, pixel_x[valid], pixel_y[valid],
        )
        ok = np.isfinite(ref_disp) & (ref_disp > 1e-6) & (radial[valid] > 1e-6)
        count = int(ok.sum())
        if count >= 64:
            ref_depth_ok = (1.0 / ref_disp[ok]).astype(np.float32)
            sharp_r_ok = radial[valid][ok]
            raw_scale = ref_depth_ok / sharp_r_ok

            # Global robust median for fallback and logging.
            lo, hi = np.quantile(raw_scale, [0.05, 0.95])
            trimmed = raw_scale[(raw_scale >= lo) & (raw_scale <= hi)]
            median_scale = (
                float(np.median(trimmed)) if trimmed.size > 0
                else float(np.median(raw_scale))
            )

            # --- Build coarse scale grid ---
            # Pixel coords of the ok-subset within the valid-subset.
            px_ok = pixel_x[valid][ok]
            py_ok = pixel_y[valid][ok]

            cell_width = image_width / grid_cells_x
            cell_height = image_height / grid_cells_y
            grid = np.full((grid_cells_y, grid_cells_x), median_scale, dtype=np.float32)
            for gy in range(grid_cells_y):
                for gx in range(grid_cells_x):
                    in_cell = (
                        (px_ok >= gx * cell_width) & (px_ok < (gx + 1) * cell_width)
                        & (py_ok >= gy * cell_height) & (py_ok < (gy + 1) * cell_height)
                    )
                    if int(in_cell.sum()) >= 8:
                        cell_scales = raw_scale[in_cell]
                        cl, ch = np.quantile(cell_scales, [0.1, 0.9])
                        cell_trimmed = cell_scales[
                            (cell_scales >= cl) & (cell_scales <= ch)
                        ]
                        if cell_trimmed.size > 0:
                            grid[gy, gx] = float(np.median(cell_trimmed))

            # Clamp extreme cells relative to global median.
            grid = np.clip(grid, median_scale * 0.1, median_scale * 10.0)

            # --- Interpolate coarse grid to every Gaussian ---
            # Sample positions are cell-center-relative: map pixel to
            # continuous grid coordinates for bilinear interpolation.
            all_px = pixel_x.copy()
            all_py = pixel_y.copy()
            # Clamp for out-of-bounds points (invalid ones get fallback).
            all_px = np.clip(all_px, 0, image_width - 1)
            all_py = np.clip(all_py, 0, image_height - 1)
            gx_cont = all_px / cell_width - 0.5
            gy_cont = all_py / cell_height - 0.5

            # Bilinear interpolation on the coarse grid.
            gx0 = np.clip(np.floor(gx_cont).astype(np.int32), 0, grid_cells_x - 1)
            gy0 = np.clip(np.floor(gy_cont).astype(np.int32), 0, grid_cells_y - 1)
            gx1 = np.clip(gx0 + 1, 0, grid_cells_x - 1)
            gy1 = np.clip(gy0 + 1, 0, grid_cells_y - 1)
            wx = np.clip(gx_cont - gx0, 0, 1).astype(np.float32)
            wy = np.clip(gy_cont - gy0, 0, 1).astype(np.float32)

            s00 = grid[gy0, gx0]
            s01 = grid[gy0, gx1]
            s10 = grid[gy1, gx0]
            s11 = grid[gy1, gx1]
            smooth_scale = (
                s00 * (1 - wx) * (1 - wy)
                + s01 * wx * (1 - wy)
                + s10 * (1 - wx) * wy
                + s11 * wx * wy
            )

            # Blend smooth grid scale with per-point raw scale.
            dw = float(np.clip(detail_weight, 0.0, 1.0))
            if dw > 0.0:
                per_point_raw = np.full(mv_np.shape[0], median_scale, dtype=np.float32)
                valid_indices = np.where(valid)[0]
                ok_within_valid = np.where(ok)[0]
                per_point_raw[valid_indices[ok_within_valid]] = raw_scale
                per_point_scale = smooth_scale * (1.0 - dw) + per_point_raw * dw
            else:
                per_point_scale = smooth_scale
            per_point_scale[~valid] = median_scale

    device = mean_vectors.device
    dtype = mean_vectors.dtype
    scale_t = (
        torch.from_numpy(per_point_scale)
        .to(device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(-1)
    )  # (1, N, 1)

    return Gaussians3D(
        mean_vectors=mean_vectors * scale_t,
        singular_values=gaussians.singular_values * scale_t,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    ), median_scale, count


def bilinear_sample(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    height, width, channels = image.shape
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = (x0 + 1) % width
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = x0 % width
    y0 = np.clip(y0, 0, height - 1)

    wx = sample_x - x0
    wy = sample_y - y0
    wx = wx[..., None]
    wy = wy[..., None]

    image_f32 = image.astype(np.float32)
    top_left = image_f32[y0, x0]
    top_right = image_f32[y0, x1]
    bottom_left = image_f32[y1, x0]
    bottom_right = image_f32[y1, x1]

    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.clip(np.rint(sampled), 0, 255).astype(np.uint8).reshape(sample_x.shape + (channels,))


def bilinear_sample_scalar(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = (x0 + 1) % width
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = x0 % width
    y0 = np.clip(y0, 0, height - 1)

    wx = sample_x - x0
    wy = sample_y - y0

    image_f32 = image.astype(np.float32)
    top_left = image_f32[y0, x0]
    top_right = image_f32[y0, x1]
    bottom_left = image_f32[y1, x0]
    bottom_right = image_f32[y1, x1]

    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def extract_perspective_view(
    panorama: np.ndarray,
    image_width: int,
    image_height: int,
    focal_x_px: float,
    focal_y_px: float,
    view: FaceOrientation,
) -> np.ndarray:
    pixel_coords_x = np.arange(image_width, dtype=np.float32) + 0.5
    pixel_coords_y = np.arange(image_height, dtype=np.float32) + 0.5
    centered_x = (pixel_coords_x - image_width / 2.0) / focal_x_px
    centered_y = (pixel_coords_y - image_height / 2.0) / focal_y_px
    grid_x, grid_y = np.meshgrid(centered_x, centered_y)

    local_dirs = np.stack((grid_x, grid_y, np.ones_like(grid_x)), axis=-1)
    local_dirs /= np.linalg.norm(local_dirs, axis=-1, keepdims=True)

    rotation = view.rotation_matrix
    world_dirs = local_dirs @ rotation.T
    world_x = world_dirs[..., 0]
    world_y = np.clip(world_dirs[..., 1], -1.0, 1.0)
    world_z = world_dirs[..., 2]

    height, width = panorama.shape[:2]
    longitude = np.arctan2(world_x, world_z)
    latitude = np.arcsin(world_y)

    sample_x = (longitude / (2.0 * np.pi) + 0.5) * width - 0.5
    sample_y = (latitude / np.pi + 0.5) * height - 0.5
    return bilinear_sample(panorama, sample_x, sample_y)


def extract_perspective_views(layout: ExtractionLayout, panorama: np.ndarray, image_width: int, image_height: int) -> dict[str, np.ndarray]:
    return {
        view.name: extract_perspective_view(panorama, image_width, image_height, layout.focal_px, layout.focal_y_px, view)
        for view in layout.views
    }


def extract_perspective_scalar_view(
    panorama: np.ndarray,
    image_width: int,
    image_height: int,
    focal_x_px: float,
    focal_y_px: float,
    view: FaceOrientation,
) -> np.ndarray:
    pixel_coords_x = np.arange(image_width, dtype=np.float32) + 0.5
    pixel_coords_y = np.arange(image_height, dtype=np.float32) + 0.5
    centered_x = (pixel_coords_x - image_width / 2.0) / focal_x_px
    centered_y = (pixel_coords_y - image_height / 2.0) / focal_y_px
    grid_x, grid_y = np.meshgrid(centered_x, centered_y)

    local_dirs = np.stack((grid_x, grid_y, np.ones_like(grid_x)), axis=-1)
    local_dirs /= np.linalg.norm(local_dirs, axis=-1, keepdims=True)

    rotation = view.rotation_matrix
    world_dirs = local_dirs @ rotation.T
    world_x = world_dirs[..., 0]
    world_y = np.clip(world_dirs[..., 1], -1.0, 1.0)
    world_z = world_dirs[..., 2]

    height, width = panorama.shape[:2]
    longitude = np.arctan2(world_x, world_z)
    latitude = np.arcsin(world_y)

    sample_x = (longitude / (2.0 * np.pi) + 0.5) * width - 0.5
    sample_y = (latitude / np.pi + 0.5) * height - 0.5
    return bilinear_sample_scalar(panorama, sample_x, sample_y)


def extract_perspective_scalar_views(
    layout: ExtractionLayout,
    panorama: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, np.ndarray]:
    return {
        view.name: extract_perspective_scalar_view(panorama, image_width, image_height, layout.focal_px, layout.focal_y_px, view)
        for view in layout.views
    }


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        raise ValueError("weighted_median requires at least one value.")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order], 1e-6)
    cutoff = 0.5 * float(sorted_weights.sum())
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")])



def build_predictor(checkpoint_path: Path | None, device: torch.device):
    from sharp.models import PredictorParams, create_predictor

    if device.type == "cuda":
        configure_cuda_inference()

    if checkpoint_path is None:
        LOGGER.info("Downloading or loading default SHARP checkpoint from cache.")
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading SHARP checkpoint from %s", checkpoint_path)
        state_dict = load_torch_checkpoint(checkpoint_path)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)
    warmup_sharp_predictor(predictor, device)
    return predictor


def build_da360_predictor(checkpoint_path: Path, device: torch.device) -> DA360Predictor:
    ensure_da360_import_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DA360 checkpoint not found: {checkpoint_path}. Download DA360_large.pth and place it there, or choose it in the GUI."
        )

    model_dict = load_torch_checkpoint(checkpoint_path)
    net_name = str(model_dict.get("net", "DA360"))
    encoder_name = str(model_dict.get("dinov2_encoder", "vits"))
    input_height = int(model_dict.get("height", 518))
    input_width = int(model_dict.get("width", 1036))

    da360_networks = importlib.import_module("networks")
    net_cls = getattr(da360_networks, net_name)
    model = net_cls(input_height, input_width, dinov2_encoder=encoder_name)
    model.to(device)

    model_state_dict = model.state_dict()
    filtered_state_dict = {key: value for key, value in model_dict.items() if key in model_state_dict}
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    if missing_keys:
        LOGGER.warning("DA360 checkpoint is missing %d model keys.", len(missing_keys))
    if unexpected_keys:
        LOGGER.debug("Ignored %d non-model keys in DA360 checkpoint.", len(unexpected_keys))
    model.eval()
    return DA360Predictor(
        model=model,
        input_height=input_height,
        input_width=input_width,
        model_name=net_name,
    )


@torch.no_grad()
def predict_da360_disparity_panorama(
    predictor: DA360Predictor,
    panorama: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    image_pt = torch.from_numpy(panorama.copy()).float().to(device).permute(2, 0, 1) / 255.0
    image_resized = F.interpolate(
        image_pt[None],
        size=(predictor.input_height, predictor.input_width),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=image_resized.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=image_resized.dtype).view(1, 3, 1, 1)
    normalized = (image_resized - mean) / std

    outputs = predictor.model(normalized)
    pred_disp = outputs["pred_disp"].detach()
    pred_disp = torch.clamp(pred_disp, min=1e-6)
    pred_disp = F.interpolate(
        pred_disp,
        size=(panorama.shape[0], panorama.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    return pred_disp[0, 0].to(torch.float32).cpu().numpy()


def face_transform_tensor(face: FaceOrientation, device: torch.device) -> torch.Tensor:
    transform = torch.eye(3, 4, dtype=torch.float32, device=device)
    transform[:, :3] = torch.from_numpy(face.rotation_matrix).to(device=device, dtype=torch.float32)
    return transform


def merge_gaussians(gaussians_list: list[Gaussians3D]) -> Gaussians3D:
    from sharp.utils.gaussians import Gaussians3D

    if not gaussians_list:
        raise ValueError("No face Gaussians were generated.")
    return Gaussians3D(
        mean_vectors=torch.cat([item.mean_vectors for item in gaussians_list], dim=1),
        singular_values=torch.cat([item.singular_values for item in gaussians_list], dim=1),
        quaternions=torch.cat([item.quaternions for item in gaussians_list], dim=1),
        colors=torch.cat([item.colors for item in gaussians_list], dim=1),
        opacities=torch.cat([item.opacities for item in gaussians_list], dim=1),
    )


def ensure_output_format(output_path: Path, requested_format: str | None, config: dict) -> tuple[Path, str]:
    selected = requested_format or config.get("default_output_format")
    if selected is None:
        selected = output_path.suffix.lstrip(".").lower() or "ply"
    selected = str(selected).lower()
    if selected not in {"ply", "spx", "spz", "sog"}:
        raise ValueError(f"Unsupported output format: {selected}")
    final_path = output_path if output_path.suffix.lower() == f".{selected}" else output_path.with_suffix(f".{selected}")
    return final_path, selected


def find_gsbox(gsbox_arg: Path | None) -> Path | None:
    candidates: list[Path] = []
    if gsbox_arg is not None:
        candidates.append(gsbox_arg)
    candidates.append(ROOT_DIR / "gsbox.exe")
    candidates.append(ROOT_DIR / "release_pkg" / "gsbox.exe")
    resolved = shutil.which("gsbox") or shutil.which("gsbox.exe")
    if resolved:
        candidates.append(Path(resolved))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def convert_with_gsbox(
    source_ply: Path,
    output_path: Path,
    output_format: str,
    quality: int,
    sh_degree: int,
    gsbox_path: Path,
) -> None:
    from plyfile import PlyData

    # gsbox rejects SHARP's richer PLY exports when extra metadata elements are
    # present after the vertex payload. For compressed conversion, rewrite the
    # temporary PLY to a minimal vertex-only form that preserves the Gaussian
    # attributes but drops SHARP-specific metadata blocks.
    ply_data = PlyData.read(source_ply)
    vertex_element = ply_data["vertex"]
    gsbox_source_ply = source_ply.with_name(f"{source_ply.stem}_gsbox.ply")
    PlyData([vertex_element], text=False).write(gsbox_source_ply)

    command = [
        str(gsbox_path),
        f"ply2{output_format}",
        "-i",
        str(gsbox_source_ply),
        "-o",
        str(output_path),
        "-sh",
        str(sh_degree),
    ]
    if output_format in COMPRESSED_OUTPUT_SUFFIXES:
        command.extend(["-q", str(quality)])
    LOGGER.info("Running gsbox conversion: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        output = output.strip()
        if output:
            raise RuntimeError(f"gsbox conversion failed with exit code {result.returncode}:\n{output}")
        raise RuntimeError(f"gsbox conversion failed with exit code {result.returncode}.")


def save_depth_visualization(depth: np.ndarray, save_path: Path) -> None:
    """Save a clipped disparity-style visualization where near structure has more contrast."""
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    if valid.size == 0:
        normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        low, high = np.quantile(valid, [0.02, 0.98])
        if high - low < 1e-8:
            normalized = np.zeros_like(depth, dtype=np.float32)
        else:
            normalized = np.clip((depth - low) / (high - low), 0.0, 1.0).astype(np.float32)
    # Higher disparity means closer geometry. Map near = warm (red), far = cool (blue).
    hue = ((1.0 - normalized) * 170).astype(np.uint8)
    sat = np.full_like(hue, 220)
    val = np.full_like(hue, 230)
    hsv_image = Image.fromarray(np.stack([hue, sat, val], axis=-1), mode="HSV")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    hsv_image.convert("RGB").save(save_path)
    LOGGER.info("Saved DA360 depth visualization to %s", save_path)


def save_intermediate_face_images(faces: dict[str, np.ndarray], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, image in faces.items():
        Image.fromarray(image).save(directory / f"{name}.png")


def save_intermediate_face_splats(
    face_gaussians: dict[str, Gaussians3D],
    focal_x_px: float,
    focal_y_px: float,
    image_width: int,
    image_height: int,
    directory: Path,
) -> None:
    from sharp.utils.gaussians import save_ply

    directory.mkdir(parents=True, exist_ok=True)
    for name, gaussians in face_gaussians.items():
        save_ply(gaussians, (focal_x_px, focal_y_px), (image_width, image_height), directory / f"{name}.ply")


def choose_intermediate_dir(args: argparse.Namespace, output_path: Path) -> Path | None:
    if args.intermediate_dir is not None:
        return args.intermediate_dir
    if args.keep_intermediates:
        return args.input.parent / "Temp" / args.input.stem
    return None


def choose_temp_root(args: argparse.Namespace) -> Path:
    if args.intermediate_dir is not None:
        return args.intermediate_dir
    return args.input.parent / "Temp" / args.input.stem


def cleanup_temp_root(args: argparse.Namespace, temp_root: Path) -> None:
    if not bool(getattr(args, "delete_temp_files", False)):
        return
    if bool(getattr(args, "keep_intermediates", False)):
        return
    if not temp_root.exists():
        return
    shutil.rmtree(temp_root, ignore_errors=True)
    LOGGER.info("Deleted Temp workspace %s", temp_root)


def get_cache_root(temp_root: Path) -> Path:
    return temp_root / "files"


def get_cache_manifest_path(temp_root: Path) -> Path:
    return get_cache_root(temp_root) / "config.json"


def describe_path_for_cache(path_value: Path | None) -> dict[str, Any] | None:
    if path_value is None:
        return None
    path = Path(path_value)
    resolved = path.resolve() if path.exists() else path.absolute()
    data: dict[str, Any] = {"path": str(resolved)}
    if path.exists():
        stat = path.stat()
        data["size"] = int(stat.st_size)
        data["mtime_ns"] = int(stat.st_mtime_ns)
    return data


def read_cache_manifest(temp_root: Path) -> dict[str, Any]:
    manifest_path = get_cache_manifest_path(temp_root)
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_cache_manifest(temp_root: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path = get_cache_manifest_path(temp_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def is_cache_step_valid(
    manifest: Mapping[str, Any],
    step_name: str,
    settings: Mapping[str, Any],
    required_paths: Sequence[Path],
) -> bool:
    steps = manifest.get("steps")
    if not isinstance(steps, dict):
        return False
    step = steps.get(step_name)
    if not isinstance(step, dict):
        return False
    if step.get("settings") != dict(settings):
        return False
    return all(path.exists() for path in required_paths)


def save_cached_image_array(path: Path, image_array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).save(path)


def load_cached_image_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def save_cached_face_images(faces: Mapping[str, np.ndarray], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, image in faces.items():
        Image.fromarray(image).save(directory / f"{name}.png")


def load_cached_face_images(view_names: Sequence[str], directory: Path) -> dict[str, np.ndarray]:
    return {
        view_name: load_cached_image_array(directory / f"{view_name}.png")
        for view_name in view_names
    }


def save_cached_depth_arrays(
    depth_panorama: np.ndarray,
    depth_views: Mapping[str, np.ndarray],
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "panorama.npy", depth_panorama)
    for view_name, depth_view in depth_views.items():
        np.save(directory / f"{view_name}.npy", depth_view)


def load_cached_depth_arrays(
    view_names: Sequence[str],
    directory: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    depth_panorama = np.load(directory / "panorama.npy").astype(np.float32)
    depth_views = {
        view_name: np.load(directory / f"{view_name}.npy").astype(np.float32)
        for view_name in view_names
    }
    return depth_panorama, depth_views


def save_cached_face_gaussians(face_gaussians: Mapping[str, Gaussians3D], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, gaussians in face_gaussians.items():
        torch.save(
            {
                "mean_vectors": gaussians.mean_vectors.detach().cpu(),
                "singular_values": gaussians.singular_values.detach().cpu(),
                "quaternions": gaussians.quaternions.detach().cpu(),
                "colors": gaussians.colors.detach().cpu(),
                "opacities": gaussians.opacities.detach().cpu(),
            },
            directory / f"{name}.pt",
        )


def load_cached_face_gaussians(view_names: Sequence[str], directory: Path) -> dict[str, Gaussians3D]:
    from sharp.utils.gaussians import Gaussians3D

    loaded: dict[str, Gaussians3D] = {}
    for view_name in view_names:
        payload_path = directory / f"{view_name}.pt"
        try:
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(payload_path, map_location="cpu")
        loaded[view_name] = Gaussians3D(
            mean_vectors=payload["mean_vectors"],
            singular_values=payload["singular_values"],
            quaternions=payload["quaternions"],
            colors=payload["colors"],
            opacities=payload["opacities"],
        )
    return loaded


def build_output_identity(
    side_count: int,
    grid_resolution: int,
    detail_weight: float,
    upscale_factor: int,
) -> dict[str, Any]:
    return {
        "side_count": int(side_count),
        "grid_resolution": int(grid_resolution),
        "detail_preservation": int(round(float(np.clip(detail_weight, 0.0, 1.0)) * 100.0)),
        "upscale_factor": int(upscale_factor),
        "alignment_mode": "da360",
    }


def resolve_output_conflict(
    output_path: Path,
    alignment_sweep_enabled: bool,
    previous_output_identity: Mapping[str, Any] | None,
    current_output_identity: Mapping[str, Any],
) -> Path:
    existing_target = output_path.with_suffix("") if alignment_sweep_enabled else output_path
    if not existing_target.exists() or not isinstance(previous_output_identity, Mapping):
        return output_path

    suffixes: list[str] = []
    if previous_output_identity.get("side_count") != current_output_identity.get("side_count"):
        suffixes.append(f"S{current_output_identity['side_count']}")
    if previous_output_identity.get("grid_resolution") != current_output_identity.get("grid_resolution"):
        suffixes.append(f"GR{current_output_identity['grid_resolution']}")
    if previous_output_identity.get("detail_preservation") != current_output_identity.get("detail_preservation"):
        suffixes.append(f"DP{current_output_identity['detail_preservation']}")
    if previous_output_identity.get("upscale_factor") != current_output_identity.get("upscale_factor"):
        suffixes.append(f"UF{current_output_identity['upscale_factor']}")
    if previous_output_identity.get("alignment_mode") != current_output_identity.get("alignment_mode"):
        suffixes.append(f"AL{current_output_identity['alignment_mode']}")
    if not suffixes:
        return output_path

    return output_path.with_name(f"{output_path.stem}_{'_'.join(suffixes)}{output_path.suffix}")


def build_sweep_axis_values(
    center: int,
    offsets: Sequence[int],
    *,
    min_value: int,
    max_value: int,
    fill_step: int,
    count: int = 5,
) -> list[int]:
    center = int(center)
    values: list[int] = []
    seen: set[int] = set()

    def add(candidate: int) -> None:
        candidate = max(min_value, min(max_value, int(candidate)))
        if candidate in seen:
            return
        seen.add(candidate)
        values.append(candidate)

    for offset in offsets:
        add(center + offset)
        if len(values) >= count:
            return values[:count]

    radius = max(1, fill_step)
    while len(values) < count and (center - radius >= min_value or center + radius <= max_value):
        if center - radius >= min_value:
            add(center - radius)
            if len(values) >= count:
                break
        if center + radius <= max_value:
            add(center + radius)
            if len(values) >= count:
                break
        radius += max(1, fill_step)

    if len(values) < count:
        for candidate in range(min_value, max_value + 1):
            add(candidate)
            if len(values) >= count:
                break

    return values[:count]


def build_alignment_sweep_combinations(
    grid_resolution: int,
    detail_weight: float,
) -> list[tuple[int, float, int]]:
    grid_values = build_sweep_axis_values(
        int(grid_resolution),
        (-6, -3, 0, 3, 6),
        min_value=1,
        max_value=64,
        fill_step=1,
    )
    detail_percent = int(round(float(np.clip(detail_weight, 0.0, 1.0)) * 100.0))
    detail_values = build_sweep_axis_values(
        detail_percent,
        (-20, -10, 0, 20, 30),
        min_value=0,
        max_value=100,
        fill_step=10,
    )
    return [
        (grid_value, detail_value / 100.0, detail_value)
        for grid_value in grid_values
        for detail_value in detail_values
    ]


def build_alignment_sweep_output_path(base_output_path: Path, grid_resolution: int, detail_percent: int) -> Path:
    output_dir = base_output_path.with_suffix("")
    return output_dir / f"{base_output_path.stem}_GR{grid_resolution}_DP{detail_percent}{base_output_path.suffix}"


def write_merged_output(
    merged: Gaussians3D,
    output_path: Path,
    output_format: str,
    quality: int,
    sh_degree: int,
    gsbox_path: Path | None,
    focal_x_px: float,
    focal_y_px: float,
    image_width: int,
    image_height: int,
    conversion_dir: Path,
) -> None:
    from sharp.utils.gaussians import save_ply

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "ply":
        LOGGER.info("Saving merged PLY to %s", output_path)
        save_ply(merged, (focal_x_px, focal_y_px), (image_width, image_height), output_path)
        LOGGER.info("Wrote merged output to %s", output_path)
        return

    if gsbox_path is None:
        raise FileNotFoundError(
            "Compressed output requires gsbox.exe. Provide --gsbox or place gsbox.exe next to the script."
        )
    conversion_dir.mkdir(parents=True, exist_ok=True)
    temp_ply = conversion_dir / f"{output_path.stem}.ply"
    save_ply(merged, (focal_x_px, focal_y_px), (image_width, image_height), temp_ply)
    convert_with_gsbox(temp_ply, output_path, output_format, quality, sh_degree, gsbox_path)
    LOGGER.info("Wrote merged output to %s", output_path)


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    from sharp.cli.predict import predict_image
    from sharp.utils.gaussians import apply_transform

    pipeline_start = time.perf_counter()
    pipeline_succeeded = False
    step_timings: list[StepTiming] = []
    step_timings_token = CURRENT_STEP_TIMINGS.set(step_timings)
    step_depth_token = CURRENT_STEP_TIMING_DEPTH.set(0)
    try:
        with log_timed_step("Resolve pipeline settings"):
            config = load_config(args.config)
            output_path, output_format = ensure_output_format(args.output, args.format, config)
            side_count = resolve_side_count(getattr(args, "side_count", 0), config)
            cutoff_height_percent = resolve_cutoff_height_percent(getattr(args, "cutoff_height_percent", 0.0), config)
            alignment_mode = resolve_alignment_mode(args, config)
            da360_alignment_enabled = alignment_mode == "da360"
            overlap_alignment_enabled = alignment_mode == "overlap"
            alignment_sweep_enabled = bool(getattr(args, "enable_alignment_sweep", False))
            da360_checkpoint_path = resolve_da360_checkpoint_path(args, config) if da360_alignment_enabled else None
            quality = args.quality if args.quality is not None else int(config.get("default_quality", 9))
            sh_degree = args.sh_degree if args.sh_degree is not None else int(config.get("default_sh_degree", 0))
            clip_horizontal_raw = config.get("merge_clip_horizontal_degrees", 0)
            clip_horizontal_degrees = (360.0 / side_count) if clip_horizontal_raw in {None, 0, 0.0, "", False} else float(clip_horizontal_raw)
            clip_vertical_raw = config.get("merge_clip_vertical_degrees")
            if clip_vertical_raw in {None, 0, 0.0, "", False}:
                clip_vertical_degrees = None
            else:
                clip_vertical_degrees = float(str(clip_vertical_raw))
            if not (1 <= quality <= 9):
                raise ValueError("Quality must be between 1 and 9.")
            if not (0 <= sh_degree <= 3):
                raise ValueError("SH degree must be between 0 and 3.")
            if alignment_sweep_enabled and not da360_alignment_enabled:
                raise ValueError("Alignment sweep mode requires DA360 alignment mode.")
            if not (1.0 <= clip_horizontal_degrees <= 180.0):
                raise ValueError("merge_clip_horizontal_degrees must be between 1 and 180.")
            if clip_vertical_degrees is not None and not (1.0 <= clip_vertical_degrees <= 180.0):
                raise ValueError("merge_clip_vertical_degrees must be between 1 and 180 when enabled.")

        with log_timed_step("Resolve device"):
            device = resolve_device(args.device)
        LOGGER.info("Using device: %s", device)
        if device.type == "cpu" and torch.cuda.is_available():
            LOGGER.warning(
                "SHARP is running on CPU even though CUDA is available on %s. "
                "Set Device to 'default' or 'cuda' in the GUI to restore GPU inference speed.",
                torch.cuda.get_device_name(0),
            )

        temp_root = choose_temp_root(args)
        cache_root = get_cache_root(temp_root)
        previous_manifest = read_cache_manifest(temp_root)
        step_manifests: dict[str, dict[str, Any]] = {}
        seedvr2_upscale_enabled = getattr(args, "enable_seedvr2_upscale", False)
        imagemagick_optimization_enabled = getattr(args, "enable_imagemagick_optimization", False)
        deblur_preprocessing_enabled = bool(getattr(args, "enable_deblur_preprocessing", False))
        deblur_strength = normalize_deblur_strength(getattr(args, "deblur_strength", "medium"))
        seedvr2_settings = load_seedvr2_settings() if seedvr2_upscale_enabled else None
        upscale_factor = int(seedvr2_settings.get("resolution_factor", 2)) if seedvr2_settings is not None else 1
        current_output_identity = build_output_identity(
            side_count,
            getattr(args, "alignment_grid_resolution", 8),
            getattr(args, "alignment_detail_weight", 0.0),
            upscale_factor,
        )
        current_output_identity["alignment_mode"] = alignment_mode
        resolved_output_path = resolve_output_conflict(
            output_path,
            alignment_sweep_enabled,
            previous_manifest.get("output_identity") if isinstance(previous_manifest, dict) else None,
            current_output_identity,
        )
        if resolved_output_path != output_path:
            LOGGER.info("Output already exists; writing this run to %s", resolved_output_path)
            output_path = resolved_output_path

        input_signature = describe_path_for_cache(args.input)
        imagemagick_commands = resolve_imagemagick_command_tokens(args) if imagemagick_optimization_enabled else []
        processed_panorama_path = cache_root / "processed_panorama.png"
        preprocess_settings = {
            "input": input_signature,
            "horizontal_mirror": False,
            "imagemagick_enabled": bool(imagemagick_optimization_enabled),
            "imagemagick_commands": list(imagemagick_commands),
            "seedvr2_sharpen_only": bool(seedvr2_upscale_enabled and not imagemagick_optimization_enabled),
        }

        with log_timed_step("Load and validate panorama"):
            if imagemagick_optimization_enabled and not imagemagick_commands:
                raise ValueError("ImageMagick optimization is enabled, but no preprocessing operations were selected.")
            if is_cache_step_valid(previous_manifest, "preprocess", preprocess_settings, [processed_panorama_path]):
                panorama = load_cached_image_array(processed_panorama_path)
                LOGGER.info("Reusing cached processed panorama from %s", processed_panorama_path)
            else:
                register_optional_image_plugins()
                panorama = load_input_panorama(args.input)
                if imagemagick_optimization_enabled:
                    magick_path = find_imagemagick_executable(getattr(args, "imagemagick", None))
                    if magick_path is None:
                        raise FileNotFoundError("ImageMagick optimization is enabled, but magick.exe was not found. Provide --imagemagick or add ImageMagick to PATH.")
                    LOGGER.info("Optimizing panorama with ImageMagick before slicing.")
                    panorama = optimize_panorama_with_imagemagick(panorama, temp_root, magick_path, imagemagick_commands)
                elif seedvr2_upscale_enabled:
                    LOGGER.info("Applying sharpening to panorama before face extraction.")
                    panorama = sharpen_panorama(panorama)
                save_cached_image_array(processed_panorama_path, panorama)
            panorama_width, panorama_height = validate_equirectangular_shape(panorama)
        step_manifests["preprocess"] = {"settings": preprocess_settings}

        with log_timed_step("Build extraction layout"):
            face_size = resolve_face_size(args.face_size, panorama_width, side_count, config)
            extraction_layout = build_extraction_layout(face_size, panorama_height, side_count, cutoff_height_percent, config)
            focal_px = extraction_layout.focal_px
            focal_y_px = extraction_layout.focal_y_px
            image_width = extraction_layout.image_width
            image_height = extraction_layout.image_height

        LOGGER.info(
            "Loaded panorama %s with resolution %dx%d. Extracting %d %dx%d perspective views using %s.",
            args.input,
            panorama_width,
            panorama_height,
            len(extraction_layout.views),
            image_width,
            image_height,
            extraction_layout.name,
        )

        view_names = [view.name for view in extraction_layout.views]
        faces_cache_dir = cache_root / "faces"
        faces_settings = {
            "preprocess": preprocess_settings,
            "config": describe_path_for_cache(args.config),
            "side_count": int(side_count),
            "cutoff_height_percent": round(float(cutoff_height_percent), 4),
            "requested_face_size": int(args.face_size),
            "resolved_face_width": int(image_width),
            "resolved_face_height": int(image_height),
            "layout": extraction_layout.name,
        }
        with log_timed_step("Extract perspective views"):
            required_face_paths = [faces_cache_dir / f"{view_name}.png" for view_name in view_names]
            if is_cache_step_valid(previous_manifest, "faces", faces_settings, required_face_paths):
                faces = load_cached_face_images(view_names, faces_cache_dir)
                LOGGER.info("Reusing cached extracted faces from %s", faces_cache_dir)
            else:
                faces = extract_perspective_views(extraction_layout, panorama, image_width, image_height)
                save_cached_face_images(faces, faces_cache_dir)
        step_manifests["faces"] = {"settings": faces_settings}

        intermediate_dir = choose_intermediate_dir(args, output_path)
        if intermediate_dir is not None:
            with log_timed_step("Save extracted face images"):
                save_intermediate_face_images(faces, intermediate_dir / "faces")

        if deblur_preprocessing_enabled:
            deblur_step_settings = {
                "enabled": True,
                "strength": deblur_strength,
                "input_face_width": int(image_width),
                "input_face_height": int(image_height),
                "view_names": list(view_names),
            }
            deblurred_faces_cache_dir = cache_root / "faces_deblurred"
            with log_timed_step("Deblur extracted faces"):
                required_deblurred_paths = [deblurred_faces_cache_dir / f"{view_name}.png" for view_name in view_names]
                if is_cache_step_valid(previous_manifest, "deblur", deblur_step_settings, required_deblurred_paths):
                    faces = load_cached_face_images(view_names, deblurred_faces_cache_dir)
                    LOGGER.info("Reusing cached deblurred faces from %s", deblurred_faces_cache_dir)
                else:
                    faces = deblur_faces_with_motion_rl(faces, deblur_strength)
                    save_cached_face_images(faces, deblurred_faces_cache_dir)
            step_manifests["deblur"] = {"settings": deblur_step_settings}
            if intermediate_dir is not None:
                with log_timed_step("Save deblurred face images"):
                    save_intermediate_face_images(faces, intermediate_dir / "faces_deblurred")
        else:
            step_manifests["deblur"] = {"settings": {"enabled": False}}

        if seedvr2_upscale_enabled:
            expected_upscaled_width, expected_upscaled_height = image_width, image_height
            seedvr2_input_width, seedvr2_input_height, seedvr2_stretch_applied = image_width, image_height, False
            if seedvr2_settings is not None:
                seedvr2_input_width, seedvr2_input_height, seedvr2_stretch_applied = resolve_seedvr2_stretch_dimensions(
                    image_width,
                    image_height,
                    seedvr2_settings,
                )
                _, expected_upscaled_width, expected_upscaled_height = resolve_seedvr2_target_dimensions(
                    seedvr2_input_width,
                    seedvr2_input_height,
                    seedvr2_settings,
                )
            seedvr2_step_settings = {
                "enabled": True,
                "settings": seedvr2_settings,
                "deblur": step_manifests["deblur"]["settings"],
                "input_face_width": int(image_width),
                "input_face_height": int(image_height),
                "seedvr2_input_face_width": int(seedvr2_input_width),
                "seedvr2_input_face_height": int(seedvr2_input_height),
                "stretch_applied": bool(seedvr2_stretch_applied),
                "expected_output_face_width": int(expected_upscaled_width),
                "expected_output_face_height": int(expected_upscaled_height),
                "view_names": list(view_names),
            }
            upscaled_faces_cache_dir = cache_root / "faces_upscaled"
            with log_timed_step("Upscale faces with SeedVR2"):
                original_face_width = image_width
                required_upscaled_paths = [upscaled_faces_cache_dir / f"{view_name}.png" for view_name in view_names]
                if is_cache_step_valid(previous_manifest, "seedvr2", seedvr2_step_settings, required_upscaled_paths):
                    faces = load_cached_face_images(view_names, upscaled_faces_cache_dir)
                    image_height, image_width = faces[view_names[0]].shape[:2]
                    LOGGER.info("Reusing cached SeedVR2 upscaled faces from %s", upscaled_faces_cache_dir)
                else:
                    faces, image_width, image_height = upscale_faces_with_seedvr2(faces, image_width, image_height, temp_root)
                    save_cached_face_images(faces, upscaled_faces_cache_dir)
                focal_px = extraction_layout.focal_px * (image_width / original_face_width)
                focal_y_px = extraction_layout.focal_y_px * (image_height / max(1, extraction_layout.image_height))
            step_manifests["seedvr2"] = {"settings": seedvr2_step_settings}
            if intermediate_dir is not None:
                with log_timed_step("Save upscaled face images"):
                    save_intermediate_face_images(faces, intermediate_dir / "faces_upscaled")
        else:
            step_manifests["seedvr2"] = {"settings": {"enabled": False}}

        depth_map_path: Path | None = None
        reference_depth_panorama: np.ndarray | None = None
        reference_depth_views: dict[str, np.ndarray] = {}
        if da360_alignment_enabled:
            grid_res = getattr(args, "alignment_grid_resolution", 8)
            detail_wt = getattr(args, "alignment_detail_weight", 0.0)
            depth_cache_dir = cache_root / "depth"
            da360_step_settings = {
                "enabled": True,
                "preprocess": preprocess_settings,
                "checkpoint": describe_path_for_cache(da360_checkpoint_path),
                "resolved_face_width": int(image_width),
                "resolved_face_height": int(image_height),
                "focal_x_px": round(float(focal_px), 6),
                "focal_y_px": round(float(focal_y_px), 6),
                "view_names": list(view_names),
            }
            with log_timed_step("Run DA360 depth inference"):
                required_depth_paths = [depth_cache_dir / "panorama.npy", *[depth_cache_dir / f"{view_name}.npy" for view_name in view_names]]
                if is_cache_step_valid(previous_manifest, "da360", da360_step_settings, required_depth_paths):
                    reference_depth_panorama, reference_depth_views = load_cached_depth_arrays(view_names, depth_cache_dir)
                    LOGGER.info("Reusing cached DA360 depth from %s", depth_cache_dir)
                else:
                    if da360_checkpoint_path is None:
                        raise ValueError("DA360 alignment is enabled, but no DA360 checkpoint path was resolved.")
                    LOGGER.info(
                        "Running DA360 panorama depth inference using checkpoint %s "
                        "(grid=%dx%d, detail=%.0f%%)",
                        da360_checkpoint_path, grid_res, grid_res, detail_wt * 100,
                    )
                    da360_predictor = build_da360_predictor(da360_checkpoint_path, device)
                    reference_depth_panorama = predict_da360_disparity_panorama(da360_predictor, panorama, device)
                    reference_depth_views = {
                        view.name: extract_perspective_scalar_view(reference_depth_panorama, image_width, image_height, focal_px, focal_y_px, view)
                        for view in extraction_layout.views
                    }
                    save_cached_depth_arrays(reference_depth_panorama, reference_depth_views, depth_cache_dir)
                    del da360_predictor
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            step_manifests["da360"] = {"settings": da360_step_settings}
            depth_map_path = temp_root / "depth" / f"{args.input.stem}_depth.png"
            with log_timed_step("Save DA360 depth visualizations"):
                save_depth_visualization(reference_depth_panorama, depth_map_path)
                if intermediate_dir is not None:
                    depth_vis_dir = intermediate_dir / "depth_views"
                    for view_name, depth_view in reference_depth_views.items():
                        save_depth_visualization(depth_view, depth_vis_dir / f"{view_name}_depth.png")
        else:
            step_manifests["da360"] = {"settings": {"enabled": False}}

        raw_face_gaussians: dict[str, Gaussians3D]
        prepared_face_gaussians: dict[str, Gaussians3D]
        sharp_cache_dir = cache_root / "raw_gaussians"
        sharp_step_settings = {
            "checkpoint": describe_path_for_cache(args.checkpoint) or {"default_model_url": DEFAULT_MODEL_URL},
            "config": describe_path_for_cache(args.config),
            "view_names": list(view_names),
            "resolved_face_width": int(image_width),
            "resolved_face_height": int(image_height),
            "focal_x_px": round(float(focal_px), 6),
            "focal_y_px": round(float(focal_y_px), 6),
            "deblur": step_manifests["deblur"]["settings"],
            "seedvr2": step_manifests["seedvr2"]["settings"],
        }
        required_sharp_paths = [sharp_cache_dir / f"{view_name}.pt" for view_name in view_names]
        if is_cache_step_valid(previous_manifest, "sharp", sharp_step_settings, required_sharp_paths):
            with log_timed_step("Load cached raw SHARP face gaussians"):
                raw_face_gaussians = load_cached_face_gaussians(view_names, sharp_cache_dir)
                LOGGER.info("Reusing cached raw SHARP face gaussians from %s", sharp_cache_dir)
        else:
            with log_timed_step("Build SHARP predictor"):
                predictor = build_predictor(args.checkpoint, device)

            raw_face_gaussians = {}
            for view in extraction_layout.views:
                with log_timed_step(f"Predict SHARP splats for view {view.name}"):
                    LOGGER.info("Predicting SHARP splats for view: %s", view.name)
                    gaussians = predict_image(predictor, faces[view.name], (focal_px, focal_y_px), device)
                raw_face_gaussians[view.name] = gaussians.to(torch.device("cpu"))

            save_cached_face_gaussians(raw_face_gaussians, sharp_cache_dir)
            del predictor
            if device.type == "cuda":
                torch.cuda.empty_cache()
        step_manifests["sharp"] = {"settings": sharp_step_settings}

        prepared_face_gaussians = {}
        for view in extraction_layout.views:
            with log_timed_step(f"Filter Gaussians for view {view.name}"):
                prepared_face_gaussians[view.name] = filter_gaussians_by_view_border(
                    raw_face_gaussians[view.name],
                    horizontal_border_degrees=clip_horizontal_degrees,
                    vertical_border_degrees=clip_vertical_degrees,
                )

        if intermediate_dir is not None:
            with log_timed_step("Save raw per-face SHARP splats"):
                save_intermediate_face_splats(
                    raw_face_gaussians,
                    focal_px,
                    focal_y_px,
                    image_width,
                    image_height,
                    intermediate_dir / "face_splats",
                )

        original_median_radii: list[float] = []
        if da360_alignment_enabled or overlap_alignment_enabled:
            for view_name in view_names:
                original_median_radii.append(float(torch.median(torch.norm(
                    prepared_face_gaussians[view_name].mean_vectors, dim=-1,
                )).item()))

        overlap_alignment_scales = {view.name: 1.0 for view in extraction_layout.views}
        overlap_alignment_fields = {view.name: {} for view in extraction_layout.views}
        overlap_depth_layer_curves = {view.name: None for view in extraction_layout.views}
        overlap_edge_refinement_fields = {view.name: {} for view in extraction_layout.views}
        if overlap_alignment_enabled:
            with log_timed_step("Resolve overlap-based face alignment"):
                overlap_alignment_scales, overlap_diagnostics = resolve_overlap_alignment_scales(
                    raw_face_gaussians,
                    extraction_layout,
                    config,
                )
                for left_name, right_name, ratio, shared_bins in overlap_diagnostics:
                    LOGGER.info(
                        "Overlap alignment constraint %s -> %s: relative_scale=%.6f (%d shared bins).",
                        left_name,
                        right_name,
                        ratio,
                        shared_bins,
                    )
                LOGGER.info(
                    "Resolved overlap alignment scales: %s",
                    ", ".join(f"{view.name}={overlap_alignment_scales[view.name]:.4f}" for view in extraction_layout.views),
                )
                overlap_alignment_fields, overlap_field_diagnostics = resolve_overlap_alignment_fields(
                    raw_face_gaussians,
                    extraction_layout,
                    config,
                    overlap_alignment_scales,
                )
                for left_name, right_name, residual_ratio, shared_bins in overlap_field_diagnostics:
                    LOGGER.info(
                        "Overlap seam field %s <-> %s: residual_ratio=%.6f (%d shared bins).",
                        left_name,
                        right_name,
                        residual_ratio,
                        shared_bins,
                    )
            with log_timed_step("Resolve overlap depth standardization"):
                clipped_overlap_preview: dict[str, Gaussians3D] = {}
                for view in extraction_layout.views:
                    preview_gaussians, _preview_points = apply_overlap_global_and_local_alignment(
                        prepared_face_gaussians[view.name],
                        view,
                        float(overlap_alignment_scales.get(view.name, 1.0)),
                        overlap_alignment_fields.get(view.name, {}),
                    )
                    clipped_overlap_preview[view.name] = preview_gaussians.to(torch.device("cpu"))

                overlap_depth_layer_curves, overlap_depth_diagnostics = resolve_overlap_depth_layer_standardization(
                    clipped_overlap_preview,
                    extraction_layout,
                    config,
                )
                for view_name, sample_count, max_scale in overlap_depth_diagnostics:
                    LOGGER.info(
                        "Overlap depth standardization for %s: %d seam samples, max layer scale %.4f.",
                        view_name,
                        sample_count,
                        max_scale,
                    )

                depth_standardized_preview: dict[str, Gaussians3D] = {}
                for view in extraction_layout.views:
                    standardized_gaussians, standardized_points, median_scale = apply_depth_layer_standardization_curve(
                        clipped_overlap_preview[view.name],
                        overlap_depth_layer_curves.get(view.name),
                    )
                    depth_standardized_preview[view.name] = standardized_gaussians.to(torch.device("cpu"))
                    if standardized_points > 0:
                        LOGGER.info(
                            "Applied depth-layer preview standardization to %s: %d points, median scale %.4f.",
                            view.name,
                            standardized_points,
                            median_scale,
                        )

                overlap_edge_refinement_fields, overlap_edge_diagnostics = resolve_clipped_edge_alignment_fields(
                    depth_standardized_preview,
                    extraction_layout,
                    config,
                    {view.name: 1.0 for view in extraction_layout.views},
                )
                for left_name, right_name, residual_ratio, shared_bins in overlap_edge_diagnostics:
                    LOGGER.info(
                        "Clipped-edge refinement field %s <-> %s: residual_ratio=%.6f (%d shared bins).",
                        left_name,
                        right_name,
                        residual_ratio,
                        shared_bins,
                    )

        face_transforms = {
            view.name: face_transform_tensor(view, device)
            for view in extraction_layout.views
        }

        gsbox_path: Path | None = None
        if output_format != "ply":
            with log_timed_step("Resolve gsbox for compressed output"):
                gsbox_path = find_gsbox(args.gsbox)
                if gsbox_path is None:
                    raise FileNotFoundError(
                        "Compressed output requires gsbox.exe. Provide --gsbox or place gsbox.exe next to the script."
                    )

        generated_outputs: list[Path] = []
        display_path: Path | None = None
        original_scene_median = float(np.median(original_median_radii)) if original_median_radii else 0.0

        if alignment_sweep_enabled:
            sweep_combinations = build_alignment_sweep_combinations(
                getattr(args, "alignment_grid_resolution", 8),
                getattr(args, "alignment_detail_weight", 0.0),
            )
            display_path = output_path.with_suffix("")
            with log_timed_step("Prepare alignment sweep output directory"):
                display_path.mkdir(parents=True, exist_ok=True)
            grid_values = sorted({grid for grid, _weight, _percent in sweep_combinations})
            detail_values = sorted({percent for _grid, _weight, percent in sweep_combinations})
            LOGGER.info(
                "Alignment sweep mode: %d variants. Grid values=%s Detail values=%s%%. Output directory: %s",
                len(sweep_combinations),
                ", ".join(str(value) for value in grid_values),
                ", ".join(str(value) for value in detail_values),
                display_path,
            )

            for grid_resolution, detail_weight, detail_percent in sweep_combinations:
                combo_name = f"GR{grid_resolution}_DP{detail_percent}"
                variant_output_path = build_alignment_sweep_output_path(output_path, grid_resolution, detail_percent)
                with log_timed_step(f"Build merged splat {combo_name}"):
                    rotated_gaussian_list: list[Gaussians3D] = []
                    for view in extraction_layout.views:
                        gaussians = prepared_face_gaussians[view.name].to(device)
                        if da360_alignment_enabled:
                            gaussians, median_scale, sample_count = align_gaussians_to_reference(
                                gaussians,
                                reference_depth_views[view.name],
                                focal_x_px=focal_px,
                                focal_y_px=focal_y_px,
                                image_width=image_width,
                                image_height=image_height,
                                grid_resolution=grid_resolution,
                                detail_weight=detail_weight,
                            )
                            LOGGER.info(
                                "Aligned %s for %s: median_scale=%.6f (%d samples).",
                                view.name, combo_name, median_scale, sample_count,
                            )
                        elif overlap_alignment_enabled:
                            overlap_scale = float(overlap_alignment_scales.get(view.name, 1.0))
                            gaussians, overlap_metrics = apply_overlap_alignment_pipeline(
                                gaussians,
                                view,
                                overlap_scale,
                                overlap_alignment_fields.get(view.name, {}),
                                overlap_depth_layer_curves.get(view.name),
                                overlap_edge_refinement_fields.get(view.name, {}),
                            )
                            LOGGER.info(
                                "Aligned %s for %s using overlap scale %.6f, %d primary seam points, %d depth-layer points (median %.4f), and %d clipped-edge refinement points.",
                                view.name,
                                combo_name,
                                overlap_scale,
                                int(overlap_metrics["primary_points"]),
                                int(overlap_metrics["depth_points"]),
                                float(overlap_metrics["depth_median_scale"]),
                                int(overlap_metrics["residual_points"]),
                            )
                        rotated = apply_transform(gaussians, face_transforms[view.name]).to(torch.device("cpu"))
                        rotated_gaussian_list.append(rotated)

                    merged = merge_gaussians(rotated_gaussian_list)
                    if original_median_radii:
                        current_median = float(torch.median(torch.norm(
                            merged.mean_vectors, dim=-1,
                        )).item())
                        if current_median > 1e-8:
                            global_restore = original_scene_median / current_median
                            merged = scale_gaussians(merged, global_restore)
                            LOGGER.info(
                                "Global scene restore for %s: scale=%.4f (%.4f -> %.4f median radius).",
                                combo_name, global_restore, current_median, original_scene_median,
                            )

                with log_timed_step(f"Write output {combo_name}"):
                    write_merged_output(
                        merged,
                        variant_output_path,
                        output_format,
                        quality,
                        sh_degree,
                        gsbox_path,
                        focal_px,
                        focal_y_px,
                        image_width,
                        image_height,
                        temp_root / "conversion",
                    )
                generated_outputs.append(variant_output_path)
        else:
            rotated_face_gaussians: dict[str, Gaussians3D] = {}
            rotated_gaussian_list: list[Gaussians3D] = []
            for view in extraction_layout.views:
                gaussians = prepared_face_gaussians[view.name].to(device)
                if da360_alignment_enabled:
                    with log_timed_step(f"Align view {view.name} to DA360 depth"):
                        orig_med = float(torch.median(torch.norm(
                            gaussians.mean_vectors, dim=-1,
                        )).item())
                        gaussians, median_scale, sample_count = align_gaussians_to_reference(
                            gaussians,
                            reference_depth_views[view.name],
                            focal_x_px=focal_px,
                            focal_y_px=focal_y_px,
                            image_width=image_width,
                            image_height=image_height,
                            grid_resolution=getattr(args, "alignment_grid_resolution", 8),
                            detail_weight=getattr(args, "alignment_detail_weight", 0.0),
                        )
                        LOGGER.info(
                            "Aligned %s to DA360 depth: median_scale=%.6f "
                            "(%d samples, orig_median_r=%.2f).",
                            view.name, median_scale, sample_count, orig_med,
                        )
                elif overlap_alignment_enabled:
                    with log_timed_step(f"Align view {view.name} from overlap"):
                        overlap_scale = float(overlap_alignment_scales.get(view.name, 1.0))
                        gaussians, overlap_metrics = apply_overlap_alignment_pipeline(
                            gaussians,
                            view,
                            overlap_scale,
                            overlap_alignment_fields.get(view.name, {}),
                            overlap_depth_layer_curves.get(view.name),
                            overlap_edge_refinement_fields.get(view.name, {}),
                        )
                        LOGGER.info(
                            "Aligned %s from overlap constraints using scale %.6f, %d primary seam points, %d depth-layer points (median %.4f), and %d clipped-edge refinement points.",
                            view.name,
                            overlap_scale,
                            int(overlap_metrics["primary_points"]),
                            int(overlap_metrics["depth_points"]),
                            float(overlap_metrics["depth_median_scale"]),
                            int(overlap_metrics["residual_points"]),
                        )
                with log_timed_step(f"Rotate view {view.name} into world frame"):
                    rotated = apply_transform(gaussians, face_transforms[view.name]).to(torch.device("cpu"))
                rotated_face_gaussians[view.name] = rotated
                rotated_gaussian_list.append(rotated)

            if intermediate_dir is not None:
                with log_timed_step("Save world-space face splats"):
                    save_intermediate_face_splats(
                        rotated_face_gaussians,
                        focal_px,
                        focal_y_px,
                        image_width,
                        image_height,
                        intermediate_dir / "face_splats_world",
                    )

            with log_timed_step("Merge rotated face splats"):
                merged = merge_gaussians(rotated_gaussian_list)

            if (da360_alignment_enabled or overlap_alignment_enabled) and original_median_radii:
                with log_timed_step("Restore global scene scale"):
                    current_median = float(torch.median(torch.norm(
                        merged.mean_vectors, dim=-1,
                    )).item())
                    if current_median > 1e-8:
                        global_restore = original_scene_median / current_median
                        merged = scale_gaussians(merged, global_restore)
                        LOGGER.info(
                            "Global scene restore: scale=%.4f (%.4f -> %.4f median radius).",
                            global_restore, current_median, original_scene_median,
                        )

            with log_timed_step("Write merged output"):
                write_merged_output(
                    merged,
                    output_path,
                    output_format,
                    quality,
                    sh_degree,
                    gsbox_path,
                    focal_px,
                    focal_y_px,
                    image_width,
                    image_height,
                    temp_root / "conversion",
                )
            generated_outputs.append(output_path)

        with log_timed_step("Write temp cache manifest"):
            write_cache_manifest(
                temp_root,
                {
                    "version": 1,
                    "input": input_signature,
                    "steps": step_manifests,
                    "output_identity": current_output_identity,
                    "generated_outputs": [str(path) for path in generated_outputs],
                    "display_path": str(display_path) if display_path is not None else None,
                },
            )

        with log_timed_step("Clean up temporary workspace"):
            cleanup_temp_root(args, temp_root)
        pipeline_succeeded = True
        return PipelineResult(
            output_path=output_path,
            depth_map_path=depth_map_path,
            generated_outputs=generated_outputs,
            display_path=display_path,
        )
    finally:
        total_elapsed = time.perf_counter() - pipeline_start
        log_step_timing_summary(step_timings, total_elapsed)
        CURRENT_STEP_TIMINGS.reset(step_timings_token)
        CURRENT_STEP_TIMING_DEPTH.reset(step_depth_token)
        if pipeline_succeeded:
            LOGGER.info("Whole pipeline completed in %s", format_duration(total_elapsed))
        else:
            LOGGER.info("Whole pipeline aborted after %s", format_duration(total_elapsed))


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    try:
        result = run_pipeline(args)
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        if args.verbose:
            raise
        return 1
    if result.generated_outputs and len(result.generated_outputs) > 1:
        LOGGER.info("Wrote %d merged outputs to %s", len(result.generated_outputs), result.display_path or result.output_path)
    else:
        LOGGER.info("Wrote merged output to %s", result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())