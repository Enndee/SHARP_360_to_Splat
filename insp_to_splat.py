from __future__ import annotations

import argparse
import importlib
import json
import logging
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

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
DEFAULT_CONFIG_PATH = resolve_resource_path("insp_settings.json")
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_DA360_CHECKPOINT_PATH = resolve_resource_path("checkpoints", "DA360_large.pth")
GUI_SETTINGS_PATH = resolve_resource_path("easy_360_sharp_gui_settings.json")
SEEDVR2_SETTINGS_PATH = resolve_resource_path("seedvr2_settings.json")
SEEDVR2_CLI_PATH = resolve_resource_path("seedvr2_videoupscaler", "inference_cli.py")
DEFAULT_IMAGEMAGICK_COMMANDS = "-auto-level -auto-gamma -normalize -enhance -despeckle -unsharp 0x1.2+0.8+0.02"
SUPPORTED_INPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}
COMPRESSED_OUTPUT_SUFFIXES = {".spx", ".spz", ".sog"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


if str(LOCAL_SHARP_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SHARP_SRC))

if TYPE_CHECKING:
    from sharp.utils.gaussians import Gaussians3D


LOGGER = logging.getLogger("insp_to_splat")


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


def find_imagemagick_executable(path_value: str | Path | None) -> Path | None:
    resolved = resolve_optional_path(path_value)
    if resolved is not None:
        if resolved.exists():
            return resolved
        return None
    magick_on_path = shutil.which("magick")
    if magick_on_path:
        return Path(magick_on_path)
    return None


def optimize_panorama_with_imagemagick(
    panorama: np.ndarray,
    temp_root: Path,
    magick_path: Path,
    command_string: str,
) -> np.ndarray:
    preprocess_dir = temp_root / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    input_path = preprocess_dir / "panorama_input.png"
    output_path = preprocess_dir / "panorama_optimized.png"
    Image.fromarray(panorama).save(input_path)

    command = [str(magick_path), str(input_path), *shlex.split(command_string), str(output_path)]
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
    face_size: int,
    temp_root: Path,
) -> tuple[dict[str, np.ndarray], int]:
    settings = load_seedvr2_settings()
    factor = int(settings.get("resolution_factor", 2))
    target_resolution = face_size * factor

    input_dir = temp_root / "seedvr2_input"
    output_dir = temp_root / "seedvr2_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, image_array in faces.items():
        Image.fromarray(image_array).save(input_dir / f"{name}.png")

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

    max_res = int(settings.get("max_resolution", 0))
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
        "Running SeedVR2 upscale: %d faces, %dx%d -> %dx%d (factor %d)",
        len(faces), face_size, face_size, target_resolution, target_resolution, factor,
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
            upscaled[name] = np.asarray(img.convert("RGB")).copy()

    new_face_size = upscaled[next(iter(upscaled))].shape[0]
    LOGGER.info("SeedVR2 upscale complete. New face size: %dx%d", new_face_size, new_face_size)
    return upscaled, new_face_size


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
    if path_value in {None, "", False}:
        return None
    path = Path(path_value)
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
    return load_panorama(path)


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


def build_extraction_layout(face_size: int, side_count: int, config: dict) -> ExtractionLayout:
    view_fov_degrees = resolve_view_fov_degrees(side_count, config)
    if not (45.0 <= view_fov_degrees < 179.0):
        raise ValueError("Resolved view FOV must be between 45 and 179 degrees.")
    focal_px = (face_size / 2.0) / np.tan(np.deg2rad(view_fov_degrees) / 2.0)
    views = tuple(make_horizon_view(index, side_count) for index in range(side_count))
    return ExtractionLayout(f"horizon{side_count}", views, focal_px)


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


def align_gaussians_to_reference(
    gaussians: Gaussians3D,
    reference_disparity_view: np.ndarray,
    focal_px: float,
    image_size: int,
) -> tuple[Gaussians3D, float, int]:
    """Align Gaussian depths to DA360 using a smooth low-frequency scale field.

    Instead of overriding every Gaussian's depth with DA360 (which destroys
    SHARP's fine detail), we:
      1. Compute per-Gaussian ``da360_depth / sharp_depth`` ratios.
      2. Bin those ratios into a coarse spatial grid and take the
         robust median per cell.
      3. Bilinearly interpolate the coarse grid back to each Gaussian.

    This corrects large-scale depth structure (fixing ground-level alignment
    between views) while preserving SHARP's high-frequency depth detail
    (object shapes, surface relief, etc.).

    Returns ``(aligned_gaussians, median_scale, sample_count)``.
    """
    from sharp.utils.gaussians import Gaussians3D

    grid_cells = 8  # 8x8 spatial bins — controls alignment granularity

    mean_vectors = gaussians.mean_vectors  # (1, N, 3)
    mv_np = mean_vectors[0].detach().cpu().numpy().astype(np.float32)
    depth_z = mv_np[:, 2]
    radial = np.linalg.norm(mv_np, axis=1)

    valid = depth_z > 1e-6
    pixel_x = (mv_np[:, 0] / np.clip(depth_z, 1e-6, None)) * focal_px + (image_size / 2.0) - 0.5
    pixel_y = (mv_np[:, 1] / np.clip(depth_z, 1e-6, None)) * focal_px + (image_size / 2.0) - 0.5
    valid &= (pixel_x >= 0) & (pixel_x <= image_size - 1)
    valid &= (pixel_y >= 0) & (pixel_y <= image_size - 1)

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

            cell_size = image_size / grid_cells
            grid = np.full((grid_cells, grid_cells), median_scale, dtype=np.float32)
            for gy in range(grid_cells):
                for gx in range(grid_cells):
                    in_cell = (
                        (px_ok >= gx * cell_size) & (px_ok < (gx + 1) * cell_size)
                        & (py_ok >= gy * cell_size) & (py_ok < (gy + 1) * cell_size)
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
            all_px = np.clip(all_px, 0, image_size - 1)
            all_py = np.clip(all_py, 0, image_size - 1)
            gx_cont = all_px / cell_size - 0.5
            gy_cont = all_py / cell_size - 0.5

            # Bilinear interpolation on the coarse grid.
            gx0 = np.clip(np.floor(gx_cont).astype(np.int32), 0, grid_cells - 1)
            gy0 = np.clip(np.floor(gy_cont).astype(np.int32), 0, grid_cells - 1)
            gx1 = np.clip(gx0 + 1, 0, grid_cells - 1)
            gy1 = np.clip(gy0 + 1, 0, grid_cells - 1)
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
    image_size: int,
    focal_px: float,
    view: FaceOrientation,
) -> np.ndarray:
    pixel_coords = np.arange(image_size, dtype=np.float32) + 0.5
    centered = (pixel_coords - image_size / 2.0) / focal_px
    grid_x, grid_y = np.meshgrid(centered, centered)

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


def extract_perspective_views(layout: ExtractionLayout, panorama: np.ndarray, image_size: int) -> dict[str, np.ndarray]:
    return {
        view.name: extract_perspective_view(panorama, image_size, layout.focal_px, view)
        for view in layout.views
    }


def extract_perspective_scalar_view(
    panorama: np.ndarray,
    image_size: int,
    focal_px: float,
    view: FaceOrientation,
) -> np.ndarray:
    pixel_coords = np.arange(image_size, dtype=np.float32) + 0.5
    centered = (pixel_coords - image_size / 2.0) / focal_px
    grid_x, grid_y = np.meshgrid(centered, centered)

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
    image_size: int,
) -> dict[str, np.ndarray]:
    return {
        view.name: extract_perspective_scalar_view(panorama, image_size, layout.focal_px, view)
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


def save_intermediate_face_splats(face_gaussians: dict[str, Gaussians3D], focal_px: float, face_size: int, directory: Path) -> None:
    from sharp.utils.gaussians import save_ply

    directory.mkdir(parents=True, exist_ok=True)
    for name, gaussians in face_gaussians.items():
        save_ply(gaussians, focal_px, (face_size, face_size), directory / f"{name}.ply")


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


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    from sharp.cli.predict import predict_image
    from sharp.utils.gaussians import apply_transform, save_ply

    config = load_config(args.config)
    output_path, output_format = ensure_output_format(args.output, args.format, config)
    side_count = resolve_side_count(getattr(args, "side_count", 0), config)
    da360_alignment_enabled = resolve_da360_alignment_enabled(args, config)
    da360_checkpoint_path = resolve_da360_checkpoint_path(args, config) if da360_alignment_enabled else None
    quality = args.quality if args.quality is not None else int(config.get("default_quality", 9))
    sh_degree = args.sh_degree if args.sh_degree is not None else int(config.get("default_sh_degree", 0))
    clip_horizontal_raw = config.get("merge_clip_horizontal_degrees", 0)
    clip_horizontal_degrees = (360.0 / side_count) if clip_horizontal_raw in {None, 0, 0.0, "", False} else float(clip_horizontal_raw)
    clip_vertical_raw = config.get("merge_clip_vertical_degrees")
    clip_vertical_degrees = None if clip_vertical_raw in {None, 0, 0.0, "", False} else float(clip_vertical_raw)
    if not (1 <= quality <= 9):
        raise ValueError("Quality must be between 1 and 9.")
    if not (0 <= sh_degree <= 3):
        raise ValueError("SH degree must be between 0 and 3.")
    if not (1.0 <= clip_horizontal_degrees <= 180.0):
        raise ValueError("merge_clip_horizontal_degrees must be between 1 and 180.")
    if clip_vertical_degrees is not None and not (1.0 <= clip_vertical_degrees <= 180.0):
        raise ValueError("merge_clip_vertical_degrees must be between 1 and 180 when enabled.")

    device = resolve_device(args.device)
    LOGGER.info("Using device: %s", device)

    temp_root = choose_temp_root(args)
    register_optional_image_plugins()
    panorama = load_input_panorama(args.input)
    panorama_width, panorama_height = validate_equirectangular_shape(panorama)
    seedvr2_upscale_enabled = getattr(args, "enable_seedvr2_upscale", False)
    imagemagick_optimization_enabled = getattr(args, "enable_imagemagick_optimization", False)

    if imagemagick_optimization_enabled:
        magick_path = find_imagemagick_executable(getattr(args, "imagemagick", None))
        if magick_path is None:
            raise FileNotFoundError("ImageMagick optimization is enabled, but magick.exe was not found. Provide --imagemagick or add ImageMagick to PATH.")
        imagemagick_commands = getattr(args, "imagemagick_commands", DEFAULT_IMAGEMAGICK_COMMANDS) or DEFAULT_IMAGEMAGICK_COMMANDS
        LOGGER.info("Optimizing panorama with ImageMagick before slicing.")
        panorama = optimize_panorama_with_imagemagick(panorama, temp_root, magick_path, imagemagick_commands)
        panorama_width, panorama_height = validate_equirectangular_shape(panorama)
    elif seedvr2_upscale_enabled:
        LOGGER.info("Applying sharpening to panorama before face extraction.")
        panorama = sharpen_panorama(panorama)

    face_size = resolve_face_size(args.face_size, panorama_width, side_count, config)
    extraction_layout = build_extraction_layout(face_size, side_count, config)
    focal_px = extraction_layout.focal_px

    LOGGER.info(
        "Loaded panorama %s with resolution %dx%d. Extracting %d %dx%d perspective views using %s.",
        args.input,
        panorama_width,
        panorama_height,
        len(extraction_layout.views),
        face_size,
        face_size,
        extraction_layout.name,
    )

    faces = extract_perspective_views(extraction_layout, panorama, face_size)
    intermediate_dir = choose_intermediate_dir(args, output_path)
    if intermediate_dir is not None:
        save_intermediate_face_images(faces, intermediate_dir / "faces")

    if seedvr2_upscale_enabled:
        original_face_size = face_size
        faces, face_size = upscale_faces_with_seedvr2(faces, face_size, temp_root)
        focal_px = extraction_layout.focal_px * (face_size / original_face_size)
        if intermediate_dir is not None:
            save_intermediate_face_images(faces, intermediate_dir / "faces_upscaled")

    depth_map_path: Path | None = None
    reference_depth_views: dict[str, np.ndarray] = {}
    if da360_alignment_enabled:
        LOGGER.info("Running DA360 panorama depth inference using checkpoint %s", da360_checkpoint_path)
        da360_predictor = build_da360_predictor(da360_checkpoint_path, device)
        reference_depth_panorama = predict_da360_disparity_panorama(da360_predictor, panorama, device)
        reference_depth_views = {
            view.name: extract_perspective_scalar_view(reference_depth_panorama, face_size, focal_px, view)
            for view in extraction_layout.views
        }
        del da360_predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        depth_map_path = temp_root / "depth" / f"{args.input.stem}_depth.png"
        save_depth_visualization(reference_depth_panorama, depth_map_path)
        if intermediate_dir is not None:
            depth_vis_dir = intermediate_dir / "depth_views"
            for view_name, depth_view in reference_depth_views.items():
                save_depth_visualization(depth_view, depth_vis_dir / f"{view_name}_depth.png")

    predictor = build_predictor(args.checkpoint, device)
    original_median_radii: list[float] = []
    rotated_face_gaussians: dict[str, Gaussians3D] = {}
    rotated_gaussian_list: list[Gaussians3D] = []

    for view in extraction_layout.views:
        LOGGER.info("Predicting SHARP splats for view: %s", view.name)
        gaussians = predict_image(predictor, faces[view.name], focal_px, device)
        gaussians = filter_gaussians_by_view_border(
            gaussians,
            horizontal_border_degrees=clip_horizontal_degrees,
            vertical_border_degrees=clip_vertical_degrees,
        )
        if da360_alignment_enabled:
            orig_med = float(torch.median(torch.norm(
                gaussians.mean_vectors, dim=-1,
            )).item())
            original_median_radii.append(orig_med)
            gaussians, median_scale, sample_count = align_gaussians_to_reference(
                gaussians,
                reference_depth_views[view.name],
                focal_px=focal_px,
                image_size=face_size,
            )
            LOGGER.info(
                "Aligned %s to DA360 depth: median_scale=%.6f "
                "(%d samples, orig_median_r=%.2f).",
                view.name, median_scale, sample_count, orig_med,
            )
        rotated = apply_transform(gaussians, face_transform_tensor(view, device)).to(torch.device("cpu"))
        rotated_face_gaussians[view.name] = rotated
        rotated_gaussian_list.append(rotated)

    if intermediate_dir is not None:
        save_intermediate_face_splats(rotated_face_gaussians, focal_px, face_size, intermediate_dir / "face_splats")

    merged = merge_gaussians(rotated_gaussian_list)

    # After per-view alignment the scene is in DA360's depth units (small).
    # Apply a uniform global scale so the output matches the original SHARP
    # magnitude — this does NOT affect inter-view consistency.
    if da360_alignment_enabled and original_median_radii:
        original_scene_median = float(np.median(original_median_radii))
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "ply":
        LOGGER.info("Saving merged PLY to %s", output_path)
        save_ply(merged, focal_px, (face_size, face_size), output_path)
        return PipelineResult(output_path=output_path, depth_map_path=depth_map_path)

    gsbox_path = find_gsbox(args.gsbox)
    if gsbox_path is None:
        raise FileNotFoundError(
            "Compressed output requires gsbox.exe. Provide --gsbox or place gsbox.exe next to the script."
        )

    conversion_dir = temp_root / "conversion"
    conversion_dir.mkdir(parents=True, exist_ok=True)
    temp_ply = conversion_dir / f"{output_path.stem}.ply"
    save_ply(merged, focal_px, (face_size, face_size), temp_ply)
    convert_with_gsbox(temp_ply, output_path, output_format, quality, sh_degree, gsbox_path)
    return PipelineResult(output_path=output_path, depth_map_path=depth_map_path)


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
    LOGGER.info("Wrote merged output to %s", result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())