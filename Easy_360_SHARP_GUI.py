from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageOps, ImageTk

import insp_to_splat


BG = "#101418"
BG2 = "#171c22"
BG3 = "#1d242c"
BG4 = "#2a333d"
ACCENT = "#78b7c9"
ACCENT_DIM = "#4f8594"
FG = "#edf3f7"
FG_DIM = "#9aa7b4"
ERR = "#ff7f7f"
WARN = "#d7b36a"
INPUT_BG = "#11171c"
INPUT_BORDER = "#3a4753"
LIST_BG = "#10161b"
CARD_BORDER = "#303c47"
PREVIEW_SIZE = (360, 220)

ROOT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT_DIR / "easy_360_sharp_gui_settings.json"
PIPELINE_DEFAULTS = insp_to_splat.load_config(insp_to_splat.DEFAULT_CONFIG_PATH)
SEEDVR2_DEFAULTS = insp_to_splat.load_config(insp_to_splat.SEEDVR2_SETTINGS_PATH)
DEFAULT_GUI_QUALITY = int(PIPELINE_DEFAULTS.get("default_quality", 9))
DEFAULT_GUI_SH_DEGREE = int(PIPELINE_DEFAULTS.get("default_sh_degree", 0))


def load_settings() -> dict:
    defaults = {
        "input_path": "",
        "output_path": "",
        "last_browse_folder": str(ROOT_DIR),
        "side_count": "4",
        "format": "ply",
        "device": "default",
        "checkpoint": "",
        "da360_checkpoint": str(insp_to_splat.DEFAULT_DA360_CHECKPOINT_PATH),
        "gsbox": "",
        "intermediate_dir": "",
        "enable_da360_alignment": True,
        "alignment_mode": str(PIPELINE_DEFAULTS.get("default_alignment_mode", "da360" if PIPELINE_DEFAULTS.get("default_enable_da360_alignment", True) else "none")),
        "alignment_grid_resolution": 4,
        "alignment_detail_weight": 0.0,
        "cutoff_height_percent": 0.0,
        "enable_alignment_sweep": False,
        "keep_intermediates": False,
        "delete_temp_files": True,
        "enable_seedvr2_upscale": False,
        "enable_imagemagick_optimization": True,
        "imagemagick_path": "",
        "enable_deblur_preprocessing": False,
        "deblur_strength": "medium",
        "seedvr2_model_name": str(SEEDVR2_DEFAULTS.get("model_name", "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors")),
        "seedvr2_output_format": str(SEEDVR2_DEFAULTS.get("output_format", "png")),
        "seedvr2_color_correction": str(SEEDVR2_DEFAULTS.get("color_correction", "lab")),
        "seedvr2_attention_mode": str(SEEDVR2_DEFAULTS.get("attention_mode", "sageattn_2")),
        "seedvr2_cuda_device": str(SEEDVR2_DEFAULTS.get("cuda_device", "0")),
        "seedvr2_dit_offload_device": str(SEEDVR2_DEFAULTS.get("dit_offload_device", "cpu")),
        "seedvr2_vae_offload_device": str(SEEDVR2_DEFAULTS.get("vae_offload_device", "cpu")),
        "seedvr2_tensor_offload_device": str(SEEDVR2_DEFAULTS.get("tensor_offload_device", "cpu")),
        "seedvr2_resolution_factor": str(SEEDVR2_DEFAULTS.get("resolution_factor", "2")),
        "seedvr2_stretch_proportions": bool(SEEDVR2_DEFAULTS.get("stretch_proportions", False)),
        "seedvr2_target_short_side": str(SEEDVR2_DEFAULTS.get("target_short_side", "0")),
        "seedvr2_min_resolution": str(SEEDVR2_DEFAULTS.get("min_resolution", "0")),
        "seedvr2_max_resolution": str(SEEDVR2_DEFAULTS.get("max_resolution", "0")),
        "seedvr2_batch_size": str(SEEDVR2_DEFAULTS.get("batch_size", "1")),
        "seedvr2_seed": str(SEEDVR2_DEFAULTS.get("seed", "42")),
        "seedvr2_skip_first_frames": str(SEEDVR2_DEFAULTS.get("skip_first_frames", "0")),
        "seedvr2_blocks_to_swap": str(SEEDVR2_DEFAULTS.get("blocks_to_swap", "36")),
        "seedvr2_vae_encode_tile_size": str(SEEDVR2_DEFAULTS.get("vae_encode_tile_size", "1024")),
        "seedvr2_vae_encode_tile_overlap": str(SEEDVR2_DEFAULTS.get("vae_encode_tile_overlap", "128")),
        "seedvr2_vae_decode_tile_size": str(SEEDVR2_DEFAULTS.get("vae_decode_tile_size", "1024")),
        "seedvr2_vae_decode_tile_overlap": str(SEEDVR2_DEFAULTS.get("vae_decode_tile_overlap", "128")),
        "seedvr2_compile_backend": str(SEEDVR2_DEFAULTS.get("compile_backend", "inductor")),
        "seedvr2_compile_mode": str(SEEDVR2_DEFAULTS.get("compile_mode", "default")),
        "seedvr2_swap_io_components": bool(SEEDVR2_DEFAULTS.get("swap_io_components", True)),
        "seedvr2_vae_encode_tiled": bool(SEEDVR2_DEFAULTS.get("vae_encode_tiled", True)),
        "seedvr2_vae_decode_tiled": bool(SEEDVR2_DEFAULTS.get("vae_decode_tiled", True)),
        "seedvr2_cache_dit": bool(SEEDVR2_DEFAULTS.get("cache_dit", True)),
        "seedvr2_cache_vae": bool(SEEDVR2_DEFAULTS.get("cache_vae", True)),
        "seedvr2_debug_enabled": bool(SEEDVR2_DEFAULTS.get("debug_enabled", False)),
    }
    defaults.update(insp_to_splat.DEFAULT_IMAGEMAGICK_GUI_SETTINGS)
    defaults.update(insp_to_splat.DEFAULT_DEBLUR_GUI_SETTINGS)
    loaded_preset_name = None
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                loaded_preset_name = loaded.get("imagemagick_preset")
                defaults.update(loaded)
                defaults.update(insp_to_splat.parse_imagemagick_gui_settings(loaded))
        except Exception:
            pass
    inferred_preset = insp_to_splat.infer_imagemagick_preset_name(defaults)
    if loaded_preset_name is None and inferred_preset == "classic":
        defaults.update(insp_to_splat.get_imagemagick_preset_settings(insp_to_splat.DEFAULT_IMAGEMAGICK_PRESET))
        defaults["imagemagick_preset"] = insp_to_splat.DEFAULT_IMAGEMAGICK_PRESET
    else:
        defaults["imagemagick_preset"] = inferred_preset
    detected_magick = insp_to_splat.find_imagemagick_executable(defaults.get("imagemagick_path", ""))
    if detected_magick is not None:
        defaults["imagemagick_path"] = str(detected_magick)
    try:
        defaults["alignment_mode"] = insp_to_splat.normalize_alignment_mode(defaults.get("alignment_mode", "da360"))
    except ValueError:
        defaults["alignment_mode"] = str(
            PIPELINE_DEFAULTS.get(
                "default_alignment_mode",
                "da360" if PIPELINE_DEFAULTS.get("default_enable_da360_alignment", True) else "none",
            )
        )
    return defaults


def save_settings(data: dict) -> None:
    try:
        if "alignment_mode" in data:
            try:
                data = dict(data)
                data["alignment_mode"] = insp_to_splat.normalize_alignment_mode(data.get("alignment_mode"))
            except ValueError:
                data = dict(data)
                data["alignment_mode"] = str(
                    PIPELINE_DEFAULTS.get(
                        "default_alignment_mode",
                        "da360" if PIPELINE_DEFAULTS.get("default_enable_da360_alignment", True) else "none",
                    )
                )
        with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    except Exception:
        pass


class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue[tuple[str, str]]):
        super().__init__()
        self._log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            level = record.levelname.lower()
            self._log_queue.put((level, message))
        except Exception:
            pass


class HoverTooltip:
    def __init__(self, widget, text: str, *, delay_ms: int = 450):
        self.widget = widget
        self.text = text.strip()
        self.delay_ms = delay_ms
        self._after_id = None
        self._window: tk.Toplevel | None = None
        if not self.text:
            return
        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")
        self.widget.bind("<Destroy>", self._on_leave, add="+")

    def _on_enter(self, _event=None) -> None:
        self._cancel_scheduled()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, _event=None) -> None:
        self._cancel_scheduled()
        self._hide()

    def _cancel_scheduled(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self._window is not None or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        window.configure(bg=INPUT_BORDER)
        label = tk.Label(
            window,
            text=self.text,
            justify="left",
            anchor="w",
            bg=BG3,
            fg=FG,
            padx=10,
            pady=6,
            wraplength=340,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
        )
        label.pack()
        self._window = window

    def _hide(self) -> None:
        if self._window is not None:
            self._window.destroy()
            self._window = None


class App(tk.Tk):
    def __init__(self, initial_input_paths: list[Path] | None = None):
        super().__init__()
        self.title("SHARP_360_to_Splat")
        self.configure(bg=BG)
        self.geometry("1380x900")
        self.minsize(980, 760)

        settings = load_settings()
        self._worker: threading.Thread | None = None
        self._running = False
        self._log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._file_paths: list[Path] = []
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._advanced_window: tk.Toplevel | None = None
        self._scroll_canvas: tk.Canvas | None = None
        self._advanced_scroll_canvas: tk.Canvas | None = None
        self._log_link_index = 0
        self._suspend_input_sync = False
        self._tooltips: list[HoverTooltip] = []
        self._initial_input_paths = [path for path in (initial_input_paths or []) if path.exists()]
        self._applying_imagemagick_preset = False

        self.browse_folder_var = tk.StringVar(value=settings["last_browse_folder"])
        self.input_var = tk.StringVar(value=settings["input_path"])
        self.output_var = tk.StringVar(value=settings["output_path"])
        self.side_count_var = tk.StringVar(value=settings["side_count"])
        self.format_var = tk.StringVar(value=settings["format"])
        self.device_var = tk.StringVar(value=settings["device"])
        self.checkpoint_var = tk.StringVar(value=settings["checkpoint"])
        self.da360_checkpoint_var = tk.StringVar(value=settings["da360_checkpoint"])
        self.gsbox_var = tk.StringVar(value=settings["gsbox"])
        self.intermediate_dir_var = tk.StringVar(value=settings["intermediate_dir"])
        self.enable_da360_alignment_var = tk.BooleanVar(value=bool(settings["enable_da360_alignment"]))
        self.alignment_mode_var = tk.StringVar(value=str(settings["alignment_mode"]))
        self.alignment_grid_resolution_var = tk.IntVar(value=int(settings["alignment_grid_resolution"]))
        self.alignment_detail_weight_var = tk.DoubleVar(value=float(settings["alignment_detail_weight"]))
        self.cutoff_height_percent_var = tk.DoubleVar(value=float(settings["cutoff_height_percent"]))
        self.enable_alignment_sweep_var = tk.BooleanVar(value=bool(settings["enable_alignment_sweep"]))
        self.keep_intermediates_var = tk.BooleanVar(value=bool(settings["keep_intermediates"]))
        self.delete_temp_files_var = tk.BooleanVar(value=bool(settings["delete_temp_files"]))
        self.enable_seedvr2_upscale_var = tk.BooleanVar(value=bool(settings["enable_seedvr2_upscale"]))
        self.enable_imagemagick_optimization_var = tk.BooleanVar(value=bool(settings["enable_imagemagick_optimization"]))
        self.enable_deblur_preprocessing_var = tk.BooleanVar(value=bool(settings["enable_deblur_preprocessing"]))
        self.deblur_strength_var = tk.StringVar(value=str(settings["deblur_strength"]))
        self.imagemagick_path_var = tk.StringVar(value=settings["imagemagick_path"])
        self.imagemagick_preset_var = tk.StringVar(value=str(settings["imagemagick_preset"]))
        self.imagemagick_auto_level_var = tk.BooleanVar(value=bool(settings["imagemagick_auto_level"]))
        self.imagemagick_auto_gamma_var = tk.BooleanVar(value=bool(settings["imagemagick_auto_gamma"]))
        self.imagemagick_normalize_var = tk.BooleanVar(value=bool(settings["imagemagick_normalize"]))
        self.imagemagick_enhance_var = tk.BooleanVar(value=bool(settings["imagemagick_enhance"]))
        self.imagemagick_despeckle_var = tk.BooleanVar(value=bool(settings["imagemagick_despeckle"]))
        self.imagemagick_unsharp_enabled_var = tk.BooleanVar(value=bool(settings["imagemagick_unsharp_enabled"]))
        self.imagemagick_unsharp_value_var = tk.StringVar(value=str(settings["imagemagick_unsharp_value"]))
        self.imagemagick_extra_args_var = tk.StringVar(value=str(settings["imagemagick_extra_args"]))
        self.seedvr2_model_name_var = tk.StringVar(value=settings["seedvr2_model_name"])
        self.seedvr2_output_format_var = tk.StringVar(value=settings["seedvr2_output_format"])
        self.seedvr2_color_correction_var = tk.StringVar(value=settings["seedvr2_color_correction"])
        self.seedvr2_attention_mode_var = tk.StringVar(value=settings["seedvr2_attention_mode"])
        self.seedvr2_cuda_device_var = tk.StringVar(value=settings["seedvr2_cuda_device"])
        self.seedvr2_dit_offload_device_var = tk.StringVar(value=settings["seedvr2_dit_offload_device"])
        self.seedvr2_vae_offload_device_var = tk.StringVar(value=settings["seedvr2_vae_offload_device"])
        self.seedvr2_tensor_offload_device_var = tk.StringVar(value=settings["seedvr2_tensor_offload_device"])
        self.seedvr2_resolution_factor_var = tk.StringVar(value=settings["seedvr2_resolution_factor"])
        self.seedvr2_stretch_proportions_var = tk.BooleanVar(value=bool(settings["seedvr2_stretch_proportions"]))
        self.seedvr2_target_short_side_var = tk.StringVar(value=settings["seedvr2_target_short_side"])
        self.seedvr2_min_resolution_var = tk.StringVar(value=settings["seedvr2_min_resolution"])
        self.seedvr2_max_resolution_var = tk.StringVar(value=settings["seedvr2_max_resolution"])
        self.seedvr2_batch_size_var = tk.StringVar(value=settings["seedvr2_batch_size"])
        self.seedvr2_seed_var = tk.StringVar(value=settings["seedvr2_seed"])
        self.seedvr2_skip_first_frames_var = tk.StringVar(value=settings["seedvr2_skip_first_frames"])
        self.seedvr2_blocks_to_swap_var = tk.StringVar(value=settings["seedvr2_blocks_to_swap"])
        self.seedvr2_vae_encode_tile_size_var = tk.StringVar(value=settings["seedvr2_vae_encode_tile_size"])
        self.seedvr2_vae_encode_tile_overlap_var = tk.StringVar(value=settings["seedvr2_vae_encode_tile_overlap"])
        self.seedvr2_vae_decode_tile_size_var = tk.StringVar(value=settings["seedvr2_vae_decode_tile_size"])
        self.seedvr2_vae_decode_tile_overlap_var = tk.StringVar(value=settings["seedvr2_vae_decode_tile_overlap"])
        self.seedvr2_compile_backend_var = tk.StringVar(value=settings["seedvr2_compile_backend"])
        self.seedvr2_compile_mode_var = tk.StringVar(value=settings["seedvr2_compile_mode"])
        self.seedvr2_swap_io_components_var = tk.BooleanVar(value=bool(settings["seedvr2_swap_io_components"]))
        self.seedvr2_vae_encode_tiled_var = tk.BooleanVar(value=bool(settings["seedvr2_vae_encode_tiled"]))
        self.seedvr2_vae_decode_tiled_var = tk.BooleanVar(value=bool(settings["seedvr2_vae_decode_tiled"]))
        self.seedvr2_cache_dit_var = tk.BooleanVar(value=bool(settings["seedvr2_cache_dit"]))
        self.seedvr2_cache_vae_var = tk.BooleanVar(value=bool(settings["seedvr2_cache_vae"]))
        self.seedvr2_debug_enabled_var = tk.BooleanVar(value=bool(settings["seedvr2_debug_enabled"]))
        self.status_var = tk.StringVar(value="Ready")
        self.selection_var = tk.StringVar(value="Select a stitched panorama from the file browser.")
        self.output_hint_var = tk.StringVar(value="Final output will be written next to the source image.")
        self.temp_hint_var = tk.StringVar(value="Working files will be written under Source/Temp.")

        self._build_styles()
        self._build_ui()
        self._bind_persistence()
        self._auto_detect_imagemagick_path()
        self._sync_temp_cleanup_state()
        self._sync_output_extension()
        initial_select_path = self.input_var.get().strip()
        if self._initial_input_paths:
            first_path = self._initial_input_paths[0]
            self.browse_folder_var.set(str(first_path.parent))
            initial_select_path = str(first_path)
        self._refresh_file_list(select_path=initial_select_path)
        if self._initial_input_paths:
            self._select_file_paths(self._initial_input_paths)
        self._update_preview()
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Control-a>", self._select_all_files, add="+")
        self.after(100, self._drain_logs)

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 18, "bold"), background=BG, foreground=FG)
        style.configure("Subtle.TLabel", background=BG, foreground=FG_DIM)
        style.configure("Panel.TFrame", background=BG2)
        style.configure("PanelTitle.TLabel", background=BG2, foreground=FG, font=("Segoe UI", 11, "bold"))
        style.configure("TButton", background=BG4, foreground=FG, borderwidth=0, padding=8)
        style.map("TButton", background=[("active", ACCENT_DIM), ("pressed", ACCENT)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#081117")
        style.map("Accent.TButton", background=[("active", "#96cfdf"), ("pressed", ACCENT_DIM)])
        style.configure("TEntry", fieldbackground=INPUT_BG, foreground=FG, bordercolor=INPUT_BORDER)
        style.configure(
            "TCombobox",
            fieldbackground=INPUT_BG,
            background=INPUT_BG,
            foreground=FG,
            arrowcolor=FG,
            bordercolor=INPUT_BORDER,
            selectbackground=ACCENT_DIM,
            selectforeground=FG,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", INPUT_BG), ("disabled", BG3)],
            background=[("readonly", INPUT_BG), ("active", BG3)],
            foreground=[("readonly", FG), ("disabled", FG_DIM)],
            selectbackground=[("readonly", ACCENT_DIM)],
            selectforeground=[("readonly", FG)],
            arrowcolor=[("readonly", FG), ("active", FG)],
            bordercolor=[("focus", ACCENT_DIM), ("readonly", INPUT_BORDER)],
        )
        self.option_add("*TCombobox*Listbox.background", INPUT_BG)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT_DIM)
        self.option_add("*TCombobox*Listbox.selectForeground", FG)
        style.configure(
            "Easy.Horizontal.TProgressbar",
            troughcolor=BG3,
            background=ACCENT,
            bordercolor=BG3,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            thickness=12,
        )

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=18)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title_block = ttk.Frame(header)
        title_block.grid(row=0, column=0, sticky="w")
        ttk.Label(title_block, text="SHARP_360_to_Splat", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            title_block,
            text="Browse stitched panoramas, preview the source image, and export the merged splat beside it while Temp files stay near the source.",
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        self.settings_button = tk.Button(
            header,
            text="\u2699",
            command=self._open_advanced_settings,
            bg=BG3,
            fg=FG,
            activebackground=ACCENT_DIM,
            activeforeground=FG,
            relief="flat",
            borderwidth=0,
            font=("Segoe UI Symbol", 15),
            padx=12,
            pady=6,
            cursor="hand2",
        )
        self.settings_button.grid(row=0, column=1, sticky="ne")

        main_panel = ttk.Frame(container, style="Panel.TFrame")
        main_panel.grid(row=1, column=0, sticky="nsew", pady=(14, 0), padx=(0, 12))
        main_panel.columnconfigure(0, weight=1)
        main_panel.rowconfigure(0, weight=1)

        self._scroll_canvas = tk.Canvas(main_panel, bg=BG2, highlightthickness=0, borderwidth=0)
        self._scroll_canvas.grid(row=0, column=0, sticky="nsew")
        main_scrollbar = ttk.Scrollbar(main_panel, orient="vertical", command=self._scroll_canvas.yview)
        main_scrollbar.grid(row=0, column=1, sticky="ns")
        self._scroll_canvas.configure(yscrollcommand=main_scrollbar.set)

        form_panel = ttk.Frame(self._scroll_canvas, style="Panel.TFrame", padding=16)
        form_panel.columnconfigure(1, weight=1)
        canvas_window = self._scroll_canvas.create_window((0, 0), window=form_panel, anchor="nw")
        form_panel.bind("<Configure>", lambda event: self._on_scrollable_configure(event, canvas_window))
        self._scroll_canvas.bind("<Configure>", lambda event: self._scroll_canvas.itemconfigure(canvas_window, width=event.width))
        log_panel = ttk.Frame(container, style="Panel.TFrame", padding=16)
        log_panel.grid(row=1, column=1, rowspan=2, sticky="nsew", pady=(14, 0))
        log_panel.columnconfigure(0, weight=1)
        log_panel.rowconfigure(1, weight=1)

        ttk.Label(form_panel, text="Source Browser", style="PanelTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._browse_row(form_panel, 1, "Source folder", self.browse_folder_var, self._browse_folder, tooltip="Folder that is scanned for stitched 2:1 panoramas shown in the file list.")
        self._browse_row(form_panel, 2, "Input panorama", self.input_var, self._browse_input, tooltip="Currently selected panorama file to process. A 2:1 equirectangular image is expected.")
        self._readonly_row(form_panel, 3, "Output splat", self.output_var, tooltip="Final merged splat output path. This is derived automatically from the selected input and format.")

        browser_frame = tk.Frame(form_panel, bg=BG2, highlightthickness=1, highlightbackground=CARD_BORDER)
        browser_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(10, 6))
        browser_frame.columnconfigure(0, weight=2)
        browser_frame.columnconfigure(1, weight=3)
        browser_frame.rowconfigure(1, weight=1)
        form_panel.rowconfigure(4, weight=1)

        tk.Label(browser_frame, text="Files", bg=BG2, fg=FG, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))
        tk.Label(browser_frame, text="Preview", bg=BG2, fg=FG, font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w", padx=12, pady=(10, 6))

        list_frame = tk.Frame(browser_frame, bg=BG2)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=(12, 8), pady=(0, 12))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.file_listbox = tk.Listbox(
            list_frame,
            bg=LIST_BG,
            fg=FG,
            selectbackground=ACCENT_DIM,
            selectforeground=FG,
            selectmode=tk.EXTENDED,
            exportselection=False,
            activestyle="none",
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
            font=("Segoe UI", 10),
        )
        file_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=file_scrollbar.set)
        self.file_listbox.grid(row=0, column=0, sticky="nsew")
        file_scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_listbox.bind("<<ListboxSelect>>", self._on_browser_select)

        preview_frame = tk.Frame(browser_frame, bg=BG2)
        preview_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 12), pady=(0, 12))
        preview_frame.columnconfigure(0, weight=1)
        self.preview_label = tk.Label(
            preview_frame,
            bg=LIST_BG,
            fg=FG_DIM,
            text="No preview",
            font=("Segoe UI", 11),
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
        )
        self.preview_label.grid(row=0, column=0, sticky="ew")
        tk.Label(preview_frame, textvariable=self.selection_var, justify="left", anchor="w", bg=BG2, fg=FG, wraplength=360).grid(row=1, column=0, sticky="ew", pady=(10, 4))
        tk.Label(preview_frame, textvariable=self.output_hint_var, justify="left", anchor="w", bg=BG2, fg=ACCENT, wraplength=360).grid(row=2, column=0, sticky="ew", pady=(0, 4))
        tk.Label(preview_frame, textvariable=self.temp_hint_var, justify="left", anchor="w", bg=BG2, fg=FG_DIM, wraplength=360).grid(row=3, column=0, sticky="ew")

        ttk.Label(form_panel, text="Options", style="PanelTitle.TLabel").grid(row=5, column=0, columnspan=3, sticky="w", pady=(18, 10))
        self._entry_row(form_panel, 6, "Sides", self.side_count_var, "2+ horizon views", tooltip="Number of perspective views extracted around the horizon. More sides improve coverage but increase runtime.")
        self._combo_row(form_panel, 7, "Format", self.format_var, ["ply", "spx", "spz", "sog"], tooltip="Output file format. Compressed formats require gsbox.exe and usually save disk space.")

        self.keep_check = tk.Checkbutton(
            form_panel,
            text="Keep intermediate face images and per-face splats",
            variable=self.keep_intermediates_var,
            bg=BG2,
            fg=FG,
            activebackground=BG2,
            activeforeground=FG,
            selectcolor=BG3,
            highlightthickness=0,
        )
        self.keep_check.grid(row=8, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self._attach_tooltip(self.keep_check, "Keep extracted faces, depth maps, and per-face splats instead of treating them as disposable working files.")

        self.delete_temp_check = tk.Checkbutton(
            form_panel,
            text="Delete Temp workspace automatically after processing",
            variable=self.delete_temp_files_var,
            bg=BG2,
            fg=FG,
            activebackground=BG2,
            activeforeground=FG,
            selectcolor=BG3,
            highlightthickness=0,
        )
        self.delete_temp_check.grid(row=9, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self._attach_tooltip(self.delete_temp_check, "Delete the Temp working directory automatically after a successful run when intermediates are not being kept.")

        self.da360_check = tk.Checkbutton(
            form_panel,
            text="Enable slice alignment after SHARP prediction",
            variable=self.enable_da360_alignment_var,
            bg=BG2,
            fg=FG,
            activebackground=BG2,
            activeforeground=FG,
            selectcolor=BG3,
            highlightthickness=0,
        )
        self.da360_check.grid(row=10, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self._attach_tooltip(self.da360_check, "Enable per-face alignment before merging. Use the mode selector below to choose DA360 depth alignment or overlap-based slice fitting.")

        # --- Depth alignment tuner sliders ---
        tuner_frame = tk.Frame(form_panel, bg=BG2)
        tuner_frame.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        tuner_frame.columnconfigure(1, weight=1)

        self._grid_res_label = tk.Label(
            tuner_frame, text="Grid Resolution: 4", bg=BG2, fg=FG, anchor="w",
        )
        self._grid_res_label.grid(row=0, column=0, sticky="w", padx=(18, 6))
        self._attach_tooltip(self._grid_res_label, "Controls how finely the DA360 alignment scale field is sampled. Higher values preserve more local structure but can follow noise more closely. Default: 4.")
        self.grid_res_scale = tk.Scale(
            tuner_frame,
            from_=1, to=64,
            orient=tk.HORIZONTAL,
            variable=self.alignment_grid_resolution_var,
            showvalue=False,
            bg=BG2, fg=FG, troughcolor=BG3, highlightthickness=0,
            command=self._on_grid_res_changed,
        )
        self.grid_res_scale.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self._on_grid_res_changed(self.alignment_grid_resolution_var.get())

        self._detail_wt_label = tk.Label(
            tuner_frame, text="Detail Preservation: 0 %", bg=BG2, fg=FG, anchor="w",
        )
        self._detail_wt_label.grid(row=1, column=0, sticky="w", padx=(18, 6))
        self._attach_tooltip(self._detail_wt_label, "Blends between smooth low-frequency alignment and per-point detail recovery. Higher values keep more small-scale SHARP depth structure. Default: 0%.")
        self.detail_wt_scale = tk.Scale(
            tuner_frame,
            from_=0, to=100,
            orient=tk.HORIZONTAL,
            variable=tk.IntVar(value=int(self.alignment_detail_weight_var.get() * 100)),
            showvalue=False,
            bg=BG2, fg=FG, troughcolor=BG3, highlightthickness=0,
            command=self._on_detail_wt_changed,
        )
        self._detail_wt_int_var = self.detail_wt_scale.cget("variable")
        self.detail_wt_scale.grid(row=1, column=1, sticky="ew", padx=(0, 6))
        self._on_detail_wt_changed(int(self.alignment_detail_weight_var.get() * 100))

        self._cutoff_height_label = tk.Label(
            tuner_frame, text="Cut-off Height: 0 %", bg=BG2, fg=FG, anchor="w",
        )
        self._cutoff_height_label.grid(row=2, column=0, sticky="w", padx=(18, 6))
        self._attach_tooltip(self._cutoff_height_label, "Cuts off the lower part of each extracted panorama slice before SHARP processing to avoid highly distorted lower regions. Default: 0%.")
        self.cutoff_height_scale = tk.Scale(
            tuner_frame,
            from_=0, to=40,
            orient=tk.HORIZONTAL,
            variable=tk.IntVar(value=int(self.cutoff_height_percent_var.get())),
            showvalue=False,
            bg=BG2, fg=FG, troughcolor=BG3, highlightthickness=0,
            command=self._on_cutoff_height_changed,
        )
        self._cutoff_height_int_var = self.cutoff_height_scale.cget("variable")
        self.cutoff_height_scale.grid(row=2, column=1, sticky="ew", padx=(0, 6))
        self._on_cutoff_height_changed(int(self.cutoff_height_percent_var.get()))

        self._alignment_mode_label = tk.Label(
            tuner_frame, text="Alignment Mode", bg=BG2, fg=FG, anchor="w",
        )
        self._alignment_mode_label.grid(row=3, column=0, sticky="w", padx=(18, 6), pady=(4, 0))
        self._attach_tooltip(self._alignment_mode_label, "Choose how neighboring face splats are aligned before merge. 'da360' uses the DA360 panorama depth reference. 'overlap' matches adjacent face splats directly in their overlap zone.")
        self.alignment_mode_combo = ttk.Combobox(
            tuner_frame,
            textvariable=self.alignment_mode_var,
            values=["da360", "overlap"],
            state="readonly",
        )
        self.alignment_mode_combo.grid(row=3, column=1, sticky="ew", padx=(0, 6), pady=(4, 0))
        self._attach_tooltip(self.alignment_mode_combo, "Switch between DA360-guided alignment and overlap-based slice fitting.")

        self.alignment_sweep_check = tk.Checkbutton(
            form_panel,
            text="Generate 25 alignment variants after one SHARP pass",
            variable=self.enable_alignment_sweep_var,
            bg=BG2,
            fg=FG,
            activebackground=BG2,
            activeforeground=FG,
            selectcolor=BG3,
            highlightthickness=0,
        )
        self.alignment_sweep_check.grid(row=12, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self._attach_tooltip(self.alignment_sweep_check, "Create a 5x5 sweep of Grid Resolution and Detail Preservation values after SHARP, SeedVR2, and DA360 have already run once.")

        self._sync_alignment_tuner_state()

        self.seedvr2_check = tk.Checkbutton(
            form_panel,
            text="Upscale face images with SeedVR2 before SHARP prediction",
            variable=self.enable_seedvr2_upscale_var,
            bg=BG2,
            fg=FG,
            activebackground=BG2,
            activeforeground=FG,
            selectcolor=BG3,
            highlightthickness=0,
        )
        self.seedvr2_check.grid(row=13, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self._attach_tooltip(self.seedvr2_check, "Upscale extracted face views with SeedVR2 before SHARP inference. This can recover detail but adds substantial processing time.")

        action_bar = ttk.Frame(container)
        action_bar.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        action_bar.columnconfigure(1, weight=1)
        ttk.Button(action_bar, text="Run Pipeline", style="Accent.TButton", command=self._start_pipeline).grid(row=0, column=0, sticky="w")
        ttk.Label(action_bar, textvariable=self.status_var, style="Subtle.TLabel").grid(row=0, column=1, sticky="e", padx=(12, 0))

        ttk.Label(log_panel, text="Log", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(log_panel, style="Easy.Horizontal.TProgressbar", mode="indeterminate")
        self.progress.grid(row=0, column=1, sticky="e")
        self.log_text = scrolledtext.ScrolledText(
            log_panel,
            bg="#11171c",
            fg=FG,
            insertbackground=FG,
            relief="flat",
            borderwidth=0,
            font=("Consolas", 10),
            height=8,
            padx=10,
            pady=10,
        )
        self.log_text.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        self.log_text.tag_config("info", foreground=FG)
        self.log_text.tag_config("warning", foreground=WARN)
        self.log_text.tag_config("error", foreground=ERR)
        self.log_text.tag_config("debug", foreground=FG_DIM)
        self.log_text.tag_config("link", foreground=ACCENT, underline=True)
        self.log_text.config(state="disabled")

    def _build_advanced_panel(self, parent) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Advanced", style="PanelTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._browse_row(parent, 1, "SHARP-Ckeckpoint", self.checkpoint_var, self._browse_checkpoint, tooltip="Optional SHARP model checkpoint. Leave empty to use the default published checkpoint.")
        self._browse_row(parent, 2, "DA360 checkpoint", self.da360_checkpoint_var, self._browse_da360_checkpoint, tooltip="Checkpoint used by DA360 to estimate panorama depth for scale alignment.")
        self._browse_row(parent, 3, "gsbox.exe", self.gsbox_var, self._browse_gsbox, tooltip="Path to gsbox.exe. Needed only for compressed output formats like .spx, .spz, or .sog.")
        self._browse_row(parent, 4, "Intermediate dir", self.intermediate_dir_var, self._browse_intermediate_dir, tooltip="Optional custom location for Temp files, extracted faces, depth maps, and conversion artifacts.")
        self._combo_row(parent, 5, "Device", self.device_var, ["default", "cuda", "cpu", "mps"], tooltip="Inference device for SHARP and DA360. 'default' automatically picks CUDA when available.")

        ttk.Label(parent, text="ImageMagick", style="PanelTitle.TLabel").grid(row=6, column=0, columnspan=3, sticky="w", pady=(18, 10))
        self._check_row(parent, 7, "Optimize panorama with ImageMagick before slicing", self.enable_imagemagick_optimization_var, tooltip="Preprocess the panorama with ImageMagick before view extraction to improve contrast and apparent sharpness.")
        self._browse_row(parent, 8, "magick.exe", self.imagemagick_path_var, self._browse_imagemagick, tooltip="Path to ImageMagick's magick.exe executable used for panorama preprocessing.")
        self._combo_row(parent, 9, "Preset", self.imagemagick_preset_var, ["blur_safe", "classic", "custom"], tooltip="ImageMagick preprocessing preset. 'blur_safe' is tuned for motion-blurred footage and avoids filters that often create mushy fake detail.")
        self._check_row(parent, 10, "Auto level", self.imagemagick_auto_level_var, tooltip="Stretch tonal levels automatically to use more of the available intensity range.")
        self._check_row(parent, 11, "Auto gamma", self.imagemagick_auto_gamma_var, tooltip="Apply automatic gamma correction based on the panorama's brightness distribution.")
        self._check_row(parent, 12, "Normalize", self.imagemagick_normalize_var, tooltip="Normalize contrast to spread highlights and shadows more evenly.")
        self._check_row(parent, 13, "Enhance", self.imagemagick_enhance_var, tooltip="Run ImageMagick's enhancement filter to improve local edge clarity. Usually best left off for motion-blurred footage.")
        self._check_row(parent, 14, "Despeckle", self.imagemagick_despeckle_var, tooltip="Reduce small isolated noise speckles before slicing the panorama into faces. Usually best left off for motion blur because it can smear fine structure.")
        self._check_row(parent, 15, "Apply unsharp mask", self.imagemagick_unsharp_enabled_var, tooltip="Apply an unsharp mask to emphasize edges and recover apparent sharpness.")
        self._entry_row(parent, 16, "Unsharp values", self.imagemagick_unsharp_value_var, "Example: 0x1.6+1.2+0.02", tooltip="Raw ImageMagick unsharp-mask parameters in the form radiusxsigma+amount+threshold.")
        self._entry_row(parent, 17, "Extra args", self.imagemagick_extra_args_var, "Optional extra ImageMagick arguments", tooltip="Additional ImageMagick command-line arguments appended after the structured preprocessing options.")

        ttk.Label(parent, text="Deblur", style="PanelTitle.TLabel").grid(row=18, column=0, columnspan=3, sticky="w", pady=(18, 10))
        self._check_row(parent, 19, "Run motion deblur on extracted faces before SHARP", self.enable_deblur_preprocessing_var, tooltip="Apply a Richardson-Lucy motion deblur stage on the extracted faces before SeedVR2 and SHARP. This targets real capture blur, not just low contrast.")
        self._combo_row(parent, 20, "Deblur strength", self.deblur_strength_var, ["low", "medium", "high"], tooltip="Controls motion-kernel size, deconvolution iterations, and blend strength for the deblur stage.")

        ttk.Label(parent, text="SeedVR2", style="PanelTitle.TLabel").grid(row=21, column=0, columnspan=3, sticky="w", pady=(18, 10))
        self._combo_row(parent, 22, "Model", self.seedvr2_model_name_var, [
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "seedvr2_ema_3b_fp16.safetensors",
            "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b_fp16.safetensors",
            "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b_sharp_fp16.safetensors",
            "seedvr2_ema_3b-Q4_K_M.gguf",
            "seedvr2_ema_3b-Q8_0.gguf",
            "seedvr2_ema_7b-Q4_K_M.gguf",
            "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
        ], tooltip="SeedVR2 model checkpoint or GGUF variant used for face upscaling.")
        self._combo_row(parent, 23, "Output format", self.seedvr2_output_format_var, ["png"], tooltip="File format used for the upscaled intermediate face images written by SeedVR2.")
        self._combo_row(parent, 24, "Color correction", self.seedvr2_color_correction_var, ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], tooltip="Color-matching method used to preserve the original panorama's look after SeedVR2 upscaling.")
        self._combo_row(parent, 25, "Attention mode", self.seedvr2_attention_mode_var, ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"], tooltip="Attention backend used by SeedVR2. Faster modes may depend on your GPU and installed libraries.")
        self._entry_row(parent, 26, "Resolution factor", self.seedvr2_resolution_factor_var, "Multiplier for extracted face size", tooltip="Requested SeedVR2 upscale factor relative to the extracted face resolution.")
        self._check_row(parent, 27, "Stretch proportions before SeedVR2", self.seedvr2_stretch_proportions_var, tooltip="Resize the short side of each SeedVR2 input face without changing the long side. If min resolution is 0, this now auto-squares to the current long side. This is a true non-uniform stretch, not padding or cropping.")
        self._entry_row(parent, 28, "Target short side", self.seedvr2_target_short_side_var, "Ignored when min resolution is 0", tooltip="Manual short-side size used only when stretch proportions is enabled and min resolution is above 0. The long side stays unchanged.")
        self._entry_row(parent, 29, "Min resolution", self.seedvr2_min_resolution_var, "0 disables the floor", tooltip="Minimum longest-side resolution after applying the factor. If the factor result is smaller, SeedVR2 upscales to this minimum instead.")
        self._entry_row(parent, 30, "Max resolution", self.seedvr2_max_resolution_var, "0 disables the cap", tooltip="Upper bound for the longest side of the upscaled face. Use 0 to disable the clamp.")
        self._entry_row(parent, 31, "Batch size", self.seedvr2_batch_size_var, "SeedVR2 image batch size", tooltip="How many faces SeedVR2 processes at once. Higher values use more VRAM.")
        self._entry_row(parent, 32, "Seed", self.seedvr2_seed_var, tooltip="Random seed used by SeedVR2 for reproducible results.")
        self._entry_row(parent, 33, "CUDA device", self.seedvr2_cuda_device_var, "Single GPU id or comma-separated list", tooltip="GPU id list passed to SeedVR2. Usually leave this as 0 on a single-GPU system.")
        self._combo_row(parent, 34, "DiT offload", self.seedvr2_dit_offload_device_var, ["none", "cpu"], tooltip="Offload the diffusion transformer to CPU to reduce VRAM usage at the cost of speed.")
        self._combo_row(parent, 35, "VAE offload", self.seedvr2_vae_offload_device_var, ["none", "cpu"], tooltip="Offload the VAE to CPU to save VRAM. This usually slows processing.")
        self._combo_row(parent, 36, "Tensor offload", self.seedvr2_tensor_offload_device_var, ["none", "cpu"], tooltip="Move additional working tensors off the GPU when VRAM is tight.")
        self._entry_row(parent, 37, "Blocks to swap", self.seedvr2_blocks_to_swap_var, "0-36 depending on model", tooltip="Number of model blocks swapped between devices to reduce VRAM usage. Higher values reduce memory but cost speed.")
        self._entry_row(parent, 38, "Skip first frames", self.seedvr2_skip_first_frames_var, tooltip="Number of initial inputs to skip before SeedVR2 starts processing. Mainly useful for video-oriented workflows.")
        self._entry_row(parent, 39, "VAE encode tile size", self.seedvr2_vae_encode_tile_size_var, tooltip="Tile size used while encoding images in the VAE to fit larger inputs into VRAM.")
        self._entry_row(parent, 40, "VAE encode overlap", self.seedvr2_vae_encode_tile_overlap_var, tooltip="Overlap between VAE encode tiles to reduce seam artifacts.")
        self._entry_row(parent, 41, "VAE decode tile size", self.seedvr2_vae_decode_tile_size_var, tooltip="Tile size used while decoding images in the VAE.")
        self._entry_row(parent, 42, "VAE decode overlap", self.seedvr2_vae_decode_tile_overlap_var, tooltip="Overlap between VAE decode tiles to reduce seam artifacts.")
        self._combo_row(parent, 43, "Compile backend", self.seedvr2_compile_backend_var, ["inductor", "cudagraphs"], tooltip="Torch compile backend for SeedVR2. Backend support and speed depend on your environment.")
        self._combo_row(parent, 44, "Compile mode", self.seedvr2_compile_mode_var, ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], tooltip="Torch compile tuning mode. More aggressive modes may improve speed after longer warmup.")
        self._check_row(parent, 45, "Swap IO components", self.seedvr2_swap_io_components_var, tooltip="Allow SeedVR2 to swap some input/output components to reduce VRAM pressure.")
        self._check_row(parent, 46, "Enable VAE encode tiling", self.seedvr2_vae_encode_tiled_var, tooltip="Process VAE encoding in tiles so large faces fit into VRAM.")
        self._check_row(parent, 47, "Enable VAE decode tiling", self.seedvr2_vae_decode_tiled_var, tooltip="Process VAE decoding in tiles so large faces fit into VRAM.")
        self._check_row(parent, 48, "Cache DiT model", self.seedvr2_cache_dit_var, tooltip="Keep the diffusion transformer cached in memory between SeedVR2 steps when possible.")
        self._check_row(parent, 49, "Cache VAE model", self.seedvr2_cache_vae_var, tooltip="Keep the VAE cached in memory between SeedVR2 steps when possible.")
        self._check_row(parent, 50, "Enable SeedVR2 debug logging", self.seedvr2_debug_enabled_var, tooltip="Enable verbose SeedVR2 logging for troubleshooting and profiling.")

        notes = (
            "Browse a source folder and select the stitched 2:1 equirect panorama from the list.\n"
            "Final output is always written next to the source image.\n"
            "Working files default to Source/Temp/<image-name>.\n"
            "Compressed formats .spx, .spz, and .sog need gsbox.exe.\n"
            "DA360 gives a panorama-wide depth reference used to normalize SHARP's per-view scale.\n"
            "ImageMagick now includes a blur-safe preset that avoids filters likely to worsen motion blur.\n"
            "Optional motion deblur runs on extracted faces before SeedVR2 and SHARP.\n"
            "SeedVR2 options are now stored in the main GUI settings file and override the legacy seedvr2_settings.json file."
        )
        notes_label = tk.Label(parent, text=notes, justify="left", bg=BG2, fg=FG_DIM, wraplength=420)
        notes_label.grid(row=51, column=0, columnspan=3, sticky="ew", pady=(18, 0))

    def _on_scrollable_configure(self, _event, canvas_window: int) -> None:
        if self._scroll_canvas is None:
            return
        self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
        self._scroll_canvas.itemconfigure(canvas_window, width=self._scroll_canvas.winfo_width())

    def _on_mousewheel(self, event) -> None:
        target_canvas = self._resolve_scroll_canvas(event.widget)
        if target_canvas is None:
            return
        target_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _resolve_scroll_canvas(self, widget) -> tk.Canvas | None:
        current = widget
        while current is not None:
            if self._advanced_scroll_canvas is not None and current is self._advanced_scroll_canvas:
                return self._advanced_scroll_canvas
            if self._scroll_canvas is not None and current is self._scroll_canvas:
                return self._scroll_canvas
            current = getattr(current, "master", None)
        return self._advanced_scroll_canvas if self._advanced_window is not None and self._advanced_window.winfo_exists() else self._scroll_canvas

    def _open_advanced_settings(self) -> None:
        if self._advanced_window is not None and self._advanced_window.winfo_exists():
            self._advanced_window.lift()
            self._advanced_window.focus_force()
            return

        window = tk.Toplevel(self)
        window.title("Advanced Settings")
        window.configure(bg=BG)
        window.geometry("860x820")
        window.minsize(760, 620)
        window.transient(self)

        outer = ttk.Frame(window, style="Panel.TFrame", padding=16)
        outer.pack(fill="both", expand=True, padx=16, pady=16)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, bg=BG2, highlightthickness=0, borderwidth=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        self._advanced_scroll_canvas = canvas
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        panel = ttk.Frame(canvas, style="Panel.TFrame", padding=8)
        panel.columnconfigure(1, weight=1)
        canvas_window = canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda event: self._on_advanced_configure(canvas, canvas_window))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(canvas_window, width=event.width))

        self._build_advanced_panel(panel)

        window.protocol("WM_DELETE_WINDOW", self._close_advanced_settings)
        self._advanced_window = window

    def _close_advanced_settings(self) -> None:
        if self._advanced_window is not None and self._advanced_window.winfo_exists():
            self._advanced_window.destroy()
        self._advanced_window = None
        self._advanced_scroll_canvas = None

    def _on_advanced_configure(self, canvas: tk.Canvas, canvas_window: int) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(canvas_window, width=canvas.winfo_width())

    def _attach_tooltip(self, widget, text: str = "") -> None:
        if not text:
            return
        self._tooltips.append(HoverTooltip(widget, text))

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, hint: str = "", tooltip: str = "") -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        self._attach_tooltip(label_widget, tooltip)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Label(parent, text=hint, style="Subtle.TLabel").grid(row=row, column=2, sticky="w", padx=(10, 0))

    def _combo_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, values: list[str], tooltip: str = "") -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        self._attach_tooltip(label_widget, tooltip)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Label(parent, text="", style="Subtle.TLabel").grid(row=row, column=2, sticky="w")

    def _check_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.BooleanVar, tooltip: str = "") -> None:
        checkbox = tk.Checkbutton(
            parent,
            text=label,
            variable=variable,
            bg=BG2,
            fg=FG,
            activebackground=BG2,
            activeforeground=FG,
            selectcolor=BG3,
            highlightthickness=0,
        )
        checkbox.grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._attach_tooltip(checkbox, tooltip)

    def _browse_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, callback, tooltip: str = "") -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        self._attach_tooltip(label_widget, tooltip)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Button(parent, text="Browse", command=callback).grid(row=row, column=2, sticky="e", padx=(10, 0), pady=6)

    def _readonly_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, tooltip: str = "") -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        self._attach_tooltip(label_widget, tooltip)
        entry = ttk.Entry(parent, textvariable=variable, state="readonly")
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Label(parent, text="Auto-saved beside source", style="Subtle.TLabel").grid(row=row, column=2, sticky="w", padx=(10, 0))

    def _bind_persistence(self) -> None:
        variables = [
            self.browse_folder_var,
            self.input_var,
            self.output_var,
            self.side_count_var,
            self.format_var,
            self.device_var,
            self.checkpoint_var,
            self.da360_checkpoint_var,
            self.alignment_mode_var,
            self.gsbox_var,
            self.intermediate_dir_var,
            self.imagemagick_preset_var,
            self.imagemagick_path_var,
            self.imagemagick_unsharp_value_var,
            self.imagemagick_extra_args_var,
            self.deblur_strength_var,
            self.seedvr2_model_name_var,
            self.seedvr2_output_format_var,
            self.seedvr2_color_correction_var,
            self.seedvr2_attention_mode_var,
            self.seedvr2_cuda_device_var,
            self.seedvr2_dit_offload_device_var,
            self.seedvr2_vae_offload_device_var,
            self.seedvr2_tensor_offload_device_var,
            self.seedvr2_resolution_factor_var,
            self.seedvr2_target_short_side_var,
            self.seedvr2_min_resolution_var,
            self.seedvr2_max_resolution_var,
            self.seedvr2_batch_size_var,
            self.seedvr2_seed_var,
            self.seedvr2_skip_first_frames_var,
            self.seedvr2_blocks_to_swap_var,
            self.seedvr2_vae_encode_tile_size_var,
            self.seedvr2_vae_encode_tile_overlap_var,
            self.seedvr2_vae_decode_tile_size_var,
            self.seedvr2_vae_decode_tile_overlap_var,
            self.seedvr2_compile_backend_var,
            self.seedvr2_compile_mode_var,
        ]
        for variable in variables:
            variable.trace_add("write", self._on_settings_changed)
        self.enable_da360_alignment_var.trace_add("write", self._on_settings_changed)
        self.enable_da360_alignment_var.trace_add("write", self._on_da360_alignment_changed)
        self.alignment_mode_var.trace_add("write", self._on_settings_changed)
        self.alignment_mode_var.trace_add("write", self._on_da360_alignment_changed)
        self.alignment_grid_resolution_var.trace_add("write", self._on_settings_changed)
        self.alignment_detail_weight_var.trace_add("write", self._on_settings_changed)
        self.cutoff_height_percent_var.trace_add("write", self._on_settings_changed)
        self.enable_alignment_sweep_var.trace_add("write", self._on_settings_changed)
        self.keep_intermediates_var.trace_add("write", self._on_settings_changed)
        self.keep_intermediates_var.trace_add("write", self._on_keep_intermediates_changed)
        self.delete_temp_files_var.trace_add("write", self._on_settings_changed)
        self.enable_seedvr2_upscale_var.trace_add("write", self._on_settings_changed)
        self.enable_imagemagick_optimization_var.trace_add("write", self._on_settings_changed)
        self.enable_deblur_preprocessing_var.trace_add("write", self._on_settings_changed)
        self.imagemagick_preset_var.trace_add("write", self._on_imagemagick_preset_changed)
        self.imagemagick_auto_level_var.trace_add("write", self._on_settings_changed)
        self.imagemagick_auto_gamma_var.trace_add("write", self._on_settings_changed)
        self.imagemagick_normalize_var.trace_add("write", self._on_settings_changed)
        self.imagemagick_enhance_var.trace_add("write", self._on_settings_changed)
        self.imagemagick_despeckle_var.trace_add("write", self._on_settings_changed)
        self.imagemagick_unsharp_enabled_var.trace_add("write", self._on_settings_changed)
        self.seedvr2_stretch_proportions_var.trace_add("write", self._on_settings_changed)
        self.seedvr2_swap_io_components_var.trace_add("write", self._on_settings_changed)
        self.seedvr2_vae_encode_tiled_var.trace_add("write", self._on_settings_changed)
        self.seedvr2_vae_decode_tiled_var.trace_add("write", self._on_settings_changed)
        self.seedvr2_cache_dit_var.trace_add("write", self._on_settings_changed)
        self.seedvr2_cache_vae_var.trace_add("write", self._on_settings_changed)
        self.seedvr2_debug_enabled_var.trace_add("write", self._on_settings_changed)
        self.format_var.trace_add("write", self._on_format_changed)
        self.input_var.trace_add("write", self._on_input_changed)

    def _on_settings_changed(self, *_args) -> None:
        imagemagick_settings = self._build_imagemagick_option_values()
        save_settings(
            {
                "last_browse_folder": self.browse_folder_var.get().strip(),
                "input_path": self.input_var.get().strip(),
                "output_path": self.output_var.get().strip(),
                "side_count": self.side_count_var.get().strip(),
                "format": self.format_var.get().strip(),
                "device": self.device_var.get().strip(),
                "checkpoint": self.checkpoint_var.get().strip(),
                "da360_checkpoint": self.da360_checkpoint_var.get().strip(),
                "gsbox": self.gsbox_var.get().strip(),
                "intermediate_dir": self.intermediate_dir_var.get().strip(),
                "enable_da360_alignment": bool(self.enable_da360_alignment_var.get()),
                "alignment_mode": insp_to_splat.normalize_alignment_mode(self.alignment_mode_var.get()),
                "alignment_grid_resolution": int(self.alignment_grid_resolution_var.get()),
                "alignment_detail_weight": float(self.alignment_detail_weight_var.get()),
                "cutoff_height_percent": float(self.cutoff_height_percent_var.get()),
                "enable_alignment_sweep": bool(self.enable_alignment_sweep_var.get()),
                "keep_intermediates": bool(self.keep_intermediates_var.get()),
                "delete_temp_files": bool(self.delete_temp_files_var.get()),
                "enable_seedvr2_upscale": bool(self.enable_seedvr2_upscale_var.get()),
                "enable_imagemagick_optimization": bool(self.enable_imagemagick_optimization_var.get()),
                "enable_deblur_preprocessing": bool(self.enable_deblur_preprocessing_var.get()),
                "deblur_strength": self.deblur_strength_var.get().strip(),
                "imagemagick_path": self.imagemagick_path_var.get().strip(),
                "imagemagick_commands": insp_to_splat.build_imagemagick_command_string(option_values=imagemagick_settings),
                "seedvr2_model_name": self.seedvr2_model_name_var.get().strip(),
                "seedvr2_output_format": self.seedvr2_output_format_var.get().strip(),
                "seedvr2_color_correction": self.seedvr2_color_correction_var.get().strip(),
                "seedvr2_attention_mode": self.seedvr2_attention_mode_var.get().strip(),
                "seedvr2_cuda_device": self.seedvr2_cuda_device_var.get().strip(),
                "seedvr2_dit_offload_device": self.seedvr2_dit_offload_device_var.get().strip(),
                "seedvr2_vae_offload_device": self.seedvr2_vae_offload_device_var.get().strip(),
                "seedvr2_tensor_offload_device": self.seedvr2_tensor_offload_device_var.get().strip(),
                "seedvr2_resolution_factor": self.seedvr2_resolution_factor_var.get().strip(),
                "seedvr2_stretch_proportions": bool(self.seedvr2_stretch_proportions_var.get()),
                "seedvr2_target_short_side": self.seedvr2_target_short_side_var.get().strip(),
                "seedvr2_min_resolution": self.seedvr2_min_resolution_var.get().strip(),
                "seedvr2_max_resolution": self.seedvr2_max_resolution_var.get().strip(),
                "seedvr2_batch_size": self.seedvr2_batch_size_var.get().strip(),
                "seedvr2_seed": self.seedvr2_seed_var.get().strip(),
                "seedvr2_skip_first_frames": self.seedvr2_skip_first_frames_var.get().strip(),
                "seedvr2_blocks_to_swap": self.seedvr2_blocks_to_swap_var.get().strip(),
                "seedvr2_vae_encode_tile_size": self.seedvr2_vae_encode_tile_size_var.get().strip(),
                "seedvr2_vae_encode_tile_overlap": self.seedvr2_vae_encode_tile_overlap_var.get().strip(),
                "seedvr2_vae_decode_tile_size": self.seedvr2_vae_decode_tile_size_var.get().strip(),
                "seedvr2_vae_decode_tile_overlap": self.seedvr2_vae_decode_tile_overlap_var.get().strip(),
                "seedvr2_compile_backend": self.seedvr2_compile_backend_var.get().strip(),
                "seedvr2_compile_mode": self.seedvr2_compile_mode_var.get().strip(),
                "seedvr2_swap_io_components": bool(self.seedvr2_swap_io_components_var.get()),
                "seedvr2_vae_encode_tiled": bool(self.seedvr2_vae_encode_tiled_var.get()),
                "seedvr2_vae_decode_tiled": bool(self.seedvr2_vae_decode_tiled_var.get()),
                "seedvr2_cache_dit": bool(self.seedvr2_cache_dit_var.get()),
                "seedvr2_cache_vae": bool(self.seedvr2_cache_vae_var.get()),
                "seedvr2_debug_enabled": bool(self.seedvr2_debug_enabled_var.get()),
                **imagemagick_settings,
            }
        )

    def _on_input_changed(self, *_args) -> None:
        if self._suspend_input_sync:
            return
        input_text = self.input_var.get().strip()
        if input_text:
            input_path = Path(input_text)
            if input_path.parent.exists() and self.browse_folder_var.get().strip() != str(input_path.parent):
                self.browse_folder_var.set(str(input_path.parent))
        self._suggest_output_path()
        self._sync_file_selection()
        self._update_preview()

    def _on_keep_intermediates_changed(self, *_args) -> None:
        self._sync_temp_cleanup_state()
        self._on_settings_changed()

    def _sync_temp_cleanup_state(self) -> None:
        if getattr(self, "keep_intermediates_var", None) is None or getattr(self, "delete_temp_check", None) is None:
            return
        if self.keep_intermediates_var.get():
            if self.delete_temp_files_var.get():
                self.delete_temp_files_var.set(False)
            self.delete_temp_check.configure(state="disabled")
        else:
            self.delete_temp_check.configure(state="normal")

    def _on_da360_alignment_changed(self, *_args) -> None:
        self._sync_alignment_tuner_state()

    def _sync_alignment_tuner_state(self) -> None:
        enabled = getattr(self, "enable_da360_alignment_var", None) and self.enable_da360_alignment_var.get()
        alignment_mode = insp_to_splat.normalize_alignment_mode(self.alignment_mode_var.get()) if hasattr(self, "alignment_mode_var") else "da360"
        general_state = "normal" if enabled else "disabled"
        da360_state = "normal" if enabled and alignment_mode == "da360" else "disabled"
        for widget in (
            getattr(self, "cutoff_height_scale", None),
            getattr(self, "_cutoff_height_label", None),
            getattr(self, "_alignment_mode_label", None),
        ):
            if widget is not None:
                widget.configure(state=general_state)
        if getattr(self, "alignment_mode_combo", None) is not None:
            self.alignment_mode_combo.configure(state="readonly" if enabled else "disabled")
        for widget in (
            getattr(self, "grid_res_scale", None),
            getattr(self, "detail_wt_scale", None),
            getattr(self, "_grid_res_label", None),
            getattr(self, "_detail_wt_label", None),
            getattr(self, "alignment_sweep_check", None),
        ):
            if widget is not None:
                widget.configure(state=da360_state)

    def _on_grid_res_changed(self, value) -> None:
        v = int(float(value))
        self._grid_res_label.configure(text=f"Grid Resolution: {v}")
        self.alignment_grid_resolution_var.set(v)

    def _on_detail_wt_changed(self, value) -> None:
        v = int(float(value))
        self._detail_wt_label.configure(text=f"Detail Preservation: {v} %")
        self.alignment_detail_weight_var.set(v / 100.0)

    def _on_cutoff_height_changed(self, value) -> None:
        v = int(float(value))
        self._cutoff_height_label.configure(text=f"Cut-off Height: {v} %")
        self.cutoff_height_percent_var.set(float(v))

    def _on_format_changed(self, *_args) -> None:
        self._sync_output_extension()
        self._update_preview()

    def _suggest_output_path(self) -> None:
        input_text = self.input_var.get().strip()
        if not input_text:
            self.output_var.set("")
            return
        input_path = Path(input_text)
        if not input_path.suffix:
            self.output_var.set("")
            return
        suffix = self.format_var.get().strip() or "ply"
        self.output_var.set(str(input_path.with_name(f"{input_path.stem}_merged.{suffix}")))

    def _sync_output_extension(self) -> None:
        self._suggest_output_path()

    def _refresh_file_list(self, select_path: str | None = None) -> None:
        folder = Path(self.browse_folder_var.get().strip() or ROOT_DIR)
        self.file_listbox.delete(0, tk.END)
        self._file_paths = []
        if not folder.exists() or not folder.is_dir():
            return
        self._file_paths = sorted(
            [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".webp"}],
            key=lambda item: item.name.lower(),
        )
        for path in self._file_paths:
            self.file_listbox.insert(tk.END, path.name)
        if select_path:
            self._sync_file_selection(Path(select_path))

    def _get_selected_paths(self) -> list[Path]:
        indices = list(self.file_listbox.curselection())
        return [self._file_paths[index] for index in indices if 0 <= index < len(self._file_paths)]

    def _select_file_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        wanted = {path.resolve() for path in paths if path.exists()}
        self.file_listbox.selection_clear(0, tk.END)
        selected_paths: list[Path] = []
        for index, path in enumerate(self._file_paths):
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if resolved in wanted:
                self.file_listbox.selection_set(index)
                selected_paths.append(path)
        if selected_paths:
            self.file_listbox.see(self._file_paths.index(selected_paths[0]))
            self._set_current_input(selected_paths[0], sync_preview=True)

    def _sync_file_selection(self, selected_path: Path | None = None) -> None:
        current = selected_path
        if current is None:
            input_text = self.input_var.get().strip()
            current = Path(input_text) if input_text else None
        self.file_listbox.selection_clear(0, tk.END)
        if current is None:
            return
        for index, path in enumerate(self._file_paths):
            if path == current:
                self.file_listbox.selection_set(index)
                self.file_listbox.see(index)
                break

    def _default_temp_dir(self, input_path: Path) -> Path:
        return input_path.parent / "Temp" / input_path.stem

    def _set_current_input(self, path: Path, sync_preview: bool = False) -> None:
        self._suspend_input_sync = True
        try:
            self.input_var.set(str(path))
            suffix = self.format_var.get().strip() or "ply"
            self.output_var.set(str(path.with_name(f"{path.stem}_merged.{suffix}")))
        finally:
            self._suspend_input_sync = False
        if sync_preview:
            self._update_preview()

    def _update_preview(self) -> None:
        selected_paths = self._get_selected_paths()
        input_text = self.input_var.get().strip()
        if not input_text:
            self._preview_photo = None
            self.preview_label.configure(image="", text="No preview", width=PREVIEW_SIZE[0], height=12)
            self.selection_var.set("Select a stitched panorama from the file browser.")
            self.output_hint_var.set("Final output will be written next to the source image.")
            self.temp_hint_var.set("Working files will be written under Source/Temp.")
            return

        input_path = selected_paths[0] if selected_paths else Path(input_text)
        if not input_path.exists():
            self._preview_photo = None
            self.preview_label.configure(image="", text="Missing file", width=PREVIEW_SIZE[0], height=12)
            self.selection_var.set(f"Selected file does not exist:\n{input_path}")
            return

        dimensions = (0, 0)
        try:
            with Image.open(input_path) as image:
                image = ImageOps.exif_transpose(image)
                dimensions = image.size
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGB")
                preview = image.copy()
                preview.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
                canvas = Image.new("RGB", PREVIEW_SIZE, LIST_BG)
                offset_x = (PREVIEW_SIZE[0] - preview.width) // 2
                offset_y = (PREVIEW_SIZE[1] - preview.height) // 2
                canvas.paste(preview, (offset_x, offset_y))
                self._preview_photo = ImageTk.PhotoImage(canvas)
                self.preview_label.configure(image=self._preview_photo, text="", width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        except Exception as exc:
            self._preview_photo = None
            self.preview_label.configure(image="", text=f"Preview unavailable\n{exc}", width=PREVIEW_SIZE[0], height=12)

        temp_dir = Path(self.intermediate_dir_var.get().strip()) if self.intermediate_dir_var.get().strip() else self._default_temp_dir(input_path)
        if len(selected_paths) > 1:
            self.selection_var.set(
                f"{len(selected_paths)} files selected\n"
                f"First: {input_path.name}\n"
                f"Detected files in folder: {len(self._file_paths)}"
            )
            self.output_hint_var.set(f"Batch output pattern: <source>_merged.{self.format_var.get().strip() or 'ply'} beside each source image")
            if self.intermediate_dir_var.get().strip():
                self.temp_hint_var.set(f"Temp workspaces: {self.intermediate_dir_var.get().strip()}\\<image-name>")
            elif self.delete_temp_files_var.get():
                self.temp_hint_var.set("Temp workspaces will be deleted automatically after each file.")
            else:
                self.temp_hint_var.set("Temp workspaces will be kept under Source/Temp for each selected image.")
        else:
            self.selection_var.set(
                f"{input_path.name}\n"
                f"Resolution: {dimensions[0]} x {dimensions[1]}\n"
                f"Detected files in folder: {len(self._file_paths)}"
            )
            self.output_hint_var.set(f"Final output: {self.output_var.get().strip()}")
            if self.delete_temp_files_var.get() and not self.keep_intermediates_var.get():
                self.temp_hint_var.set(f"Temp workspace: {temp_dir} (will be deleted after processing)")
            else:
                self.temp_hint_var.set(f"Temp workspace: {temp_dir}")

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select 2:1 equirect panorama",
            filetypes=[("Panorama images", "*.jpg *.jpeg *.png *.heic"), ("All files", "*.*")],
        )
        if path:
            self.browse_folder_var.set(str(Path(path).parent))
            self._refresh_file_list(select_path=path)
            self.input_var.set(path)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Select source folder", initialdir=self.browse_folder_var.get().strip() or str(ROOT_DIR))
        if path:
            self.browse_folder_var.set(path)
            self._refresh_file_list(select_path=self.input_var.get().strip())

    def _browse_output(self) -> None:
        selected_format = self.format_var.get().strip() or "ply"
        path = filedialog.asksaveasfilename(
            title="Select output splat",
            defaultextension=f".{selected_format}",
            filetypes=[("Splat files", "*.ply *.spx *.spz *.sog"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)
            self._sync_output_extension()

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(title="Select SHARP-Ckeckpoint", filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.checkpoint_var.set(path)

    def _browse_imagemagick(self) -> None:
        path = filedialog.askopenfilename(title="Select magick.exe", filetypes=[("Executable", "magick.exe"), ("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            self.imagemagick_path_var.set(path)

    def _auto_detect_imagemagick_path(self) -> None:
        current_value = self.imagemagick_path_var.get().strip()
        detected = insp_to_splat.find_imagemagick_executable(current_value)
        if detected is None:
            return
        detected_text = str(detected)
        if current_value != detected_text:
            self.imagemagick_path_var.set(detected_text)

    def _build_imagemagick_option_values(self) -> dict[str, object]:
        return {
            "imagemagick_preset": self.imagemagick_preset_var.get().strip(),
            "imagemagick_auto_level": bool(self.imagemagick_auto_level_var.get()),
            "imagemagick_auto_gamma": bool(self.imagemagick_auto_gamma_var.get()),
            "imagemagick_normalize": bool(self.imagemagick_normalize_var.get()),
            "imagemagick_enhance": bool(self.imagemagick_enhance_var.get()),
            "imagemagick_despeckle": bool(self.imagemagick_despeckle_var.get()),
            "imagemagick_unsharp_enabled": bool(self.imagemagick_unsharp_enabled_var.get()),
            "imagemagick_unsharp_value": self.imagemagick_unsharp_value_var.get().strip(),
            "imagemagick_extra_args": self.imagemagick_extra_args_var.get().strip(),
        }

    def _apply_imagemagick_preset(self, preset_name: str) -> None:
        normalized = insp_to_splat.normalize_imagemagick_preset_name(preset_name)
        if normalized == "custom":
            self._on_settings_changed()
            return
        preset_settings = insp_to_splat.get_imagemagick_preset_settings(normalized)
        self._applying_imagemagick_preset = True
        try:
            self.imagemagick_preset_var.set(normalized)
            self.imagemagick_auto_level_var.set(bool(preset_settings["imagemagick_auto_level"]))
            self.imagemagick_auto_gamma_var.set(bool(preset_settings["imagemagick_auto_gamma"]))
            self.imagemagick_normalize_var.set(bool(preset_settings["imagemagick_normalize"]))
            self.imagemagick_enhance_var.set(bool(preset_settings["imagemagick_enhance"]))
            self.imagemagick_despeckle_var.set(bool(preset_settings["imagemagick_despeckle"]))
            self.imagemagick_unsharp_enabled_var.set(bool(preset_settings["imagemagick_unsharp_enabled"]))
            self.imagemagick_unsharp_value_var.set(str(preset_settings["imagemagick_unsharp_value"]))
            self.imagemagick_extra_args_var.set(str(preset_settings["imagemagick_extra_args"]))
        finally:
            self._applying_imagemagick_preset = False
        self._on_settings_changed()

    def _on_imagemagick_preset_changed(self, *_args) -> None:
        if self._applying_imagemagick_preset:
            return
        self._apply_imagemagick_preset(self.imagemagick_preset_var.get())

    def _browse_da360_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select DA360 checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")],
        )
        if path:
            self.da360_checkpoint_var.set(path)

    def _browse_gsbox(self) -> None:
        path = filedialog.askopenfilename(title="Select gsbox.exe", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            self.gsbox_var.set(path)

    def _browse_intermediate_dir(self) -> None:
        path = filedialog.askdirectory(title="Select intermediate output directory")
        if path:
            self.intermediate_dir_var.set(path)
            self._update_preview()

    def _on_browser_select(self, _event=None) -> None:
        selected_paths = self._get_selected_paths()
        if not selected_paths:
            return
        selected = selected_paths[0]
        if self.input_var.get().strip() != str(selected):
            self._set_current_input(selected)
        self._update_preview()

    def _select_all_files(self, event=None):
        widget = self.focus_get()
        if widget is not self.file_listbox:
            return None
        self.file_listbox.selection_set(0, tk.END)
        self._on_browser_select()
        return "break"

    def _append_log(self, level: str, message: str) -> None:
        self.log_text.config(state="normal")
        start_index = self.log_text.index("end-1c")
        self.log_text.insert("end", message + "\n", level if level in {"debug", "info", "warning", "error"} else "info")
        self._tag_output_link(start_index, message)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _tag_output_link(self, start_index: str, message: str) -> None:
        output_path = self._extract_output_path_from_log_message(message)
        if output_path is None:
            return

        path_text = str(output_path)
        path_offset = message.find(path_text)
        if path_offset < 0:
            return

        line_start = self.log_text.index(start_index)
        path_start = f"{line_start}+{path_offset}c"
        path_end = f"{path_start}+{len(path_text)}c"
        tag_name = f"output_link_{self._log_link_index}"
        self._log_link_index += 1
        self.log_text.tag_add("link", path_start, path_end)
        self.log_text.tag_add(tag_name, path_start, path_end)
        self.log_text.tag_bind(tag_name, "<Button-1>", lambda _event, path=output_path: self._open_output_location(path))
        self.log_text.tag_bind(tag_name, "<Enter>", lambda _event: self.log_text.config(cursor="hand2"))
        self.log_text.tag_bind(tag_name, "<Leave>", lambda _event: self.log_text.config(cursor="xterm"))

    def _extract_output_path_from_log_message(self, message: str) -> Path | None:
        prefixes = (
            "Created ",
            "Finished successfully: ",
            "Wrote merged output to ",
        )
        for prefix in prefixes:
            if prefix == "Created ":
                marker = " outputs in: "
                if message.startswith(prefix) and marker in message:
                    candidate = message.split(marker, 1)[1].strip()
                    if candidate:
                        return Path(candidate)
                continue
            if message.startswith(prefix):
                candidate = message[len(prefix):].strip()
                if candidate:
                    return Path(candidate)
        return None

    def _open_output_location(self, output_path: Path) -> None:
        target = output_path if output_path.exists() else output_path.parent
        try:
            if target.exists() and target.is_file():
                subprocess.Popen(["explorer", "/select,", str(target)])
            elif target.exists() and target.is_dir():
                os.startfile(str(target))
            elif output_path.parent.exists():
                os.startfile(str(output_path.parent))
            else:
                raise FileNotFoundError(output_path.parent)
        except Exception as exc:
            messagebox.showerror("Open output folder failed", f"Could not open output location:\n{exc}")

    def _show_completion_popup(self, output_path: Path) -> None:
        window = tk.Toplevel(self)
        window.title("Pipeline complete")
        window.configure(bg=BG)
        window.transient(self)
        window.resizable(False, False)

        panel = ttk.Frame(window, style="Panel.TFrame", padding=16)
        panel.pack(fill="both", expand=True, padx=16, pady=16)

        ttk.Label(panel, text="Pipeline complete", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(panel, text="Output written to:", style="Subtle.TLabel").pack(anchor="w", pady=(10, 4))

        link = tk.Label(
            panel,
            text=str(output_path),
            bg=BG2,
            fg=ACCENT,
            cursor="hand2",
            justify="left",
            wraplength=640,
            font=("Segoe UI", 10, "underline"),
        )
        link.pack(anchor="w")
        link.bind("<Button-1>", lambda _event: self._open_output_location(output_path))

        actions = ttk.Frame(panel)
        actions.pack(fill="x", pady=(16, 0))
        ttk.Button(actions, text="Open Output Folder", command=lambda: self._open_output_location(output_path)).pack(side="left")
        ttk.Button(actions, text="Close", command=window.destroy).pack(side="right")

    def _drain_logs(self) -> None:
        while True:
            try:
                level, message = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(level, message)
        self.after(100, self._drain_logs)

    def _validate_inputs(self, input_paths: list[Path]) -> bool:
        if self._running:
            return False
        if not input_paths:
            messagebox.showerror("Missing input", "Select a 2:1 equirect panorama file.")
            return False
        missing_paths = [path for path in input_paths if not path.exists()]
        if missing_paths:
            messagebox.showerror("Missing input", f"The selected input file does not exist:\n{missing_paths[0]}")
            return False
        try:
            if self.side_count_var.get().strip():
                side_count = int(self.side_count_var.get().strip())
                if side_count < 2:
                    raise ValueError
        except ValueError:
            messagebox.showerror("Invalid values", "Sides must be an integer and at least 2.")
            return False
        alignment_mode = insp_to_splat.normalize_alignment_mode(self.alignment_mode_var.get())
        if self.enable_da360_alignment_var.get():
            if alignment_mode == "da360":
                da360_checkpoint = self.da360_checkpoint_var.get().strip()
                if not da360_checkpoint:
                    messagebox.showerror("DA360 checkpoint required", "Choose a DA360 checkpoint or switch the alignment mode to overlap.")
                    return False
                if not Path(da360_checkpoint).exists():
                    messagebox.showerror("Missing DA360 checkpoint", "The selected DA360 checkpoint does not exist.")
                    return False
        elif self.enable_alignment_sweep_var.get():
            messagebox.showerror("DA360 alignment required", "Alignment sweep mode requires DA360 alignment to stay enabled.")
            return False
        if self.enable_alignment_sweep_var.get() and alignment_mode != "da360":
            messagebox.showerror("DA360 mode required", "Alignment sweep mode currently works only with DA360 alignment mode.")
            return False
        if self.enable_imagemagick_optimization_var.get():
            magick_path = insp_to_splat.find_imagemagick_executable(self.imagemagick_path_var.get().strip())
            if magick_path is None:
                messagebox.showerror("ImageMagick required", "Panorama optimization is enabled, but magick.exe was not found. Pick it in the GUI or add ImageMagick to PATH.")
                return False
            imagemagick_commands = insp_to_splat.build_imagemagick_command_tokens(option_values=self._build_imagemagick_option_values())
            if not imagemagick_commands:
                messagebox.showerror("ImageMagick commands required", "Enable at least one ImageMagick operation or disable panorama optimization.")
                return False
            if self.imagemagick_unsharp_enabled_var.get() and not self.imagemagick_unsharp_value_var.get().strip():
                messagebox.showerror("ImageMagick unsharp required", "Provide unsharp values or disable the unsharp option.")
                return False
        if self.enable_seedvr2_upscale_var.get() and self.seedvr2_stretch_proportions_var.get():
            min_resolution_text = self.seedvr2_min_resolution_var.get().strip()
            try:
                min_resolution = int(min_resolution_text) if min_resolution_text else 0
            except ValueError:
                messagebox.showerror("Invalid SeedVR2 minimum resolution", "Min resolution must be an integer.")
                return False
            if min_resolution > 0:
                try:
                    if int(self.seedvr2_target_short_side_var.get().strip()) <= 0:
                        raise ValueError
                except ValueError:
                    messagebox.showerror("Invalid SeedVR2 stretch", "Target short side must be an integer greater than 0 when stretch proportions is enabled and min resolution is above 0.")
                    return False
        if self.format_var.get().strip() in {"spx", "spz", "sog"} and not self.gsbox_var.get().strip() and not (ROOT_DIR / "gsbox.exe").exists():
            messagebox.showerror("gsbox required", "Compressed output needs gsbox.exe. Pick it in the GUI or place it next to this script.")
            return False
        return True

    def _build_args_for_input(self, input_path: Path, batch_mode: bool = False) -> SimpleNamespace:
        side_count_text = self.side_count_var.get().strip()
        checkpoint_text = self.checkpoint_var.get().strip()
        da360_checkpoint_text = self.da360_checkpoint_var.get().strip()
        gsbox_text = self.gsbox_var.get().strip()
        intermediate_dir_text = self.intermediate_dir_var.get().strip()
        imagemagick_settings = self._build_imagemagick_option_values()
        suffix = self.format_var.get().strip() or "ply"
        output_path = input_path.with_name(f"{input_path.stem}_merged.{suffix}") if batch_mode else Path(self.output_var.get().strip())
        intermediate_dir = None
        if intermediate_dir_text:
            intermediate_root = Path(intermediate_dir_text)
            intermediate_dir = (intermediate_root / input_path.stem) if batch_mode else intermediate_root
        return SimpleNamespace(
            input=input_path,
            output=output_path,
            side_count=int(side_count_text) if side_count_text else 4,
            face_size=0,
            format=self.format_var.get().strip(),
            quality=DEFAULT_GUI_QUALITY,
            sh_degree=DEFAULT_GUI_SH_DEGREE,
            device=self.device_var.get().strip() or "default",
            checkpoint=Path(checkpoint_text) if checkpoint_text else None,
            da360_checkpoint=Path(da360_checkpoint_text) if da360_checkpoint_text else None,
            enable_da360_alignment=bool(self.enable_da360_alignment_var.get()),
            alignment_mode=insp_to_splat.normalize_alignment_mode(self.alignment_mode_var.get()),
            alignment_grid_resolution=int(self.alignment_grid_resolution_var.get()),
            alignment_detail_weight=float(self.alignment_detail_weight_var.get()),
            cutoff_height_percent=float(self.cutoff_height_percent_var.get()),
            enable_alignment_sweep=bool(self.enable_alignment_sweep_var.get()),
            keep_intermediates=bool(self.keep_intermediates_var.get()),
            delete_temp_files=bool(self.delete_temp_files_var.get()),
            enable_seedvr2_upscale=bool(self.enable_seedvr2_upscale_var.get()),
            enable_imagemagick_optimization=bool(self.enable_imagemagick_optimization_var.get()),
            enable_deblur_preprocessing=bool(self.enable_deblur_preprocessing_var.get()),
            deblur_strength=self.deblur_strength_var.get().strip() or "medium",
            imagemagick=Path(self.imagemagick_path_var.get().strip()) if self.imagemagick_path_var.get().strip() else None,
            imagemagick_commands=insp_to_splat.build_imagemagick_command_string(option_values=imagemagick_settings),
            intermediate_dir=intermediate_dir,
            config=insp_to_splat.DEFAULT_CONFIG_PATH,
            gsbox=Path(gsbox_text) if gsbox_text else None,
            verbose=True,
            **imagemagick_settings,
        )

    def _start_pipeline(self) -> None:
        selected_paths = self._get_selected_paths()
        if not selected_paths and self.input_var.get().strip():
            selected_paths = [Path(self.input_var.get().strip())]
        if not self._validate_inputs(selected_paths):
            return
        self._running = True
        batch_count = len(selected_paths)
        self.status_var.set(f"Running pipeline for {batch_count} image{'s' if batch_count != 1 else ''}...")
        self.progress.start(10)
        if batch_count == 1:
            self._append_log("info", f"Starting 360 SHARP pipeline for {selected_paths[0].name}.")
        else:
            self._append_log("info", f"Starting batch pipeline for {batch_count} panoramas.")
        args_list = [self._build_args_for_input(path, batch_mode=batch_count > 1) for path in selected_paths]
        self._worker = threading.Thread(target=self._run_pipeline_worker, args=(args_list,), daemon=True)
        self._worker.start()

    def _run_pipeline_worker(self, args_list: list[SimpleNamespace]) -> None:
        logger = logging.getLogger()
        handler = QueueLogHandler(self._log_queue)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            results = []
            total = len(args_list)
            for index, args in enumerate(args_list, start=1):
                self._log_queue.put(("info", f"[{index}/{total}] Processing {args.input.name}"))
                result = insp_to_splat.run_pipeline(args)
                results.append(result)
        except Exception as exc:
            self._log_queue.put(("error", f"Pipeline failed: {exc}"))
            self.after(0, lambda: self._finish_pipeline(False, None))
        else:
            for result in results:
                display_path = getattr(result, "display_path", None) or result.output_path
                if getattr(result, "generated_outputs", None) and len(result.generated_outputs) > 1:
                    self._log_queue.put(("info", f"Created {len(result.generated_outputs)} outputs in: {display_path}"))
                self._log_queue.put(("info", f"Finished successfully: {display_path}"))
            self.after(0, lambda r=results: self._finish_pipeline(True, r))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    def _finish_pipeline(self, success: bool, results: list | None = None) -> None:
        self._running = False
        self.progress.stop()
        if success and results:
            last_result = results[-1]
            display_path = getattr(last_result, "display_path", None) or last_result.output_path
            generated_count = len(getattr(last_result, "generated_outputs", []) or [])
            if len(results) == 1:
                if generated_count > 1:
                    self.status_var.set(f"Done: {display_path.name} ({generated_count} outputs)")
                else:
                    self.status_var.set(f"Done: {display_path.name}")
            else:
                self.status_var.set(f"Done: {len(results)} files processed")
            self.output_var.set(str(last_result.output_path))
            self._update_preview()
            if len(results) == 1:
                self._show_completion_popup(display_path)
            else:
                messagebox.showinfo("Batch complete", f"Processed {len(results)} panoramas successfully.\nSee the log for clickable output paths.")
        else:
            self.status_var.set("Pipeline failed")
            messagebox.showerror("Pipeline failed", "The pipeline did not finish. Check the log for details.")


def main() -> int:
    initial_input_paths = [Path(arg).resolve() for arg in sys.argv[1:] if Path(arg).exists()]
    app = App(initial_input_paths=initial_input_paths)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())