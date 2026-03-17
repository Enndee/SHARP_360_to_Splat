from __future__ import annotations

import json
import logging
import os
import queue
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


def load_settings() -> dict:
    defaults = {
        "input_path": "",
        "output_path": "",
        "last_browse_folder": str(ROOT_DIR),
        "side_count": "4",
        "format": "ply",
        "quality": "9",
        "sh_degree": "0",
        "device": "default",
        "checkpoint": "",
        "da360_checkpoint": str(insp_to_splat.DEFAULT_DA360_CHECKPOINT_PATH),
        "gsbox": "",
        "intermediate_dir": "",
        "enable_da360_alignment": True,
        "keep_intermediates": False,
    }
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                defaults.update(loaded)
        except Exception:
            pass
    return defaults


def save_settings(data: dict) -> None:
    try:
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


class App(tk.Tk):
    def __init__(self):
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

        self.browse_folder_var = tk.StringVar(value=settings["last_browse_folder"])
        self.input_var = tk.StringVar(value=settings["input_path"])
        self.output_var = tk.StringVar(value=settings["output_path"])
        self.side_count_var = tk.StringVar(value=settings["side_count"])
        self.format_var = tk.StringVar(value=settings["format"])
        self.quality_var = tk.StringVar(value=settings["quality"])
        self.sh_degree_var = tk.StringVar(value=settings["sh_degree"])
        self.device_var = tk.StringVar(value=settings["device"])
        self.checkpoint_var = tk.StringVar(value=settings["checkpoint"])
        self.da360_checkpoint_var = tk.StringVar(value=settings["da360_checkpoint"])
        self.gsbox_var = tk.StringVar(value=settings["gsbox"])
        self.intermediate_dir_var = tk.StringVar(value=settings["intermediate_dir"])
        self.enable_da360_alignment_var = tk.BooleanVar(value=bool(settings["enable_da360_alignment"]))
        self.keep_intermediates_var = tk.BooleanVar(value=bool(settings["keep_intermediates"]))
        self.status_var = tk.StringVar(value="Ready")
        self.selection_var = tk.StringVar(value="Select a stitched panorama from the file browser.")
        self.output_hint_var = tk.StringVar(value="Final output will be written next to the source image.")
        self.temp_hint_var = tk.StringVar(value="Working files will be written under Source/Temp.")

        self._build_styles()
        self._build_ui()
        self._bind_persistence()
        self._sync_output_extension()
        self._refresh_file_list(select_path=self.input_var.get().strip())
        self._update_preview()
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
        style.configure("TCombobox", fieldbackground=INPUT_BG, foreground=FG, bordercolor=INPUT_BORDER)
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
        container.columnconfigure(1, weight=2)
        container.rowconfigure(2, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="SHARP_360_to_Splat", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Browse stitched panoramas, preview the source image, and export the merged splat beside it while Temp files stay near the source.",
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        form_panel = ttk.Frame(container, style="Panel.TFrame", padding=16)
        form_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 12), pady=(14, 12))
        form_panel.columnconfigure(1, weight=1)

        advanced_panel = ttk.Frame(container, style="Panel.TFrame", padding=16)
        advanced_panel.grid(row=1, column=1, sticky="nsew", pady=(14, 12))
        advanced_panel.columnconfigure(1, weight=1)

        log_panel = ttk.Frame(container, style="Panel.TFrame", padding=16)
        log_panel.grid(row=2, column=0, columnspan=2, sticky="nsew")
        log_panel.columnconfigure(0, weight=1)
        log_panel.rowconfigure(1, weight=1)

        ttk.Label(form_panel, text="Source Browser", style="PanelTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._browse_row(form_panel, 1, "Source folder", self.browse_folder_var, self._browse_folder)
        self._browse_row(form_panel, 2, "Input panorama", self.input_var, self._browse_input)
        self._readonly_row(form_panel, 3, "Output splat", self.output_var)

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
        self._entry_row(form_panel, 6, "Sides", self.side_count_var, "2+ horizon views")
        self._combo_row(form_panel, 7, "Format", self.format_var, ["ply", "spx", "spz", "sog"])
        self._entry_row(form_panel, 8, "Quality", self.quality_var, "1-9")
        self._entry_row(form_panel, 9, "SH degree", self.sh_degree_var, "0-3")
        self._combo_row(form_panel, 10, "Device", self.device_var, ["default", "cuda", "cpu", "mps"])

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
        self.keep_check.grid(row=11, column=0, columnspan=3, sticky="w", pady=(12, 0))

        self.da360_check = tk.Checkbutton(
            form_panel,
            text="Align SHARP depth scale to DA360 panorama depth",
            variable=self.enable_da360_alignment_var,
            bg=BG2,
            fg=FG,
            activebackground=BG2,
            activeforeground=FG,
            selectcolor=BG3,
            highlightthickness=0,
        )
        self.da360_check.grid(row=12, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Label(advanced_panel, text="Advanced", style="PanelTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._browse_row(advanced_panel, 1, "Checkpoint", self.checkpoint_var, self._browse_checkpoint)
        self._browse_row(advanced_panel, 2, "DA360 checkpoint", self.da360_checkpoint_var, self._browse_da360_checkpoint)
        self._browse_row(advanced_panel, 3, "gsbox.exe", self.gsbox_var, self._browse_gsbox)
        self._browse_row(advanced_panel, 4, "Intermediate dir", self.intermediate_dir_var, self._browse_intermediate_dir)

        notes = (
            "Browse a source folder and select the stitched 2:1 equirect panorama from the list.\n"
            "Final output is always written next to the source image.\n"
            "Working files default to Source/Temp/<image-name>.\n"
            "Compressed formats .spx, .spz, and .sog need gsbox.exe.\n"
            "DA360 gives a panorama-wide depth reference used to normalize SHARP's per-view scale."
        )
        notes_label = tk.Label(
            advanced_panel,
            text=notes,
            justify="left",
            bg=BG2,
            fg=FG_DIM,
            wraplength=300,
        )
        notes_label.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(18, 0))

        action_bar = ttk.Frame(container)
        action_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
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
            padx=10,
            pady=10,
        )
        self.log_text.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        self.log_text.tag_config("info", foreground=FG)
        self.log_text.tag_config("warning", foreground=WARN)
        self.log_text.tag_config("error", foreground=ERR)
        self.log_text.tag_config("debug", foreground=FG_DIM)
        self.log_text.config(state="disabled")

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, hint: str = "") -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Label(parent, text=hint, style="Subtle.TLabel").grid(row=row, column=2, sticky="w", padx=(10, 0))

    def _combo_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, values: list[str]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Label(parent, text="", style="Subtle.TLabel").grid(row=row, column=2, sticky="w")

    def _browse_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, callback) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Button(parent, text="Browse", command=callback).grid(row=row, column=2, sticky="e", padx=(10, 0), pady=6)

    def _readonly_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
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
            self.quality_var,
            self.sh_degree_var,
            self.device_var,
            self.checkpoint_var,
            self.da360_checkpoint_var,
            self.gsbox_var,
            self.intermediate_dir_var,
        ]
        for variable in variables:
            variable.trace_add("write", self._on_settings_changed)
        self.enable_da360_alignment_var.trace_add("write", self._on_settings_changed)
        self.keep_intermediates_var.trace_add("write", self._on_settings_changed)
        self.format_var.trace_add("write", self._on_format_changed)
        self.input_var.trace_add("write", self._on_input_changed)

    def _on_settings_changed(self, *_args) -> None:
        save_settings(
            {
                "last_browse_folder": self.browse_folder_var.get().strip(),
                "input_path": self.input_var.get().strip(),
                "output_path": self.output_var.get().strip(),
                "side_count": self.side_count_var.get().strip(),
                "format": self.format_var.get().strip(),
                "quality": self.quality_var.get().strip(),
                "sh_degree": self.sh_degree_var.get().strip(),
                "device": self.device_var.get().strip(),
                "checkpoint": self.checkpoint_var.get().strip(),
                "da360_checkpoint": self.da360_checkpoint_var.get().strip(),
                "gsbox": self.gsbox_var.get().strip(),
                "intermediate_dir": self.intermediate_dir_var.get().strip(),
                "enable_da360_alignment": bool(self.enable_da360_alignment_var.get()),
                "keep_intermediates": bool(self.keep_intermediates_var.get()),
            }
        )

    def _on_input_changed(self, *_args) -> None:
        input_text = self.input_var.get().strip()
        if input_text:
            input_path = Path(input_text)
            if input_path.parent.exists() and self.browse_folder_var.get().strip() != str(input_path.parent):
                self.browse_folder_var.set(str(input_path.parent))
        self._suggest_output_path()
        self._sync_file_selection()
        self._update_preview()

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

    def _update_preview(self) -> None:
        input_text = self.input_var.get().strip()
        if not input_text:
            self._preview_photo = None
            self.preview_label.configure(image="", text="No preview", width=PREVIEW_SIZE[0], height=12)
            self.selection_var.set("Select a stitched panorama from the file browser.")
            self.output_hint_var.set("Final output will be written next to the source image.")
            self.temp_hint_var.set("Working files will be written under Source/Temp.")
            return

        input_path = Path(input_text)
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
        self.selection_var.set(
            f"{input_path.name}\n"
            f"Resolution: {dimensions[0]} x {dimensions[1]}\n"
            f"Detected files in folder: {len(self._file_paths)}"
        )
        self.output_hint_var.set(f"Final output: {self.output_var.get().strip()}")
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
        path = filedialog.askopenfilename(title="Select SHARP checkpoint", filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.checkpoint_var.set(path)

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
        selection = self.file_listbox.curselection()
        if not selection:
            return
        selected = self._file_paths[selection[0]]
        if self.input_var.get().strip() != str(selected):
            self.input_var.set(str(selected))

    def _append_log(self, level: str, message: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n", level if level in {"debug", "info", "warning", "error"} else "info")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _drain_logs(self) -> None:
        while True:
            try:
                level, message = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(level, message)
        self.after(100, self._drain_logs)

    def _validate_inputs(self) -> bool:
        if self._running:
            return False
        if not self.input_var.get().strip():
            messagebox.showerror("Missing input", "Select a 2:1 equirect panorama file.")
            return False
        if not Path(self.input_var.get().strip()).exists():
            messagebox.showerror("Missing input", "The selected input file does not exist.")
            return False
        try:
            if self.side_count_var.get().strip():
                side_count = int(self.side_count_var.get().strip())
                if side_count < 2:
                    raise ValueError
            quality = int(self.quality_var.get().strip())
            sh_degree = int(self.sh_degree_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid values", "Sides, quality, and SH degree must be integers, and sides must be at least 2.")
            return False
        if not (1 <= quality <= 9):
            messagebox.showerror("Invalid quality", "Quality must be between 1 and 9.")
            return False
        if not (0 <= sh_degree <= 3):
            messagebox.showerror("Invalid SH degree", "SH degree must be between 0 and 3.")
            return False
        if self.enable_da360_alignment_var.get():
            da360_checkpoint = self.da360_checkpoint_var.get().strip()
            if not da360_checkpoint:
                messagebox.showerror("DA360 checkpoint required", "Choose a DA360 checkpoint or disable DA360 alignment.")
                return False
            if not Path(da360_checkpoint).exists():
                messagebox.showerror("Missing DA360 checkpoint", "The selected DA360 checkpoint does not exist.")
                return False
        if self.format_var.get().strip() in {"spx", "spz", "sog"} and not self.gsbox_var.get().strip() and not (ROOT_DIR / "gsbox.exe").exists():
            messagebox.showerror("gsbox required", "Compressed output needs gsbox.exe. Pick it in the GUI or place it next to this script.")
            return False
        return True

    def _build_args(self) -> SimpleNamespace:
        side_count_text = self.side_count_var.get().strip()
        checkpoint_text = self.checkpoint_var.get().strip()
        da360_checkpoint_text = self.da360_checkpoint_var.get().strip()
        gsbox_text = self.gsbox_var.get().strip()
        intermediate_dir_text = self.intermediate_dir_var.get().strip()
        return SimpleNamespace(
            input=Path(self.input_var.get().strip()),
            output=Path(self.output_var.get().strip()),
            side_count=int(side_count_text) if side_count_text else 4,
            face_size=0,
            format=self.format_var.get().strip(),
            quality=int(self.quality_var.get().strip()),
            sh_degree=int(self.sh_degree_var.get().strip()),
            device=self.device_var.get().strip() or "default",
            checkpoint=Path(checkpoint_text) if checkpoint_text else None,
            da360_checkpoint=Path(da360_checkpoint_text) if da360_checkpoint_text else None,
            enable_da360_alignment=bool(self.enable_da360_alignment_var.get()),
            keep_intermediates=bool(self.keep_intermediates_var.get()),
            intermediate_dir=Path(intermediate_dir_text) if intermediate_dir_text else None,
            config=insp_to_splat.DEFAULT_CONFIG_PATH,
            gsbox=Path(gsbox_text) if gsbox_text else None,
            verbose=True,
        )

    def _start_pipeline(self) -> None:
        if not self._validate_inputs():
            return
        self._running = True
        self.status_var.set("Running pipeline...")
        self.progress.start(10)
        self._append_log("info", f"Starting 360 SHARP pipeline for {Path(self.input_var.get().strip()).name}.")
        args = self._build_args()
        self._worker = threading.Thread(target=self._run_pipeline_worker, args=(args,), daemon=True)
        self._worker.start()

    def _run_pipeline_worker(self, args: SimpleNamespace) -> None:
        logger = logging.getLogger()
        handler = QueueLogHandler(self._log_queue)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            result = insp_to_splat.run_pipeline(args)
        except Exception as exc:
            self._log_queue.put(("error", f"Pipeline failed: {exc}"))
            self.after(0, lambda: self._finish_pipeline(False, None))
        else:
            self._log_queue.put(("info", f"Finished successfully: {result.output_path}"))
            self.after(0, lambda r=result: self._finish_pipeline(True, r))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    def _finish_pipeline(self, success: bool, result=None) -> None:
        self._running = False
        self.progress.stop()
        if success and result is not None:
            self.status_var.set(f"Done: {result.output_path.name}")
            self.output_var.set(str(result.output_path))
            self._update_preview()
            messagebox.showinfo("Pipeline complete", f"Wrote output to:\n{result.output_path}")
            if result.depth_map_path and result.depth_map_path.exists():
                self._show_depth_map(result.depth_map_path)
        else:
            self.status_var.set("Pipeline failed")
            messagebox.showerror("Pipeline failed", "The pipeline did not finish. Check the log for details.")

    def _show_depth_map(self, depth_map_path: Path) -> None:
        try:
            img = Image.open(depth_map_path)
        except Exception:
            return
        max_w, max_h = 1000, 500
        w, h = img.size
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        win = tk.Toplevel(self)
        win.title("DA360 Depth Map")
        win.configure(bg=BG)
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(win, image=photo, bg=BG)
        label.image = photo  # prevent garbage collection
        label.pack(padx=10, pady=10)
        tk.Label(win, text=str(depth_map_path), bg=BG, fg=FG_DIM, font=("Consolas", 9)).pack(pady=(0, 10))


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())