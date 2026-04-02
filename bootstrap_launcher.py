from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tkinter import Tk, messagebox


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
GUI_SCRIPT = APP_DIR / "Easy_360_SHARP_GUI.py"
SETUP_SCRIPT = APP_DIR / "Setup_NewPC.bat"
PYTHON_EXE = APP_DIR / ".venv" / "Scripts" / "python.exe"


def ask_yes_no(title: str, message: str) -> bool:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return bool(messagebox.askyesno(title, message, parent=root))
    finally:
        root.destroy()


def show_error(title: str, message: str) -> None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        messagebox.showerror(title, message, parent=root)
    finally:
        root.destroy()


def run_setup() -> bool:
    if not SETUP_SCRIPT.exists():
        show_error("Setup missing", f"Could not find {SETUP_SCRIPT.name} next to this launcher.")
        return False
    result = subprocess.run(["cmd", "/c", str(SETUP_SCRIPT)], cwd=APP_DIR)
    return result.returncode == 0


def launch_gui() -> int:
    if not GUI_SCRIPT.exists():
        show_error("GUI script missing", f"Could not find {GUI_SCRIPT.name} next to this launcher.")
        return 1

    if not PYTHON_EXE.exists():
        should_run_setup = ask_yes_no(
            "First-time setup required",
            "The local runtime environment is not installed yet.\n\nRun Setup_NewPC.bat now?",
        )
        if not should_run_setup:
            return 1
        if not run_setup() or not PYTHON_EXE.exists():
            show_error("Setup failed", "The runtime environment could not be prepared. Run Setup_NewPC.bat manually and try again.")
            return 1

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run([str(PYTHON_EXE), str(GUI_SCRIPT), *sys.argv[1:]], cwd=APP_DIR, env=env)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(launch_gui())