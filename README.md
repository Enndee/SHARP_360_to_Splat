# SHARP_360_to_Splat

SHARP_360_to_Splat is a Windows desktop workflow for turning stitched 2:1 equirectangular 360 panoramas into Gaussian splat outputs using Apple's SHARP model.

It wraps the full process in a GUI with source browsing, preview, batch selection, optional preprocessing, optional SeedVR2 upscaling, optional DA360 depth alignment, and export to standard or compressed splat formats.

This project expects already-stitched panoramas exported from tools such as Insta360 Studio. It does not ingest raw `.insp` files directly.

## What It Does

- Loads stitched 2:1 panorama images from a folder browser
- Mirrors the input panorama horizontally before SHARP processing to compensate for SHARP's orientation behavior in this workflow
- Optionally preprocesses the panorama with ImageMagick before slicing
- Optionally sharpens and upscales extracted faces with SeedVR2 before SHARP prediction
- Optionally aligns SHARP depth scale against DA360 panorama depth
- Merges all predicted view splats into one final output
- Exports `.ply`, `.spx`, `.spz`, or `.sog`

## Main Features

- GUI file browser with preview and source-folder workflow
- Multi-select batch processing from the file list
- `Ctrl+A` support in the file list to select all panoramas in the current folder
- Windows Send To integration: send one or more images to the launcher and open the GUI with them preselected
- Clickable output paths in the log and completion dialog
- Optional automatic Temp workspace cleanup after processing
- Optional repo-managed ImageMagick install under `third_party/ImageMagick`
- Optional SeedVR2 face upscaling with settings exposed in the GUI
- Optional DA360-based depth normalization for cross-view scale consistency

## Supported Inputs

- `.jpg`
- `.jpeg`
- `.png`
- `.heic`
- `.webp` in the GUI browser when Pillow can read it

Input panoramas must be stitched equirectangular images with a 2:1 aspect ratio.

## Supported Outputs

- `.ply`
- `.spx`
- `.spz`
- `.sog`

Compressed exports require `gsbox.exe`.

## Quick Start

1. Install Python 3.13.
2. Install Git for Windows.
3. Run `Setup_NewPC.bat`.
4. Start `SHARP_360_to_Splat.exe` or `!Launch_SHARP_360_to_Splat.bat`.
5. Choose a source folder with stitched panoramas.
6. Select one image or multi-select several images in the file list.
7. Click `Run Pipeline`.

## Setup Details

`Setup_NewPC.bat` prepares a local portable-style environment for this repo.

It will:

1. Detect Python 3.13
2. Create or reuse `.venv`
3. Install PyTorch CUDA 12.8 wheels
4. Install the vendored `ml-sharp` package in editable mode
5. Clone or update `seedvr2_videoupscaler` from its upstream repository and install its Python requirements
6. Download and install ImageMagick into `third_party/ImageMagick`
7. Download the default DA360 checkpoint into `checkpoints/DA360_large.pth`
8. Create a Windows Send To shortcut for SHARP_360_to_Splat

Supported setup flags:

- `--dry-run`
- `--skip-checkpoint`

## GUI Workflow

The main window is built around a source-folder browser.

- Pick a source folder
- Select one or more panoramas from the file list
- Review the preview and output hint
- Configure format, device, and optional processing features
- Run the pipeline for the current selection

Batch behavior:

- When multiple files are selected, each image is processed independently
- Each result is written beside its source image using the pattern `<source_stem>_merged.<format>`
- The log shows progress per file and exposes clickable output paths

Temp behavior:

- By default the Temp workspace is deleted automatically after processing
- If `Keep intermediate face images and per-face splats` is enabled, Temp cleanup is disabled automatically
- If an intermediate directory is configured, batch runs create a per-image subfolder inside that location

## Send To Integration

`Setup_NewPC.bat` creates a Windows Send To shortcut.

After setup, you can:

1. Right-click one or more supported panorama images in Explorer
2. Choose `Send to -> SHARP_360_to_Splat`
3. The launcher opens the GUI with those images already selected in the file list

The batch launcher `!Launch_SHARP_360_to_Splat.bat` forwards incoming file arguments directly to the GUI.

## ImageMagick Integration

ImageMagick is handled as a repo-managed third-party tool instead of being committed into Git.

- Setup installs it under `third_party/ImageMagick`
- Runtime prefers that local copy before PATH lookup
- The GUI auto-detects `magick.exe`
- Panorama preprocessing is controlled by explicit GUI toggles instead of one raw command string

Current ImageMagick operations exposed in the GUI:

- Auto level
- Auto gamma
- Normalize
- Enhance
- Despeckle
- Unsharp mask
- Extra args

## SeedVR2 Integration

SeedVR2 is not vendored into the main repository. Setup clones it from its upstream repository and installs its runtime dependencies locally.

In the GUI you can enable face upscaling before SHARP prediction and configure key SeedVR2 parameters such as:

- model
- resolution factor
- batch size
- offload settings
- compile backend and mode
- VAE tiling

## DA360 Alignment

DA360 is used as an optional panorama-wide depth reference.

When enabled, the pipeline predicts DA360 panorama depth, projects that reference into each extracted SHARP view, and aligns SHARP's per-view scale accordingly. This helps produce more consistent merged geometry across views.

The default checkpoint path is `checkpoints/DA360_large.pth`.

## Build

Run `build_exe.bat` from a machine where the local `.venv` already exists.

The build script:

1. Installs PyInstaller into the local environment
2. Builds `SHARP_360_to_Splat.exe`
3. Assembles a release folder under `release_pkg/`
4. Bundles DA360 assets and optional extras
5. Creates a zip archive for distribution

If `third_party/ImageMagick` exists locally at build time, it is bundled into the release package as well.

## Release Package Contents

- `SHARP_360_to_Splat.exe`
- `!Launch_SHARP_360_to_Splat.bat`
- `Setup_NewPC.bat`
- `gsbox.exe`
- `third_party/DA360/`
- `third_party/ImageMagick/` when a local repo-managed ImageMagick install is present at build time
- `checkpoints/DA360_large.pth` when available at build time
- `splatapult/build/Release/`

## Repository Layout

- `Easy_360_SHARP_GUI.py`: main Windows GUI
- `insp_to_splat.py`: main panorama-to-splat pipeline
- `Setup_NewPC.bat`: one-shot setup/bootstrap script
- `build_exe.bat`: Windows packaging script
- `!Launch_SHARP_360_to_Splat.bat`: script launcher with argument forwarding
- `ml-sharp/`: vendored SHARP source tree
- `third_party/DA360/`: DA360 integration assets

## Notes And Requirements

- Windows-focused workflow
- NVIDIA GPU strongly recommended for practical performance
- First SHARP run may download model weights from Apple's servers
- `gsbox.exe` is required for `.spx`, `.spz`, and `.sog`
- Input panoramas must be stitched before they enter this workflow

## Troubleshooting

### ImageMagick setup fails

- Run `Setup_NewPC.bat` again
- Check that GitHub API access and GitHub release downloads are reachable
- If needed, browse manually to `magick.exe` in the GUI

### DA360 checkpoint missing

- Re-run `Setup_NewPC.bat`
- Or browse manually to a DA360 checkpoint in the advanced settings
- Or disable DA360 alignment

### Compressed output fails

- Confirm `gsbox.exe` exists next to the app or is selected in the advanced settings

### GUI opens from Send To but nothing is selected

- Re-run `Setup_NewPC.bat` so the Send To shortcut is recreated
- Make sure you are using the current launcher or rebuilt executable

## License And Third-Party Components

This repository integrates several third-party components, including SHARP-related code, DA360 assets, SeedVR2 setup, and ImageMagick downloads from upstream sources. Review the license files in this repository and in each upstream dependency before redistribution.
