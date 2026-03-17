# SHARP_360_to_Splat

SHARP_360_to_Splat is a Windows GUI for converting stitched 2:1 equirectangular 360 panoramas into splat outputs with Apple's SHARP model and `gsbox`.

Export your source footage from Insta360 Studio first. This repo no longer ingests raw `.insp` files directly.

The pipeline can optionally normalize SHARP's per-view depth scale with Insta360's DA360 depth model. The default checkpoint path is `checkpoints/DA360_large.pth`.

## Features

- Batch image conversion with parallel SHARP inference
- Direct conversion for `.ply`, `.spx`, `.spz`, and `.sog`
- Output export as `.ply`, `.spx`, `.spz`, or `.sog`
- Optional transform, scale, and crop post-processing
- Built-in file browser with thumbnails, metadata, and multi-select
- Configurable splat viewer launch on double click or Enter

## New PC Setup

1. Install Anaconda or Miniconda.
2. Install Git for Windows.
3. Run `Setup_NewPC.bat`.
4. Start `SHARP_360_to_Splat.exe` or `Launch_SHARP_360_to_Splat.bat`.

`Setup_NewPC.bat` now also downloads the default DA360 checkpoint used for depth alignment.

## Build

Run `build_exe.bat` from a machine where the `sharp` conda environment already exists. The script builds the executable and assembles a versioned release package under `release_pkg/`.

## Release Package Contents

- `SHARP_360_to_Splat.exe`
- `Launch_SHARP_360_to_Splat.bat`
- `Setup_NewPC.bat`
- `gsbox.exe`
- `third_party/DA360/`
- `checkpoints/DA360_large.pth` when available at build time
- `splatapult/build/Release/`
