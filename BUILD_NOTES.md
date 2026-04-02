# Build Notes

## v1.5.2 Release Notes

### Packaging changes

1. The release package is now intended to ship the updated overlap alignment pipeline, the corrected tall-face extraction behavior, and the new SeedVR2 equalize and pre-downscale controls together.
2. The packaged README text should describe `Equalize proportions via SeedVR2`, `Pre-upscale downscale factor`, and the current min/max resolution rules instead of the older resolution-factor workflow.

### Behavior changes worth verifying before release

1. Face extraction keeps panorama-derived vertical coverage, and `Cut-off Height` trims the top and bottom symmetrically.
2. SeedVR2 pre-downscale is now input-only; the final upscaled output size is still based on the original extracted face size.
3. In non-equalized SeedVR2 mode, output size is driven only by min/max longest-side limits.
4. In equalized SeedVR2 mode, only max resolution constrains the final output size.

## v1.5 Release Postmortem

### Problems encountered

1. The Splatapult build initially failed because the Visual Studio C++ toolchain was not being launched from a reliable shell entry point.
2. The bundled `vcpkg` checkout inside `splatapult/vcpkg` was too old and attempted to download MSYS artifacts from URLs that now returned `404`, which broke dependency setup during the `eigen3` port build.
3. Early release packages were created before `splatapult/build/Release` existed, so the viewer payload was missing from the package.
4. The README generation block in `build_exe.bat` used `->` inside a redirected batch block, and the `>` characters were interpreted as redirection, causing repeated "file not found" messages during packaging.

### How it was solved

1. Use the Visual Studio Build Tools developer shell directly:

   ```bat
   %comspec% /k "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
   ```

2. Update the local `splatapult/vcpkg` checkout to a newer snapshot, run `bootstrap-vcpkg.bat` again, and configure CMake with the absolute `vcpkg` toolchain path.
3. Build Splatapult first so `splatapult/build/Release` exists, then run `build_exe.bat` to assemble the final package.
4. Escape `>` characters in redirected batch `echo` blocks, for example `-^>` instead of `->`.

### Faster next time

1. Start the release from the Visual Studio developer shell above.
2. Confirm the `vcpkg` snapshot is current enough before attempting the Splatapult build.
3. Build Splatapult before packaging the Python app.
4. Keep batch-file README text free of unescaped redirection characters.