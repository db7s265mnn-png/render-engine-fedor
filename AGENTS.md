# AGENTS.md

## Cursor Cloud specific instructions

Solstice (binary/user name **Bob_Render**) is a single native C++17 desktop
path tracer built with CMake. There are no web servers, databases, or
background services — you build one binary and either open its Qt GUI or run it
headless. Standard build/test/run commands live in `README.md` ("Building",
"Tests", "Command line"); the notes below only capture the non-obvious bits for
this Linux cloud environment.

### Compiler: use GCC, not the default `c++`

The default `cc`/`c++` alternatives point to `clang`, and clang here fails to
link (`cannot find -lstdc++`). Always configure with GCC explicitly:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      -DSOLSTICE_ENABLE_TINYUSDZ=OFF
cmake --build build -j"$(nproc)"
```

### Disable TinyUSDZ on this toolchain

`-DSOLSTICE_ENABLE_TINYUSDZ=OFF` is required: the fetched TinyUSDZ `v0.9.4`
fails to link under GCC with duplicate-symbol errors
(`multiple definition of tinyusdz::OrientationFromString...`). It is optional
(binary `.usdc` import only); `.usda` text USD, Alembic, procedural geometry,
HDRI and everything else still work. All tests pass with it off (the
`binary-usd-usdc` test self-skips).

### Optional deps that are OFF here

- **Alembic** is not available via apt, so `.abc` import is auto-disabled (CMake
  prints a warning and continues). The `sol_make_test_abc` tool is not built as
  a result. To enable it, build Alembic 1.8.6 from source per `README.md`.
- **OptiX/CUDA GPU backend** stays OFF (no GPU); the renderer uses the Embree
  CPU backend. This is expected.
- OpenEXR, libtiff, MaterialX, oneTBB, OpenPGL and OpenSubdiv are all ON.

### First build fetches and compiles large deps

Configure/build downloads and compiles MaterialX, OpenPGL, OpenSubdiv (and
libtiff/oneTBB if not from apt) via CMake `FetchContent`. The first build takes
several minutes and needs network access; later builds are incremental. The
`build/` directory is git-ignored.

### Running the GUI (needs a display)

The Qt desktop app requires an X display. This environment is headless, so run
it under Xvfb:

```bash
Xvfb :99 -screen 0 1600x1000x24 -ac &
export DISPLAY=:99 XDG_RUNTIME_DIR=/tmp/runtime-ubuntu
mkdir -p "$XDG_RUNTIME_DIR"
./build/bin/Bob_Render examples/studio_sphere.bobsc
```

Set `XDG_RUNTIME_DIR` (any writable dir) to silence the QStandardPaths warning.
Grab a frame/video of `:99` with `ffmpeg -f x11grab -video_size 1600x1000 -i :99 ...`.

### Headless rendering (no display needed)

```bash
./build/bin/Bob_Render --headless examples/studio_sphere.bobsc \
    -o /tmp/out.png -s 64 --width 480 --height 320
```

`studio_sphere.bobsc` / `glass_sphere.bobsc` use procedural geometry and work
without Alembic; `alembic_hdri.bobsc` needs the Alembic-enabled build.
