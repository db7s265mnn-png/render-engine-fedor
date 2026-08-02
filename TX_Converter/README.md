# TX Converter

Standalone Qt tool that shares Bob Render’s `maketx` / OCIO TX core (`src/io/tx_convert.*`).

## Build

Built with the main app (`TX_Converter` / `TX_Converter.exe`).

## Usage

1. **Source** — texture path, or a UDIM pattern with `<UDIM>` (Houdini-style).
2. **Output Folder** — directory for `.tx` files (names from source basenames).
3. **Color Space** — input space; output is always **ACEScg**.
4. **Advanced** — full colour-space list from the OCIO config.
5. **Use OCIO from Environment** — read `OCIO` (default on).

Requires `maketx` (or `oiiotool`) on `PATH` and a system OCIO config for colour converts.
