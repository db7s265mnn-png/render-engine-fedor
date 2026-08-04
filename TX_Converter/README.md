# TX Converter

Standalone Qt tool that shares Bob Render’s `maketx` / OCIO TX core (`src/io/tx_convert.*`),
plus a texture viewer (LDR / HDR / `.tx`, UDIM timeline).

## Build

Built with the main app (`TX_Converter` / `TX_Converter.exe`).

## Usage

1. **Source** — texture path, or a UDIM pattern with `<UDIM>` (Houdini-style).
2. **Output Folder** — directory for `.tx` files (names from source basenames).
3. **Color Space** — input space; output is always **ACEScg**.
4. **Advanced** — full colour-space list from the OCIO config.
5. **Use OCIO from Environment** — read `OCIO` (default on).
6. **Preview Source** — view the source (or UDIM sequence). Timeline length = tile count.
7. **Preview Output Folder .tx** — view converted `.tx` next to the output folder.

Requires `maketx` (or `oiiotool`). The Windows zip ships `maketx.exe` next to
`TX_Converter.exe`. On other platforms, put them on `PATH`. OCIO config comes from
the `OCIO` env var or the OCIO Config field (not bundled).
