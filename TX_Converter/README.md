# TX Converter

Standalone Qt tool that shares Bob Render’s `maketx` / OCIO TX core (`src/io/tx_convert.*`),
plus a texture viewer (LDR / HDR / `.tx`, UDIM timeline).

## Build

Built with the main app (`TX_Converter` / `TX_Converter.exe`).

## Usage

1. **Source** — texture path, or a UDIM / `$F` pattern.
2. **Output Folder** — destination directory (names from source basenames).
3. **Format** — **TX** / **PNG** / **JPG**.
4. **Bit Depth** — adapts to format (JPG hidden/8; PNG 8/16; TX 8/16/32).
5. **Resolution** — Original or long-side presets 256…8192 (aspect preserved).
6. **Channels** — RGBA / RGB / R / G / B / A (single-channel for R/G/B/A).
7. **Color Space / OCIO** — shown for **TX** only (output ACEScg). PNG/JPG keep source colour.
8. **Convert** — TX via maketx (oiiotool preprocess when resize/bit/channels); PNG/JPG via oiiotool.
9. **Viewer** — View (Source / Converted) first, then channel toggles, then Classic/OCIO + monitor view.
    Timeline uses UDIM/`$F` numbers from filenames. Bright = exposure stops (click label to reset).

Requires `maketx` and/or `oiiotool`. The Windows zip ships them next to `TX_Converter.exe`.
OCIO config comes from the `OCIO` env var or the OCIO Config field (not bundled).
