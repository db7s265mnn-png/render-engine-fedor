# TX Converter

Standalone Qt tool that shares Bob Render’s `maketx` / OCIO TX core (`src/io/tx_convert.*`),
plus a texture viewer (LDR / HDR / `.tx`, UDIM timeline).

## Build

Built with the main app (`TX_Converter` / `TX_Converter.exe`).

## Usage

1. **Source** — texture path, or a UDIM / `$F` pattern.
2. **Output Folder** — must be chosen (empty by default; Convert blocks if unset).
3. **Format** — **Original** / **EXR** / **TX** (default) / **PNG** / **JPG**.
   Original follows the source extension and its bit/channel rules.
4. **Bit Depth** — adapts (JPG hidden; PNG 8/16; EXR 16/32; TX 8/16/32).
5. **Resolution** — Original or long-side presets 256…8192 (aspect preserved).
6. **Channels** — format-aware (JPG: RGB/R/G/B; others: RGBA…A).
7. **Color Space / OCIO** — shown for **TX** (and Original→TX) only.
8. **Convert** — TX via maketx (+ oiiotool preprocess when needed); EXR/PNG/JPG/Original via oiiotool.
9. **Viewer** — View first, channel swatches, Classic/OCIO + monitor view.
    Timeline uses filename numbers. **Brightness** = exposure stops (click label to reset).
    Drag to pan (even without zoom), **F** or Fit to frame.

Requires `maketx` and/or `oiiotool`. The Windows zip ships them next to `TX_Converter.exe`.
OCIO config comes from the `OCIO` env var or the OCIO Config field (not bundled).
