# TX Tools (Alpha / Omega)

Standalone Qt tool that shares Grendizer_Render’s `maketx` / OCIO TX core (`src/io/tx_convert.*`),
plus a texture viewer (LDR / HDR / `.tx`, UDIM timeline).

## Editions

| Edition | Exe | Viewport |
|---------|-----|----------|
| **Alpha** | `Grendizer_TX_Tools_Alpha` | Qt `FloatPreviewCanvas` (CPU bake + QPainter) |
| **Omega** | `Grendizer_TX_Tools_Omega` | Same CPU bake (OCIO/grade → 8-bit RGB in RAM), then stream to **DX12** (Windows only) |

Convert UI, Source/Output, UDIM timeline, grade, and channels are shared — feature parity between editions.
Omega v1 does not re-implement color on the GPU; display is baked on CPU then uploaded.

CMake options:

- `-DSOLSTICE_BUILD_TX_TOOLS_ALPHA=ON|OFF`
- `-DSOLSTICE_BUILD_TX_TOOLS_OMEGA=ON|OFF` (ignored on non-Windows)

## Usage

1. **Source** — texture path, or a UDIM / `$F` pattern.
2. **Output Folder** — must be chosen (empty by default; Convert blocks if unset).
3. **Format** — **Original** / **EXR** / **TIFF** / **TX** (default) / **PNG** / **JPG**.
   Original follows the source extension and its bit/channel rules.
4. **Bit Depth** — **Original** (from source) or explicit `8 (uint)` / `16 (uint)` / `16 (half)` / `32 (float)`.
   JPG hides the control; PNG = Original / 8u / 16u only.
5. **Resolution** — Original or long-side presets 256…8192 (aspect preserved).
6. **Channels** — **Original** (default: 1→R, 3→RGB, 4→RGBA; mono stays 1ch) or explicit RGBA/RGB/R/G/B/A.
   JPG: Original / RGB / R / G / B (no alpha).
7. **Color Space / OCIO** — shown for **TX** (and Original→TX) only.
8. **Convert** — TX via maketx (+ oiiotool preprocess when needed); EXR/PNG/JPG/Original via oiiotool.
9. **Viewer** — Source/Output, channel swatches, Classic/OCIO + monitor view.
    Timeline uses filename numbers. **Brightness** = exposure stops (click label to reset).
    Drag to pan (even without zoom), **F** or Fit to frame.

Requires `maketx` and/or `oiiotool`. The Windows zip ships them next to the TX Tools exe.
OCIO config comes from the `OCIO` env var or the OCIO Config field (not bundled).
