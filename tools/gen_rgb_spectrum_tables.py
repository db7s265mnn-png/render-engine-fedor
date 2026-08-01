#!/usr/bin/env python3
"""Generate Jakob & Hanika (2019) RGB→spectrum coefficient tables for Solstice.

Tables are optimized so that the continuous equal-energy white-balanced
linear-sRGB conversion (matching spectrum.h / spectrumToRgb) reproduces the
target RGB. Albedo uses CIE Lab residual; illuminant uses linear RGB residual.
"""
from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

LAMBDA_MIN = 360.0
LAMBDA_MAX = 830.0
RES = 16  # compact; round-trip polish recovers residual error


def cie_xyz_at_lambda(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match spectrum.h analytic CIE fit (Wyman-style Gaussians)."""

    def gauss(lam, mu, s1, s2):
        g = np.where(lam < mu, s1, s2)
        d = (lam - mu) / g
        return np.exp(-0.5 * d * d)

    x = 1.065 * gauss(lam, 595.8, 33.33, 37.05) + 0.366 * gauss(lam, 446.8, 16.01, 22.40)
    y = 1.014 * gauss(lam, 556.7, 46.07, 40.83)
    z = 1.839 * gauss(lam, 449.1, 19.44, 28.54)
    return x, y, z


def xyz_to_linear_srgb_vec(X, Y, Z):
    r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    return np.array([r, g, b], dtype=np.float64)


def build_weights(n: int = 95):
    """Quadrature weights mapping sigmoid spectrum → our white-balanced linear sRGB."""
    lam = np.linspace(LAMBDA_MIN, LAMBDA_MAX, n)
    # Trapezoid weights on wavelength
    h = (LAMBDA_MAX - LAMBDA_MIN) / (n - 1)
    w = np.full(n, h)
    w[0] *= 0.5
    w[-1] *= 0.5

    cx, cy, cz = cie_xyz_at_lambda(lam)
    # Equal-energy white integrals (spectrumToRgb reference white).
    Xw = float(np.sum(cx * w))
    Yw = float(np.sum(cy * w))
    Zw = float(np.sum(cz * w))

    # Contribution of s(λ_i) to relative XYZ, then to white-balanced linear sRGB.
    # X_rel = sum(s * cx * w) / Xw  → rgb = xyzToSrgb(X_rel) / xyzToSrgb(1,1,1)
    white_rgb = xyz_to_linear_srgb_vec(1.0, 1.0, 1.0)
    # Precompute: rgb_k = sum_i M[k,i] * s_i
    # M comes from: srgb = (A @ [X/Xw, Y/Yw, Z/Zw]) / white_rgb
    # where A is XYZ→sRGB matrix.
    A = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    # d(rgb_out)/d(s_i) = (A @ diag(1/Xw,1/Yw,1/Zw) @ [cx,cy,cz]_i * w_i) / white_rgb
    xyz_basis = np.stack([cx / Xw, cy / Yw, cz / Zw], axis=0)  # 3 x n
    rgb_basis = (A @ xyz_basis) * w[None, :]  # 3 x n
    rgb_basis /= white_rgb[:, None]

    # Scaled λ in [0,1] for well-conditioned polynomials during optimization.
    lam01 = (lam - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)
    return lam, lam01, rgb_basis.astype(np.float64)


LAM, LAM01, RGB_W = build_weights()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 + 0.5 * x / np.sqrt(1.0 + x * x)


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    # d/dx [0.5 + 0.5 x / sqrt(1+x^2)] = 0.5 / (1+x^2)^{3/2}
    return 0.5 / np.power(1.0 + x * x, 1.5)


def eval_spectrum_rgb(coeffs01: np.ndarray) -> np.ndarray:
    """Evaluate RGB of Jakob spectrum with coeffs in scaled-[0,1] wavelength domain."""
    c0, c1, c2 = coeffs01
    p = (c0 * LAM01 + c1) * LAM01 + c2
    s = sigmoid(p)
    return RGB_W @ s


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Approximate CIE Lab via XYZ of our white-balanced linear RGB (D65-ish)."""
    # Invert white-balanced sRGB → relative XYZ using inverse of our pipeline.
    # white_rgb scaling first.
    white_rgb = xyz_to_linear_srgb_vec(1.0, 1.0, 1.0)
    linear = np.maximum(rgb, 0.0) * white_rgb
    # sRGB→XYZ (inverse of matrix in spectrum.h)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = M @ linear
    # Whitepoint: equal-energy mapped through same path → (1,1,1) relative XYZ
    wn = np.array([1.0, 1.0, 1.0])

    def f(t):
        d = 6.0 / 29.0
        return np.where(t > d**3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)

    ft = f(xyz / wn)
    return np.array([116.0 * ft[1] - 16.0, 500.0 * (ft[0] - ft[1]), 200.0 * (ft[1] - ft[2])])


def residual(coeffs01: np.ndarray, target_rgb: np.ndarray, mode: str) -> np.ndarray:
    out = eval_spectrum_rgb(coeffs01)
    if mode == "lab":
        return rgb_to_lab(target_rgb) - rgb_to_lab(out)
    return target_rgb - out


def residual_jac(coeffs01: np.ndarray, target_rgb: np.ndarray, mode: str):
    c0, c1, c2 = coeffs01
    p = (c0 * LAM01 + c1) * LAM01 + c2
    s = sigmoid(p)
    sp = sigmoid_deriv(p)
    out = RGB_W @ s
    # dout/dc: 3x3
    dout = np.zeros((3, 3), dtype=np.float64)
    dout[:, 0] = RGB_W @ (sp * (LAM01 * LAM01))
    dout[:, 1] = RGB_W @ (sp * LAM01)
    dout[:, 2] = RGB_W @ sp

    if mode == "lab":
        # Numerical Lab jacobian w.r.t. out (3x3) — small and stable enough.
        eps = 1e-5
        lab_out = rgb_to_lab(out)
        lab_jac = np.zeros((3, 3), dtype=np.float64)
        for k in range(3):
            o2 = out.copy()
            o2[k] += eps
            lab_jac[:, k] = (rgb_to_lab(o2) - lab_out) / eps
        J = -lab_jac @ dout
        res = rgb_to_lab(target_rgb) - lab_out
        return res, J

    res = target_rgb - out
    J = -dout
    return res, J


def lm_fit(target_rgb: np.ndarray, coeffs01: np.ndarray, mode: str, iters: int = 20) -> np.ndarray:
    c = coeffs01.astype(np.float64).copy()
    res, J = residual_jac(c, target_rgb, mode)
    cost = float(np.dot(res, res))
    lam = 1e-3
    for _ in range(iters):
        if cost < 1e-14:
            break
        A = J.T @ J
        g = J.T @ res
        accepted = False
        for _t in range(12):
            M = A + lam * np.eye(3)
            try:
                step = np.linalg.solve(M, g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            trial = c - step
            trial_res = residual(trial, target_rgb, mode)
            trial_cost = float(np.dot(trial_res, trial_res))
            if trial_cost < cost:
                c = trial
                cost = trial_cost
                lam = max(lam * 0.5, 1e-12)
                accepted = True
                break
            lam *= 10.0
            if lam > 1e12:
                break
        if not accepted:
            break
        res, J = residual_jac(c, target_rgb, mode)
    return c


def coeffs01_to_nm(c01: np.ndarray) -> np.ndarray:
    """Remap polynomial from λ∈[0,1] to λ in nanometres (PBRT / rgb2spec convention)."""
    c0 = LAMBDA_MIN
    c1 = 1.0 / (LAMBDA_MAX - LAMBDA_MIN)
    A, B, C = c01
    return np.array(
        [
            A * (c1 * c1),
            B * c1 - 2 * A * c0 * (c1 * c1),
            C - B * c0 * c1 + A * (c0 * c1) * (c0 * c1),
        ],
        dtype=np.float64,
    )


def smoothstep(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def generate_table(mode: str, res: int = RES) -> tuple[np.ndarray, np.ndarray]:
    scale = np.array([smoothstep(smoothstep(k / (res - 1))) for k in range(res)], dtype=np.float64)
    data = np.zeros((3, res, res, res, 3), dtype=np.float32)

    def work(l: int, j: int):
        y = j / (res - 1)
        local = np.zeros((res, res, 3), dtype=np.float32)
        for i in range(res):
            x = i / (res - 1)
            coeffs = np.zeros(3, dtype=np.float64)
            # Seed from mid brightness outward (Jakob rgb2spec_opt).
            start = res // 5
            for k in range(start, res):
                b = scale[k]
                rgb = np.zeros(3)
                rgb[l] = b
                rgb[(l + 1) % 3] = x * b
                rgb[(l + 2) % 3] = y * b
                if abs(rgb[0] - rgb[1]) < 1e-12 and abs(rgb[1] - rgb[2]) < 1e-12:
                    v = float(rgb[0])
                    if v <= 0:
                        c01 = np.array([0.0, 0.0, -8192.0])
                    elif v >= 1:
                        c01 = np.array([0.0, 0.0, 8192.0])
                    else:
                        c01 = np.array([0.0, 0.0, (v - 0.5) / math.sqrt(v * (1 - v))])
                else:
                    coeffs = lm_fit(rgb, coeffs, mode)
                    c01 = coeffs
                local[k, i] = coeffs01_to_nm(c01).astype(np.float32)

            coeffs = np.zeros(3, dtype=np.float64)
            for k in range(start, -1, -1):
                b = scale[k]
                rgb = np.zeros(3)
                rgb[l] = b
                rgb[(l + 1) % 3] = x * b
                rgb[(l + 2) % 3] = y * b
                if abs(rgb[0] - rgb[1]) < 1e-12 and abs(rgb[1] - rgb[2]) < 1e-12:
                    v = float(rgb[0])
                    if v <= 0:
                        c01 = np.array([0.0, 0.0, -8192.0])
                    elif v >= 1:
                        c01 = np.array([0.0, 0.0, 8192.0])
                    else:
                        c01 = np.array([0.0, 0.0, (v - 0.5) / math.sqrt(v * (1 - v))])
                else:
                    coeffs = lm_fit(rgb, coeffs, mode)
                    c01 = coeffs
                local[k, i] = coeffs01_to_nm(c01).astype(np.float32)
        return l, j, local

    tasks = [(l, j) for l in range(3) for j in range(res)]
    done = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        futs = [ex.submit(work, l, j) for l, j in tasks]
        for fut in as_completed(futs):
            l, j, local = fut.result()
            # local[k, i, coeff] → data[l, k, j, i, coeff]
            data[l, :, j, :, :] = local
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  {mode}: {done}/{len(tasks)}", flush=True)
    return scale.astype(np.float32), data


def emit_cpp(path: str, albedo_scale, albedo, illum_scale, illum, res: int):
    def fmt_arr(name: str, arr: np.ndarray) -> str:
        flat = arr.ravel()
        lines = [f"const float {name}[{flat.size}] = {{"]

        def fmt_f(v: float) -> str:
            # Default float formatting always includes a decimal / exponent
            # (avoids invalid C++ literals like `0f` / `8192f`).
            return f"{float(v)}f"

        row = []
        for i, v in enumerate(flat):
            row.append(fmt_f(float(v)))
            if len(row) == 8:
                lines.append("    " + ", ".join(row) + ",")
                row = []
        if row:
            lines.append("    " + ", ".join(row) + ",")
        lines.append("};")
        return "\n".join(lines)

    body = f"""// Auto-generated by tools/gen_rgb_spectrum_tables.py — do not edit by hand.
// Jakob & Hanika (2019) RGBSigmoidPolynomial coefficient tables for linear sRGB.
// Resolution {res}; equal-energy white-balanced objective matching spectrumToRgb.
#include "render/rgb_spectrum_tables.h"

namespace sol {{
namespace rgb_spec {{

constexpr int kTableRes = {res};

{fmt_arr("kAlbedoScale", albedo_scale)}

{fmt_arr("kAlbedoCoeffs", albedo)}

{fmt_arr("kIlluminantScale", illum_scale)}

{fmt_arr("kIlluminantCoeffs", illum)}

namespace {{

inline int findScaleInterval(const float* nodes, int n, float z) {{
    int left = 0;
    int size = n - 2;
    while (size > 0) {{
        const int half = size >> 1;
        const int mid = left + half + 1;
        if (nodes[mid] <= z) {{
            left = mid;
            size -= half + 1;
        }} else {{
            size = half;
        }}
    }}
    return left < n - 2 ? left : n - 2;
}}

inline RgbSigmoidPolynomial fetchTable(const float* scale, const float* coeffs, Vec3 rgb) {{
    float r = clampf(rgb.x, 0.0f, 1.0f);
    float g = clampf(rgb.y, 0.0f, 1.0f);
    float b = clampf(rgb.z, 0.0f, 1.0f);

    // Closed form for greys (constant spectrum).
    if (r == g && g == b) {{
        float c2;
        if (r <= 0.0f)
            c2 = -8192.0f;
        else if (r >= 1.0f)
            c2 = 8192.0f;
        else
            c2 = (r - 0.5f) / sqrtf(r * (1.0f - r));
        return RgbSigmoidPolynomial{{0.0f, 0.0f, c2}};
    }}

    int maxc = (r > g) ? ((r > b) ? 0 : 2) : ((g > b) ? 1 : 2);
    const float rgbv[3] = {{r, g, b}};
    const float z = rgbv[maxc];
    if (z <= 0.0f)
        return RgbSigmoidPolynomial{{-8192.0f, 0.0f, 0.0f}};

    const int res = kTableRes;
    const float x = rgbv[(maxc + 1) % 3] * float(res - 1) / z;
    const float y = rgbv[(maxc + 2) % 3] * float(res - 1) / z;
    const int xi = int(x) < res - 2 ? int(x) : res - 2;
    const int yi = int(y) < res - 2 ? int(y) : res - 2;
    const int zi = findScaleInterval(scale, res, z);
    const float dx = x - float(xi);
    const float dy = y - float(yi);
    const float dz = (z - scale[zi]) / (scale[zi + 1] - scale[zi]);

    auto co = [&](int di, int dj, int dk, int ci) -> float {{
        const int idx =
            ((((maxc * res + (zi + dk)) * res + (yi + dj)) * res + (xi + di)) * 3) + ci;
        return coeffs[idx];
    }};

    float c[3];
    for (int ci = 0; ci < 3; ++ci) {{
        const float c00 = co(0, 0, 0, ci) * (1 - dx) + co(1, 0, 0, ci) * dx;
        const float c10 = co(0, 1, 0, ci) * (1 - dx) + co(1, 1, 0, ci) * dx;
        const float c01 = co(0, 0, 1, ci) * (1 - dx) + co(1, 0, 1, ci) * dx;
        const float c11 = co(0, 1, 1, ci) * (1 - dx) + co(1, 1, 1, ci) * dx;
        const float c0 = c00 * (1 - dy) + c10 * dy;
        const float c1 = c01 * (1 - dy) + c11 * dy;
        c[ci] = c0 * (1 - dz) + c1 * dz;
    }}
    return RgbSigmoidPolynomial{{c[0], c[1], c[2]}};
}}

}}  // namespace

RgbSigmoidPolynomial fetchAlbedo(Vec3 rgb) {{
    return fetchTable(kAlbedoScale, kAlbedoCoeffs, rgb);
}}

RgbSigmoidPolynomial fetchIlluminant(Vec3 rgb) {{
    return fetchTable(kIlluminantScale, kIlluminantCoeffs, rgb);
}}

}}  // namespace rgb_spec
}}  // namespace sol
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


def main():
    out = os.path.join(os.path.dirname(__file__), "..", "src", "render", "rgb_spectrum_tables.cpp")
    out = os.path.normpath(out)
    print(f"Generating albedo (Lab) res={RES}...")
    a_scale, a_data = generate_table("lab", RES)
    print(f"Generating illuminant (linear RGB) res={RES}...")
    i_scale, i_data = generate_table("rgb", RES)

    # Quick validation
    def eval_nm(c, lam):
        p = (c[0] * lam + c[1]) * lam + c[2]
        return 0.5 + 0.5 * p / np.sqrt(1 + p * p)

    # Spot-check white via mono path is exact; check a red via continuous RGB_W
    print("Spot checks done by C++ tests.")
    emit_cpp(out, a_scale, a_data, i_scale, i_data, RES)


if __name__ == "__main__":
    main()
