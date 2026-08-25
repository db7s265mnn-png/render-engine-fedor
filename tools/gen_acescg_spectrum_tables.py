#!/usr/bin/env python3
"""Jakob RGB→spectrum tables for ACEScg (AP1) under CIE D60.

Forward model matches pbrt SampledSpectrum::ToXYZ / CIE_Y then ACEScg:
  rgb = M_ACEScg @ (∫ rsp(λ) D60(λ) CMF(λ) dλ / CIE_Y)

RGBIlluminantSpectrum at runtime is scale * rsp * D60; RGBAlbedoSpectrum is rsp
and is lit by D60. Both tables share this integral.
"""
from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

LAMBDA_MIN = 360.0
LAMBDA_MAX = 830.0
RES = 16
CIE_Y = 106.85691710

# CIE 1931 2° 5 nm 360–830 (cie_tables.h).
CIE_X = np.array([
    0.00012990, 0.00023210, 0.00041490, 0.00074160, 0.00136800, 0.00223600, 0.00424300, 0.00765000,
    0.01431000, 0.02319000, 0.04351000, 0.07763000, 0.13438000, 0.21477000, 0.28390000, 0.32850000,
    0.34828000, 0.34806000, 0.33620000, 0.31870000, 0.29080000, 0.25110000, 0.19536000, 0.14210000,
    0.09564000, 0.05795001, 0.03201000, 0.01470000, 0.00490000, 0.00240000, 0.00930000, 0.02910000,
    0.06327000, 0.10960000, 0.16550000, 0.22574990, 0.29040000, 0.35970000, 0.43344990, 0.51205010,
    0.59450000, 0.67840000, 0.76210000, 0.84250000, 0.91630000, 0.97860000, 1.02630000, 1.05670000,
    1.06220000, 1.04560000, 1.00260000, 0.93840000, 0.85444990, 0.75140000, 0.64240000, 0.54190000,
    0.44790000, 0.36080000, 0.28350000, 0.21870000, 0.16490000, 0.12120000, 0.08740000, 0.06360000,
    0.04677000, 0.03290000, 0.02270000, 0.01584000, 0.01135916, 0.00811092, 0.00579035, 0.00410946,
    0.00289933, 0.00204919, 0.00143997, 0.00099995, 0.00069008, 0.00047602, 0.00033230, 0.00023483,
    0.00016615, 0.00011741, 0.00008308, 0.00005871, 0.00004151, 0.00002935, 0.00002067, 0.00001456,
    0.00001025, 0.00000722, 0.00000509, 0.00000358, 0.00000252, 0.00000178, 0.00000125,
], dtype=np.float64)
CIE_Y_TAB = np.array([
    0.00000392, 0.00000697, 0.00001239, 0.00002202, 0.00003900, 0.00006400, 0.00012000, 0.00021700,
    0.00039600, 0.00064000, 0.00121000, 0.00218000, 0.00400000, 0.00730000, 0.01160000, 0.01684000,
    0.02300000, 0.02980000, 0.03800000, 0.04800000, 0.06000000, 0.07390000, 0.09098000, 0.11260000,
    0.13902000, 0.16930000, 0.20802000, 0.25860000, 0.32300000, 0.40730000, 0.50300000, 0.60820000,
    0.71000000, 0.79320000, 0.86200000, 0.91485010, 0.95400000, 0.98030000, 0.99495010, 1.00000000,
    0.99500000, 0.97860000, 0.95200000, 0.91540000, 0.87000000, 0.81630000, 0.75700000, 0.69490000,
    0.63100000, 0.56680000, 0.50300000, 0.44120000, 0.38100000, 0.32100000, 0.26500000, 0.21700000,
    0.17500000, 0.13820000, 0.10700000, 0.08160000, 0.06100000, 0.04458000, 0.03200000, 0.02320000,
    0.01700000, 0.01192000, 0.00821000, 0.00572300, 0.00410200, 0.00292900, 0.00209100, 0.00148400,
    0.00104700, 0.00074000, 0.00052000, 0.00036110, 0.00024920, 0.00017190, 0.00012000, 0.00008480,
    0.00006000, 0.00004240, 0.00003000, 0.00002120, 0.00001499, 0.00001060, 0.00000747, 0.00000526,
    0.00000370, 0.00000261, 0.00000184, 0.00000129, 0.00000091, 0.00000064, 0.00000045,
], dtype=np.float64)
CIE_Z = np.array([
    0.00060610, 0.00108600, 0.00194600, 0.00348600, 0.00645000, 0.01054999, 0.02005001, 0.03621000,
    0.06785001, 0.11020000, 0.20740000, 0.37130000, 0.64560000, 1.03905010, 1.38560000, 1.62296000,
    1.74706000, 1.78260000, 1.77211000, 1.74410000, 1.66920000, 1.52810000, 1.28764000, 1.04190000,
    0.81295010, 0.61620000, 0.46518000, 0.35330000, 0.27200000, 0.21230000, 0.15820000, 0.11170000,
    0.07824999, 0.05725001, 0.04216000, 0.02984000, 0.02030000, 0.01340000, 0.00875000, 0.00575000,
    0.00390000, 0.00275000, 0.00210000, 0.00180000, 0.00165000, 0.00140000, 0.00110000, 0.00100000,
    0.00080000, 0.00060000, 0.00034000, 0.00024000, 0.00019000, 0.00010000, 0.00005000, 0.00003000,
    0.00002000, 0.00001000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
], dtype=np.float64)
CIE_LAM = np.linspace(360.0, 830.0, 95)

# pbrt ACES_Illum_D60 5 nm 300–830, then Y-normalized (CIE_Y / InnerProduct).
D60_LAM = np.arange(300.0, 830.0 + 1e-6, 5.0)
D60_RAW = np.array([
    0.02928, 1.28964, 2.55, 9.0338, 15.5176, 21.94705, 28.3765, 29.93335, 31.4902, 33.75765,
    36.0251, 37.2032, 38.3813, 40.6445, 42.9077, 42.05735, 41.207, 43.8121, 46.4172, 59.26285,
    72.1085, 76.1756, 80.2427, 81.4878, 82.7329, 80.13505, 77.5372, 86.5577, 95.5782, 101.72045,
    107.8627, 108.67115, 109.4796, 108.5873, 107.695, 108.6598, 109.6246, 106.6426, 103.6606,
    104.42795, 105.1953, 104.7974, 104.3995, 103.45635, 102.5132, 104.2813, 106.0494, 104.67885,
    103.3083, 103.4228, 103.5373, 101.76865, 100.0, 98.3769, 96.7538, 96.73515, 96.7165, 93.3013,
    89.8861, 90.91705, 91.948, 91.98965, 92.0313, 91.3008, 90.5703, 88.5077, 86.4451, 86.9551,
    87.4651, 85.6558, 83.8465, 84.20755, 84.5686, 85.9432, 87.3178, 85.3068, 83.2958, 78.66005,
    74.0243, 75.23535, 76.4464, 77.67465, 78.9029, 72.12575, 65.3486, 69.6609, 73.9732, 76.6802,
    79.3872, 73.28855, 67.1899, 58.18595, 49.182, 59.9723, 70.7626, 68.9039, 67.0452, 67.5469,
    68.0486, 65.4631, 62.8776, 58.88595, 54.8943, 57.8066, 60.7189, 62.2491, 63.7793,
], dtype=np.float64)

# ACEScg XYZ→RGB (color_space.h).
M_ACES = np.array(
    [
        [1.6410233797, -0.3248032942, -0.2364246952],
        [-0.6636628587, 1.6153315917, 0.0167563477],
        [0.0117218943, -0.0082844420, 0.9883948585],
    ],
    dtype=np.float64,
)
M_ACES_INV = np.linalg.inv(M_ACES)


def _d60_normalized(lam: np.ndarray) -> np.ndarray:
    grid = np.arange(360.0, 831.0)
    s = np.interp(grid, D60_LAM, D60_RAW)
    y = np.interp(grid, CIE_LAM, CIE_Y_TAB)
    inner = float(np.trapezoid(s * y, grid))
    scale = CIE_Y / inner
    return np.interp(lam, D60_LAM, D60_RAW) * scale


def build_weights():
    lam = CIE_LAM.copy()
    h = (LAMBDA_MAX - LAMBDA_MIN) / (len(lam) - 1)
    w = np.full(len(lam), h)
    w[0] *= 0.5
    w[-1] *= 0.5
    d60 = _d60_normalized(lam)
    # ∫ rsp D60 CMF dλ / CIE_Y  →  ACEScg
    xyz_basis = np.stack([CIE_X * d60 * w, CIE_Y_TAB * d60 * w, CIE_Z * d60 * w], axis=0) / CIE_Y
    rgb_basis = M_ACES @ xyz_basis
    lam01 = (lam - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)
    d60_xyz = M_ACES @ (np.stack([
        float(np.sum(CIE_X * d60 * w)),
        float(np.sum(CIE_Y_TAB * d60 * w)),
        float(np.sum(CIE_Z * d60 * w)),
    ]) / CIE_Y)
    print(f"D60 → ACEScg white {d60_xyz}", flush=True)
    return lam, lam01, rgb_basis.astype(np.float64)


LAM, LAM01, RGB_W = build_weights()
D60_WP = M_ACES_INV @ np.array([1.0, 1.0, 1.0])  # ACEScg white → XYZ (should be D60)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 + 0.5 * x / np.sqrt(1.0 + x * x)


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    return 0.5 / np.power(1.0 + x * x, 1.5)


def eval_spectrum_rgb(coeffs01: np.ndarray) -> np.ndarray:
    c0, c1, c2 = coeffs01
    p = (c0 * LAM01 + c1) * LAM01 + c2
    return RGB_W @ sigmoid(p)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = M_ACES_INV @ np.maximum(rgb, 0.0)
    wn = np.array([0.9526456, 1.0, 1.0086820])  # Y-normalized D60

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
    dout = np.zeros((3, 3), dtype=np.float64)
    dout[:, 0] = RGB_W @ (sp * (LAM01 * LAM01))
    dout[:, 1] = RGB_W @ (sp * LAM01)
    dout[:, 2] = RGB_W @ sp
    if mode == "lab":
        eps = 1e-5
        lab_out = rgb_to_lab(out)
        lab_jac = np.zeros((3, 3), dtype=np.float64)
        for k in range(3):
            o2 = out.copy()
            o2[k] += eps
            lab_jac[:, k] = (rgb_to_lab(o2) - lab_out) / eps
        return rgb_to_lab(target_rgb) - lab_out, -lab_jac @ dout
    return target_rgb - out, -dout


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

    body = f"""// Auto-generated by tools/gen_acescg_spectrum_tables.py — do not edit by hand.
// Jakob & Hanika ACEScg (AP1) tables. Objective: ∫ rsp(λ) D60(λ) CMF dλ / CIE_Y → ACEScg.
// Resolution {res}.
#include "render/rgb_spectrum_tables.h"

namespace sol {{
namespace rgb_spec {{

constexpr int kAcesTableRes = {res};

{fmt_arr("kAcesAlbedoScale", albedo_scale)}

{fmt_arr("kAcesAlbedoCoeffs", albedo)}

{fmt_arr("kAcesIlluminantScale", illum_scale)}

{fmt_arr("kAcesIlluminantCoeffs", illum)}

namespace {{

inline int findScaleIntervalAces(const float* nodes, int n, float z) {{
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

inline RgbSigmoidPolynomial fetchTableAces(const float* scale, const float* coeffs, Vec3 rgb) {{
    float r = clampf(rgb.x, 0.0f, 1.0f);
    float g = clampf(rgb.y, 0.0f, 1.0f);
    float b = clampf(rgb.z, 0.0f, 1.0f);

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

    const int res = kAcesTableRes;
    const float x = rgbv[(maxc + 1) % 3] * float(res - 1) / z;
    const float y = rgbv[(maxc + 2) % 3] * float(res - 1) / z;
    const int xi = int(x) < res - 2 ? int(x) : res - 2;
    const int yi = int(y) < res - 2 ? int(y) : res - 2;
    const int zi = findScaleIntervalAces(scale, res, z);
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

RgbSigmoidPolynomial fetchAlbedoAces(Vec3 rgb) {{
    return fetchTableAces(kAcesAlbedoScale, kAcesAlbedoCoeffs, rgb);
}}

RgbSigmoidPolynomial fetchIlluminantAces(Vec3 rgb) {{
    return fetchTableAces(kAcesIlluminantScale, kAcesIlluminantCoeffs, rgb);
}}

}}  // namespace rgb_spec
}}  // namespace sol
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


def main():
    out = os.path.join(os.path.dirname(__file__), "..", "src", "render", "rgb_spectrum_tables_aces.cpp")
    out = os.path.normpath(out)
    print(f"Generating ACEScg albedo (Lab) res={RES}...")
    a_scale, a_data = generate_table("lab", RES)
    print(f"Generating ACEScg illuminant (linear RGB) res={RES}...")
    i_scale, i_data = generate_table("rgb", RES)
    emit_cpp(out, a_scale, a_data, i_scale, i_data, RES)


if __name__ == "__main__":
    main()
