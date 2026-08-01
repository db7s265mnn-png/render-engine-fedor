// Smooth RGB→spectrum reflectance upsampling (Smits-style piecewise spectra).
// Used when materials/lights only author RGB (no measured SPD textures).
#pragma once

#include "render/spectrum.h"

namespace sol {

// Smits 1999-inspired basis: construct a non-negative spectrum that roughly
// matches the given linear RGB under CIE. Good for reflectances in [0,1].
inline SampledSpectrum rgbToSpectrumReflectance(Vec3 rgb, const SampledWavelengths& w) {
    SampledSpectrum s(w.n);
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    for (int i = 0; i < w.n; ++i) {
        const float lam = w.lambda[i];
        float v = 0.0f;
        // White base
        const float white = srMin(r, srMin(g, b));
        v += white;
        float rr = r - white, gg = g - white, bb = b - white;
        // Cyan / magenta / yellow residuals (Smits partitioning).
        if (rr >= gg && rr >= bb) {
            // red dominant → yellow + magenta
            const float yel = srMin(gg, bb) > 0 ? 0.0f : srMin(rr, gg);  // unused path kept simple
            (void)yel;
            if (gg > bb) {
                // yellow + red
                const float y = srMin(rr, gg);
                v += y * ((lam > 580.0f) ? 1.0f : (lam > 500.0f ? (lam - 500.0f) / 80.0f : 0.0f));
                rr -= y;
                gg -= y;
                v += rr * ((lam > 650.0f) ? 1.0f : (lam > 600.0f ? (lam - 600.0f) / 50.0f : 0.0f));
            } else {
                const float m = srMin(rr, bb);
                v += m * ((lam < 500.0f) ? 1.0f : (lam < 580.0f ? (580.0f - lam) / 80.0f : 0.0f));
                // plus residual red if any after magenta — approximate
                v += srMax(0.0f, rr - m) *
                     ((lam > 650.0f) ? 1.0f : (lam > 600.0f ? (lam - 600.0f) / 50.0f : 0.0f));
            }
        } else if (gg >= rr && gg >= bb) {
            if (rr > bb) {
                const float y = srMin(rr, gg);
                v += y * ((lam > 580.0f) ? 1.0f : (lam > 500.0f ? (lam - 500.0f) / 80.0f : 0.0f));
                v += srMax(0.0f, gg - y) *
                     ((lam > 500.0f && lam < 600.0f)
                          ? 1.0f
                          : (lam > 450.0f && lam <= 500.0f ? (lam - 450.0f) / 50.0f
                                                           : (lam >= 600.0f && lam < 650.0f ? (650.0f - lam) / 50.0f
                                                                                           : 0.0f)));
            } else {
                const float c = srMin(gg, bb);
                v += c * ((lam < 550.0f) ? 1.0f : (lam < 610.0f ? (610.0f - lam) / 60.0f : 0.0f));
            }
        } else {
            if (gg > rr) {
                const float c = srMin(gg, bb);
                v += c * ((lam < 550.0f) ? 1.0f : (lam < 610.0f ? (610.0f - lam) / 60.0f : 0.0f));
            } else {
                const float m = srMin(rr, bb);
                v += m * ((lam < 500.0f) ? 1.0f : (lam < 580.0f ? (580.0f - lam) / 80.0f : 0.0f));
                v += srMax(0.0f, bb - m) *
                     ((lam < 450.0f) ? 1.0f : (lam < 500.0f ? (500.0f - lam) / 50.0f : 0.0f));
            }
        }
        s.values[i] = srMax(0.0f, v);
    }
    return s;
}

inline SampledSpectrum rgbToSpectrumEmission(Vec3 rgb, const SampledWavelengths& w) {
    // Same smooth construction; emission can exceed 1.
    return rgbToSpectrumReflectance(rgb, w);
}

}  // namespace sol
