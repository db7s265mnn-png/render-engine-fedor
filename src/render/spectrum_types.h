// First-class spectral distributions (pbrt-v4 subset): sample into SampledSpectrum.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/math.h"
#include "render/spectrum.h"

namespace sol {

struct ConstantSpectrum {
    float c = 0.0f;
    explicit ConstantSpectrum(float c_ = 0.0f) : c(c_) {}
    float operator()(float) const { return c; }
    float maxValue() const { return c; }
    SampledSpectrum sample(const SampledWavelengths& w) const {
        return SampledSpectrum::constant(w.n, c);
    }
};

struct DenselySampledSpectrum {
    int lambdaMin = int(kSpectrumLambdaMin);
    int lambdaMax = int(kSpectrumLambdaMax);
    std::vector<float> values;  // 1 nm steps

    DenselySampledSpectrum() = default;
    DenselySampledSpectrum(int lamMin, int lamMax)
        : lambdaMin(lamMin), lambdaMax(lamMax), values(size_t(std::max(0, lamMax - lamMin + 1)), 0.0f) {}

    template <typename Fn>
    static DenselySampledSpectrum fromFunction(Fn&& fn, int lamMin = int(kSpectrumLambdaMin),
                                               int lamMax = int(kSpectrumLambdaMax)) {
        DenselySampledSpectrum s(lamMin, lamMax);
        for (int lam = lamMin; lam <= lamMax; ++lam)
            s.values[size_t(lam - lamMin)] = float(fn(float(lam)));
        return s;
    }

    float operator()(float lambda) const {
        const int offset = int(std::lround(lambda)) - lambdaMin;
        if (offset < 0 || offset >= int(values.size())) return 0.0f;
        return values[size_t(offset)];
    }
    float maxValue() const {
        float m = 0.0f;
        for (float v : values) m = srMax(m, v);
        return m;
    }
    SampledSpectrum sample(const SampledWavelengths& w) const {
        SampledSpectrum s(w.n);
        for (int i = 0; i < w.n; ++i) s.values[i] = (*this)(w.lambda[i]);
        return s;
    }
};

struct PiecewiseLinearSpectrum {
    std::vector<float> lambdas;
    std::vector<float> values;

    PiecewiseLinearSpectrum() = default;
    PiecewiseLinearSpectrum(std::vector<float> lams, std::vector<float> vals)
        : lambdas(std::move(lams)), values(std::move(vals)) {}

    float operator()(float lambda) const {
        if (lambdas.empty()) return 0.0f;
        if (lambda <= lambdas.front()) return values.front();
        if (lambda >= lambdas.back()) return values.back();
        const auto it = std::upper_bound(lambdas.begin(), lambdas.end(), lambda);
        const size_t i1 = size_t(it - lambdas.begin());
        const size_t i0 = i1 - 1;
        const float t =
            (lambda - lambdas[i0]) / srMax(1e-6f, lambdas[i1] - lambdas[i0]);
        return values[i0] * (1.0f - t) + values[i1] * t;
    }
    float maxValue() const {
        float m = 0.0f;
        for (float v : values) m = srMax(m, v);
        return m;
    }
    SampledSpectrum sample(const SampledWavelengths& w) const {
        SampledSpectrum s(w.n);
        for (int i = 0; i < w.n; ++i) s.values[i] = (*this)(w.lambda[i]);
        return s;
    }
};

// Planck blackbody spectral radiance (W / (sr·m²·nm) relative shape). T in Kelvin.
struct BlackbodySpectrum {
    float T = 6500.0f;
    explicit BlackbodySpectrum(float kelvin = 6500.0f) : T(srMax(100.0f, kelvin)) {}

    static float radiance(float lambdaNm, float kelvin) {
        // pbrt-style constants; λ in meters.
        const float lambda = lambdaNm * 1e-9f;
        if (!(lambda > 0.0f) || !(kelvin > 0.0f)) return 0.0f;
        constexpr float c = 299792458.0f;
        constexpr float h = 6.62606957e-34f;
        constexpr float kb = 1.3806488e-23f;
        const float l5 = (lambda * lambda) * (lambda * lambda) * lambda;
        const float exponent = h * c / (lambda * kb * kelvin);
        if (exponent > 80.0f) return 0.0f;  // underflow
        return (2.0f * h * c * c) / (l5 * (expf(exponent) - 1.0f));
    }

    float operator()(float lambdaNm) const { return radiance(lambdaNm, T); }
    float maxValue() const {
        // Wien peak (~nm): 2.897e6 / T
        const float peakNm = clampf(2.897771955e6f / T, kSpectrumLambdaMin, kSpectrumLambdaMax);
        return radiance(peakNm, T);
    }
    SampledSpectrum sample(const SampledWavelengths& w) const {
        SampledSpectrum s(w.n);
        const float norm = 1.0f / srMax(1e-30f, maxValue());
        for (int i = 0; i < w.n; ++i) s.values[i] = radiance(w.lambda[i], T) * norm;
        return s;
    }
};

}  // namespace sol
