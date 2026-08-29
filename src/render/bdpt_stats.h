// Aggregated CPU-BDPT timers (Render Settings → Diagnostic → BDPT Timers).
// Cheap atomics, one log line block per sample. Off = no clocks in the integrator.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

#include "core/log.h"

namespace sol {

struct BdptPassStats {
    std::atomic<uint64_t> nsTotal{0};     // whole Li()
    std::atomic<uint64_t> nsAlloc{0};     // 6× std::vector construction
    std::atomic<uint64_t> nsWalk{0};      // eye + light randomWalk
    std::atomic<uint64_t> nsSss{0};       // Chiang walk (subset of nsWalk)
    std::atomic<uint64_t> nsConnect{0};   // s=0 / s=1 / s≥2 + photons
    std::atomic<uint64_t> nsSplat{0};     // t=1 light tracing
    std::atomic<uint64_t> pixels{0};
    std::atomic<uint64_t> nEyeSum{0};
    std::atomic<uint64_t> nLightSum{0};
    std::atomic<uint64_t> nEyeMax{0};
    std::atomic<uint64_t> nLightMax{0};
    std::atomic<uint64_t> pairs{0};          // connectable (s≥2) attempts
    std::atomic<uint64_t> shadows{0};        // connection / NEE visibility rays
    std::atomic<uint64_t> splatDeposits{0};  // addSplat that reached the plane
    std::atomic<uint64_t> casRetries{0};     // failed compare_exchange in addSplat
};

struct BdptPassMeta {
    int spp = 0;
    int maxDepth = 0;
    int maxVerts = 0;
    int poolThreads = 0;
    size_t vertBytes = 0;
    size_t allocBytesPerPixel = 0;
    uint64_t wallNs = 0;
};

struct BdptPhaseTimer {
    std::atomic<uint64_t>* acc = nullptr;
    std::chrono::steady_clock::time_point t0{};

    explicit BdptPhaseTimer(std::atomic<uint64_t>* a) : acc(a) {
        if (acc) t0 = std::chrono::steady_clock::now();
    }
    BdptPhaseTimer(const BdptPhaseTimer&) = delete;
    BdptPhaseTimer& operator=(const BdptPhaseTimer&) = delete;
    ~BdptPhaseTimer() {
        if (!acc) return;
        const uint64_t ns = uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
                .count());
        acc->fetch_add(ns, std::memory_order_relaxed);
    }
};

inline void bdptAtomicMax(std::atomic<uint64_t>& a, uint64_t v) {
    uint64_t prev = a.load(std::memory_order_relaxed);
    while (v > prev && !a.compare_exchange_weak(prev, v, std::memory_order_relaxed)) {
    }
}

inline double bdptNsToMs(uint64_t ns) { return double(ns) * 1.0e-6; }

inline std::string formatBdptPassStats(const BdptPassStats& s, const BdptPassMeta& meta) {
    const uint64_t pixels = s.pixels.load(std::memory_order_relaxed);
    const uint64_t alloc = s.nsAlloc.load(std::memory_order_relaxed);
    const uint64_t walk = s.nsWalk.load(std::memory_order_relaxed);
    const uint64_t sss = s.nsSss.load(std::memory_order_relaxed);
    const uint64_t connect = s.nsConnect.load(std::memory_order_relaxed);
    const uint64_t splat = s.nsSplat.load(std::memory_order_relaxed);
    const uint64_t total = s.nsTotal.load(std::memory_order_relaxed);
    const uint64_t phases = alloc + walk + connect + splat;
    const uint64_t other = total > phases ? total - phases : 0;
    const uint64_t denom = total > 0 ? total : 1;

    auto pct = [&](uint64_t ns) { return 100.0 * double(ns) / double(denom); };
    auto perPx = [&](uint64_t v) { return pixels > 0 ? double(v) / double(pixels) : 0.0; };

    const double wallMs = bdptNsToMs(meta.wallNs);
    const double threadMs = bdptNsToMs(total);
    const double parallel = wallMs > 1e-6 ? threadMs / wallMs : 0.0;
    const double allocKiB = double(meta.allocBytesPerPixel) / 1024.0;

    std::string out;
    out.reserve(900);
    char line[384];

    std::snprintf(line, sizeof(line),
                  "BDPT timers  spp=%d  maxDepth=%d  maxVerts=%d  poolThreads=%d (+caller)  "
                  "pixels=%llu  wall=%.1f ms  thread=%.1f ms  parallel=%.2fx\n",
                  meta.spp, meta.maxDepth, meta.maxVerts, meta.poolThreads,
                  static_cast<unsigned long long>(pixels), wallMs, threadMs, parallel);
    out += line;

    std::snprintf(line, sizeof(line),
                  "  alloc    %8.1f ms  %5.1f%%   (%d verts, Vert=%zu B, %.1f KiB/pixel)\n",
                  bdptNsToMs(alloc), pct(alloc), meta.maxVerts, meta.vertBytes, allocKiB);
    out += line;
    std::snprintf(line, sizeof(line), "  walk     %8.1f ms  %5.1f%%\n", bdptNsToMs(walk), pct(walk));
    out += line;
    std::snprintf(line, sizeof(line), "    sss    %8.1f ms  %5.1f%%  (inside walk)\n", bdptNsToMs(sss),
                  pct(sss));
    out += line;
    std::snprintf(line, sizeof(line), "  connect  %8.1f ms  %5.1f%%\n", bdptNsToMs(connect), pct(connect));
    out += line;
    std::snprintf(line, sizeof(line), "  splat    %8.1f ms  %5.1f%%\n", bdptNsToMs(splat), pct(splat));
    out += line;
    std::snprintf(line, sizeof(line), "  other    %8.1f ms  %5.1f%%\n", bdptNsToMs(other), pct(other));
    out += line;

    std::snprintf(line, sizeof(line),
                  "  per pixel: nEye=%.2f (max %llu)  nLight=%.2f (max %llu)  pairs=%.1f  "
                  "shadows=%.1f  splats=%.2f  casRetries=%.1f\n",
                  perPx(s.nEyeSum.load(std::memory_order_relaxed)),
                  static_cast<unsigned long long>(s.nEyeMax.load(std::memory_order_relaxed)),
                  perPx(s.nLightSum.load(std::memory_order_relaxed)),
                  static_cast<unsigned long long>(s.nLightMax.load(std::memory_order_relaxed)),
                  perPx(s.pairs.load(std::memory_order_relaxed)),
                  perPx(s.shadows.load(std::memory_order_relaxed)),
                  perPx(s.splatDeposits.load(std::memory_order_relaxed)),
                  perPx(s.casRetries.load(std::memory_order_relaxed)));
    out += line;
    return out;
}

inline void logBdptPassStats(const BdptPassStats& s, const BdptPassMeta& meta) {
    const std::string text = formatBdptPassStats(s, meta);
    // One logInfo per line so the UI log panel stays readable.
    std::string line;
    for (char c : text) {
        if (c == '\n') {
            if (!line.empty()) logInfo(line);
            line.clear();
        } else {
            line.push_back(c);
        }
    }
    if (!line.empty()) logInfo(line);
}

}  // namespace sol
