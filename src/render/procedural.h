// Shade-time MaterialX procedural evaluation (Arnold/Karma style).
// Noise / math / triplanar run per ray hit from object P / UV — not baked to UV maps.
#pragma once

#include "core/math.h"
#include "scene/types.h"

namespace sol {

struct ProceduralCtx {
    Vec2 uv{0.0f, 0.0f};
    Vec3 pObject{0.0f, 0.0f, 0.0f};
    Vec3 nObject{0.0f, 0.0f, 1.0f};
    float filterWidth = 0.0f;
};

// ---------------------------------------------------------------------------
// Noise primitives (MaterialX / OSL style ≈ -1..1)
// ---------------------------------------------------------------------------
SR_INL SR_HD uint32_t procHashU32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

SR_INL SR_HD float procHashToFloat(uint32_t h) { return float(h >> 8) / float(0x00ffffffu); }

SR_INL SR_HD float procFade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

SR_INL SR_HD float procGrad2(uint32_t h, float x, float y) {
    const uint32_t g = h & 7u;
    const float u = (g < 4u) ? x : y;
    const float v = 2.0f * ((g < 4u) ? y : x);
    return ((g & 1u) ? -u : u) + ((g & 2u) ? -v : v);
}

SR_INL SR_HD float procGrad3(uint32_t h, float x, float y, float z) {
    const uint32_t g = h & 15u;
    const float u = (g < 8u) ? x : y;
    const float v = (g < 4u) ? y : ((g == 12u || g == 14u) ? x : z);
    return ((g & 1u) ? -u : u) + ((g & 2u) ? -v : v);
}

SR_INL SR_HD float procPerlin2(float x, float y) {
    if (!srIsFinite(x) || !srIsFinite(y)) return 0.0f;
    x = clampf(x, -1.0e5f, 1.0e5f);
    y = clampf(y, -1.0e5f, 1.0e5f);
    const int x0 = int(floorf(x));
    const int y0 = int(floorf(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const float u = procFade(fx);
    const float v = procFade(fy);
    const uint32_t h00 = procHashU32(uint32_t(x0) * 374761393u + uint32_t(y0) * 668265263u);
    const uint32_t h10 = procHashU32(uint32_t(x0 + 1) * 374761393u + uint32_t(y0) * 668265263u);
    const uint32_t h01 = procHashU32(uint32_t(x0) * 374761393u + uint32_t(y0 + 1) * 668265263u);
    const uint32_t h11 = procHashU32(uint32_t(x0 + 1) * 374761393u + uint32_t(y0 + 1) * 668265263u);
    const float n00 = procGrad2(h00, fx, fy);
    const float n10 = procGrad2(h10, fx - 1.0f, fy);
    const float n01 = procGrad2(h01, fx, fy - 1.0f);
    const float n11 = procGrad2(h11, fx - 1.0f, fy - 1.0f);
    const float nx0 = n00 * (1.0f - u) + n10 * u;
    const float nx1 = n01 * (1.0f - u) + n11 * u;
    return 0.7071f * (nx0 * (1.0f - v) + nx1 * v);
}

SR_INL SR_HD float procPerlin3(float x, float y, float z) {
    if (!srIsFinite(x) || !srIsFinite(y) || !srIsFinite(z)) return 0.0f;
    x = clampf(x, -1.0e5f, 1.0e5f);
    y = clampf(y, -1.0e5f, 1.0e5f);
    z = clampf(z, -1.0e5f, 1.0e5f);
    const int x0 = int(floorf(x));
    const int y0 = int(floorf(y));
    const int z0 = int(floorf(z));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const float fz = z - float(z0);
    const float u = procFade(fx);
    const float v = procFade(fy);
    const float w = procFade(fz);
    float n = 0.0f;
    for (int dz = 0; dz <= 1; ++dz) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                const uint32_t h = procHashU32(uint32_t(x0 + dx) * 374761393u + uint32_t(y0 + dy) * 668265263u +
                                               uint32_t(z0 + dz) * 2147483647u);
                const float g = procGrad3(h, fx - float(dx), fy - float(dy), fz - float(dz));
                const float wx = dx ? u : (1.0f - u);
                const float wy = dy ? v : (1.0f - v);
                const float wz = dz ? w : (1.0f - w);
                n += g * wx * wy * wz;
            }
        }
    }
    return 0.866f * n;
}

SR_INL SR_HD Vec3 procPerlin2Vec3(float x, float y) {
    return Vec3(procPerlin2(x, y), procPerlin2(x + 17.1f, y - 9.3f), procPerlin2(x - 5.7f, y + 23.4f));
}

SR_INL SR_HD Vec3 procPerlin3Vec3(float x, float y, float z) {
    return Vec3(procPerlin3(x, y, z), procPerlin3(x + 17.1f, y - 9.3f, z + 3.2f),
                procPerlin3(x - 5.7f, y + 23.4f, z - 11.8f));
}

// MaterialX / OSL cell noise → [0,1] (not signed).
SR_INL SR_HD float procCell2(float x, float y) {
    if (!srIsFinite(x) || !srIsFinite(y)) return 0.0f;
    x = clampf(x, -1.0e5f, 1.0e5f);
    y = clampf(y, -1.0e5f, 1.0e5f);
    const int xi = int(floorf(x));
    const int yi = int(floorf(y));
    return procHashToFloat(procHashU32(uint32_t(xi) * 374761393u + uint32_t(yi) * 668265263u));
}

SR_INL SR_HD float procCell3(float x, float y, float z) {
    if (!srIsFinite(x) || !srIsFinite(y) || !srIsFinite(z)) return 0.0f;
    x = clampf(x, -1.0e5f, 1.0e5f);
    y = clampf(y, -1.0e5f, 1.0e5f);
    z = clampf(z, -1.0e5f, 1.0e5f);
    const int xi = int(floorf(x));
    const int yi = int(floorf(y));
    const int zi = int(floorf(z));
    return procHashToFloat(procHashU32(uint32_t(xi) * 374761393u + uint32_t(yi) * 668265263u +
                                       uint32_t(zi) * 2147483647u));
}

// Worley F1 distance (MaterialX style). jitter in [0,1], style 0=distance 1=solid.
SR_INL SR_HD float procWorley2(float x, float y, float jitter, int style) {
    if (!srIsFinite(x) || !srIsFinite(y)) return 0.0f;
    x = clampf(x, -1.0e5f, 1.0e5f);
    y = clampf(y, -1.0e5f, 1.0e5f);
    jitter = clampf(jitter, 0.0f, 1.0f);
    const int xi = int(floorf(x));
    const int yi = int(floorf(y));
    const float fx = x - float(xi);
    const float fy = y - float(yi);
    float minD = 1.0e10f;
    float solid = 0.0f;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            const uint32_t h =
                procHashU32(uint32_t(xi + ox) * 374761393u + uint32_t(yi + oy) * 668265263u);
            const float px = float(ox) + (procHashToFloat(h) - 0.5f) * jitter + 0.5f;
            const float py = float(oy) + (procHashToFloat(h * 0x85ebca6bu + 1u) - 0.5f) * jitter + 0.5f;
            const float dx = px - fx;
            const float dy = py - fy;
            const float d = dx * dx + dy * dy;
            if (d < minD) {
                minD = d;
                solid = procHashToFloat(h ^ 0x27d4eb2du);
            }
        }
    }
    if (style != 0) return solid;
    return sqrtf(srMax(0.0f, minD));
}

SR_INL SR_HD float procWorley3(float x, float y, float z, float jitter, int style) {
    if (!srIsFinite(x) || !srIsFinite(y) || !srIsFinite(z)) return 0.0f;
    x = clampf(x, -1.0e5f, 1.0e5f);
    y = clampf(y, -1.0e5f, 1.0e5f);
    z = clampf(z, -1.0e5f, 1.0e5f);
    jitter = clampf(jitter, 0.0f, 1.0f);
    const int xi = int(floorf(x));
    const int yi = int(floorf(y));
    const int zi = int(floorf(z));
    const float fx = x - float(xi);
    const float fy = y - float(yi);
    const float fz = z - float(zi);
    float minD = 1.0e10f;
    float solid = 0.0f;
    for (int oz = -1; oz <= 1; ++oz) {
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                const uint32_t h = procHashU32(uint32_t(xi + ox) * 374761393u +
                                              uint32_t(yi + oy) * 668265263u +
                                              uint32_t(zi + oz) * 2147483647u);
                const float px = float(ox) + (procHashToFloat(h) - 0.5f) * jitter + 0.5f;
                const float py =
                    float(oy) + (procHashToFloat(h * 0x85ebca6bu + 1u) - 0.5f) * jitter + 0.5f;
                const float pz =
                    float(oz) + (procHashToFloat(h * 0xc2b2ae3du + 2u) - 0.5f) * jitter + 0.5f;
                const float dx = px - fx;
                const float dy = py - fy;
                const float dz = pz - fz;
                const float d = dx * dx + dy * dy + dz * dz;
                if (d < minD) {
                    minD = d;
                    solid = procHashToFloat(h ^ 0x27d4eb2du);
                }
            }
        }
    }
    if (style != 0) return solid;
    return sqrtf(srMax(0.0f, minD));
}

SR_INL SR_HD float procFractal3(float x, float y, float z, int octaves, float lacunarity, float diminish) {
    float amp = 1.0f;
    float sum = 0.0f;
    float maxAmp = 0.0f;
    float px = x, py = y, pz = z;
    octaves = octaves < 1 ? 1 : (octaves > 16 ? 16 : octaves);
    for (int i = 0; i < octaves; ++i) {
        sum += amp * procPerlin3(px, py, pz);
        maxAmp += amp;
        amp *= diminish;
        px *= lacunarity;
        py *= lacunarity;
        pz *= lacunarity;
    }
    return maxAmp > 0.0f ? sum / maxAmp : 0.0f;
}

SR_INL SR_HD Vec3 procFractal3Vec3(float x, float y, float z, int octaves, float lacunarity, float diminish) {
    return Vec3(procFractal3(x, y, z, octaves, lacunarity, diminish),
                procFractal3(x + 19.1f, y - 7.4f, z + 2.3f, octaves, lacunarity, diminish),
                procFractal3(x - 4.2f, y + 13.8f, z - 9.6f, octaves, lacunarity, diminish));
}

SR_INL SR_HD Vec4 procAsChannels(Vec4 v, int channels) {
    if (channels <= 1) return Vec4(v.x, v.x, v.x, 1.0f);
    if (channels == 2) return Vec4(v.x, v.y, 0.0f, 1.0f);
    if (channels >= 4) return v;
    return Vec4(v.x, v.y, v.z, 1.0f);
}

SR_INL SR_HD Vec4 procSplat(float x) { return Vec4(x, x, x, 1.0f); }

SR_INL SR_HD Vec4 remapSignedColor(Vec4 c) {
    if (c.x < 0.0f || c.y < 0.0f || c.z < 0.0f) {
        c.x = c.x * 0.5f + 0.5f;
        c.y = c.y * 0.5f + 0.5f;
        c.z = c.z * 0.5f + 0.5f;
    }
    return c;
}

// Local bilinear sample so this header does not depend on shading.h.
// Periodic wrap on both U and V (Arnold/Maya texture repeat) — needed for triplanar.
// No lambdas: this header is compiled into OptiX/CUDA device code.
SR_INL SR_HD Vec4 procFetchTexel(const TextureView& tex, int ix, int iy) {
    if (!tex.valid()) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    ix = ((ix % tex.width) + tex.width) % tex.width;
    iy = ((iy % tex.height) + tex.height) % tex.height;
    const size_t idx = (size_t(iy) * size_t(tex.width) + size_t(ix)) * 4;
    return Vec4(tex.pixels[idx + 0], tex.pixels[idx + 1], tex.pixels[idx + 2], tex.pixels[idx + 3]);
}

SR_INL SR_HD Vec4 procSampleTexture(const TextureView& tex, Vec2 uv) {
    if (!tex.valid()) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    // Guard non-finite UVs from bad scale/offset (can crash floor/% on some platforms).
    if (!srIsFinite(uv.x) || !srIsFinite(uv.y)) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    float u = uv.x - floorf(uv.x);
    float v = uv.y - floorf(uv.y);
    if (!srIsFinite(u)) u = 0.0f;
    if (!srIsFinite(v)) v = 0.0f;
    const float x = u * float(tex.width) - 0.5f;
    const float y = v * float(tex.height) - 0.5f;
    const int x0 = int(floorf(x));
    const int y0 = int(floorf(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const Vec4 c00 = procFetchTexel(tex, x0, y0);
    const Vec4 c10 = procFetchTexel(tex, x0 + 1, y0);
    const Vec4 c01 = procFetchTexel(tex, x0, y0 + 1);
    const Vec4 c11 = procFetchTexel(tex, x0 + 1, y0 + 1);
    const Vec4 c0 = Vec4(c00.x * (1.0f - fx) + c10.x * fx, c00.y * (1.0f - fx) + c10.y * fx,
                          c00.z * (1.0f - fx) + c10.z * fx, c00.w * (1.0f - fx) + c10.w * fx);
    const Vec4 c1 = Vec4(c01.x * (1.0f - fx) + c11.x * fx, c01.y * (1.0f - fx) + c11.y * fx,
                          c01.z * (1.0f - fx) + c11.z * fx, c01.w * (1.0f - fx) + c11.w * fx);
    return Vec4(c0.x * (1.0f - fy) + c1.x * fy, c0.y * (1.0f - fy) + c1.y * fy, c0.z * (1.0f - fy) + c1.z * fy,
                c0.w * (1.0f - fy) + c1.w * fy);
}

SR_INL SR_HD Vec4 evalProceduralNode(const SceneView& scene, int index, const ProceduralCtx& ctx, int depth);

SR_INL SR_HD Vec4 evalProceduralChild(const SceneView& scene, int index, const ProceduralCtx& ctx, int depth,
                                      Vec4 fallback) {
    if (index < 0) return fallback;
    return evalProceduralNode(scene, index, ctx, depth + 1);
}

SR_INL SR_HD Vec4 evalProceduralNode(const SceneView& scene, int index, const ProceduralCtx& ctx, int depth) {
    if (!scene.procedurals || index < 0 || index >= scene.proceduralCount || depth > 48)
        return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    const ProceduralNode& n = scene.procedurals[index];
    Vec4 result(0.0f, 0.0f, 0.0f, 1.0f);

    switch (n.op) {
        case kProcConst:
            result = n.p0;
            break;
        case kProcUv:
            result = Vec4(ctx.uv.x, ctx.uv.y, 0.0f, 1.0f);
            break;
        case kProcPosition:
            result = Vec4(ctx.pObject.x, ctx.pObject.y, ctx.pObject.z, 1.0f);
            break;
        case kProcNormal:
            result = Vec4(ctx.nObject.x, ctx.nObject.y, ctx.nObject.z, 1.0f);
            break;
        case kProcNoise2d: {
            Vec2 tc = ctx.uv;
            if (n.in0 >= 0) {
                const Vec4 t = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(tc.x, tc.y, 0.0f, 1.0f));
                tc = Vec2(t.x, t.y);
            }
            const float sx = tc.x * 8.0f;
            const float sy = tc.y * 8.0f;
            const float pivot = n.p1.x;
            if (n.channels <= 1) {
                result = procSplat(procPerlin2(sx, sy) * n.p0.x + pivot);
            } else {
                const Vec3 noise = procPerlin2Vec3(sx, sy);
                result = Vec4(noise.x * n.p0.x + pivot, noise.y * n.p0.y + pivot, noise.z * n.p0.z + pivot, 1.0f);
            }
            break;
        }
        case kProcNoise3d: {
            Vec3 pos = ctx.pObject;
            if (n.in0 >= 0) {
                const Vec4 p = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(pos.x, pos.y, pos.z, 1.0f));
                pos = Vec3(p.x, p.y, p.z);
            } else {
                // No authored position wire → object P; fall back to UV plane if P is degenerate.
                if (lengthSquared(pos) < 1e-20f) pos = Vec3(ctx.uv.x * 8.0f, ctx.uv.y * 8.0f, 0.0f);
            }
            if (!srIsFinite(pos.x) || !srIsFinite(pos.y) || !srIsFinite(pos.z))
                pos = Vec3(ctx.uv.x * 8.0f, ctx.uv.y * 8.0f, 0.0f);
            const float pivot = n.p1.x;
            if (n.channels <= 1) {
                result = procSplat(procPerlin3(pos.x, pos.y, pos.z) * n.p0.x + pivot);
            } else {
                const Vec3 noise = procPerlin3Vec3(pos.x, pos.y, pos.z);
                result = Vec4(noise.x * n.p0.x + pivot, noise.y * n.p0.y + pivot, noise.z * n.p0.z + pivot, 1.0f);
            }
            break;
        }
        case kProcFractal: {
            Vec3 pos = ctx.pObject;
            if (n.in0 >= 0) {
                const Vec4 p = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(pos.x, pos.y, pos.z, 1.0f));
                pos = Vec3(p.x, p.y, p.z);
            } else if (lengthSquared(pos) < 1e-20f) {
                pos = Vec3(ctx.uv.x * 4.0f, ctx.uv.y * 4.0f, 0.0f);
            }
            if (!srIsFinite(pos.x) || !srIsFinite(pos.y) || !srIsFinite(pos.z))
                pos = Vec3(ctx.uv.x * 4.0f, ctx.uv.y * 4.0f, 0.0f);
            const int octaves = int(floorf(n.s0 + 0.5f));
            float lac = n.s1;
            float dim = n.s2;
            if (!srIsFinite(lac) || lac < 1.0e-3f) lac = 2.0f;
            if (!srIsFinite(dim)) dim = 0.5f;
            if (n.channels <= 1) {
                result = procSplat(procFractal3(pos.x, pos.y, pos.z, octaves, lac, dim) * n.p0.x);
            } else {
                const Vec3 noise = procFractal3Vec3(pos.x, pos.y, pos.z, octaves, lac, dim);
                result = Vec4(noise.x * n.p0.x, noise.y * n.p0.y, noise.z * n.p0.z, 1.0f);
            }
            break;
        }
        case kProcCell2d: {
            Vec2 tc = ctx.uv;
            if (n.in0 >= 0) {
                const Vec4 t = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(ctx.uv.x, ctx.uv.y, 0.0f, 1.0f));
                tc = Vec2(t.x, t.y);
            }
            if (!srIsFinite(tc.x) || !srIsFinite(tc.y)) tc = ctx.uv;
            // MaterialX cellnoise uses raw texcoord (no baked *8); keep mild default scale
            // only when the port is unbound so artists still see structure on [0,1] UVs.
            if (n.in0 < 0) tc = tc * 8.0f;
            const float jitter = n.s0 > 0.0f ? n.s0 : 1.0f;
            const int style = int(floorf(n.s1 + 0.5f));
            if (n.s2 > 0.5f)
                result = procSplat(procWorley2(tc.x, tc.y, jitter, style));
            else
                result = procSplat(procCell2(tc.x, tc.y));
            break;
        }
        case kProcCell3d: {
            Vec3 pos = ctx.pObject;
            if (n.in0 >= 0) {
                const Vec4 p = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(pos.x, pos.y, pos.z, 1.0f));
                pos = Vec3(p.x, p.y, p.z);
            } else if (lengthSquared(pos) < 1e-20f) {
                pos = Vec3(ctx.uv.x * 8.0f, ctx.uv.y * 8.0f, 0.0f);
            }
            if (!srIsFinite(pos.x) || !srIsFinite(pos.y) || !srIsFinite(pos.z))
                pos = Vec3(ctx.uv.x * 8.0f, ctx.uv.y * 8.0f, 0.0f);
            const float jitter = n.s0 > 0.0f ? n.s0 : 1.0f;
            const int style = int(floorf(n.s1 + 0.5f));
            if (n.s2 > 0.5f)
                result = procSplat(procWorley3(pos.x, pos.y, pos.z, jitter, style));
            else
                result = procSplat(procCell3(pos.x, pos.y, pos.z));
            break;
        }
        case kProcUnified2d: {
            Vec2 tc = ctx.uv;
            if (n.in0 >= 0) {
                const Vec4 t = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(tc.x, tc.y, 0.0f, 1.0f));
                tc = Vec2(t.x, t.y);
            }
            if (!srIsFinite(tc.x) || !srIsFinite(tc.y)) tc = ctx.uv;
            // Frequency is vector2 — each axis scales independently (MaterialX / Arnold).
            float freqX = n.p0.x;
            float freqY = n.p0.y;
            if (!srIsFinite(freqX)) freqX = 1.0f;
            if (!srIsFinite(freqY)) freqY = freqX;
            float ox = srIsFinite(n.p1.x) ? n.p1.x : 0.0f;
            float oy = srIsFinite(n.p1.y) ? n.p1.y : 0.0f;
            float x = tc.x * freqX + ox;
            float y = tc.y * freqY + oy;
            const float jitter = clampf(srIsFinite(n.p1.w) ? n.p1.w : 1.0f, 0.0f, 1.0f);
            // MaterialX applies a huge rotation when jitter != 1 for non-Worley types.
            if (fabsf(jitter - 1.0f) > 1.0e-4f) {
                const float amount = (jitter - 1.0f) * 90000.0f * 0.017453292519943295f;
                const float ca = cosf(amount);
                const float sa = sinf(amount);
                const float rx = x * ca - y * sa;
                const float ry = x * sa + y * ca;
                x = rx;
                y = ry;
            }
            const int type = int(floorf(n.p2.z + 0.5f));
            const int style = int(floorf(n.p2.w + 0.5f));
            const int octaves = int(floorf(n.s0 + 0.5f));
            float lac = n.s1;
            float dim = n.s2;
            if (!srIsFinite(lac) || lac < 1.0e-3f) lac = 2.0f;
            if (!srIsFinite(dim)) dim = 0.5f;
            float noise = 0.0f;
            // 0 Perlin, 1 Cell, 2 Worley, 3 Fractal — octaves apply to Fractal (and
            // Arnold-like detail when Perlin asks for more than one octave).
            if (type == 1)
                noise = procCell2(x, y);
            else if (type == 2)
                noise = procWorley2(x, y, jitter, style);
            else if (type == 3 || (type == 0 && octaves > 1))
                noise = procFractal3(x, y, 0.0f, octaves, lac, dim);
            else
                noise = procPerlin2(x, y);
            // Cell/Worley are already [0,1]; Perlin/Fractal are signed ≈[-1,1].
            float t = (type == 1 || type == 2) ? noise : (noise * 0.5f + 0.5f);
            if (!srIsFinite(t)) t = 0.0f;
            float outMin = srIsFinite(n.p2.x) ? n.p2.x : 0.0f;
            float outMax = srIsFinite(n.p2.y) ? n.p2.y : 1.0f;
            float out = outMin * (1.0f - t) + outMax * t;
            if (n.s3 > 0.5f) out = clampf(out, outMin < outMax ? outMin : outMax, outMin < outMax ? outMax : outMin);
            result = procSplat(out);
            break;
        }
        case kProcUnified3d: {
            Vec3 pos = ctx.pObject;
            if (n.in0 >= 0) {
                const Vec4 p = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(pos.x, pos.y, pos.z, 1.0f));
                pos = Vec3(p.x, p.y, p.z);
            } else if (lengthSquared(pos) < 1e-20f) {
                pos = Vec3(ctx.uv.x, ctx.uv.y, 0.0f);
            }
            if (!srIsFinite(pos.x) || !srIsFinite(pos.y) || !srIsFinite(pos.z))
                pos = Vec3(ctx.uv.x, ctx.uv.y, 0.0f);
            float freqX = n.p0.x;
            float freqY = n.p0.y;
            float freqZ = n.p0.z;
            if (!srIsFinite(freqX)) freqX = 1.0f;
            if (!srIsFinite(freqY)) freqY = freqX;
            if (!srIsFinite(freqZ)) freqZ = freqX;
            float ox = srIsFinite(n.p1.x) ? n.p1.x : 0.0f;
            float oy = srIsFinite(n.p1.y) ? n.p1.y : 0.0f;
            float oz = srIsFinite(n.p1.z) ? n.p1.z : 0.0f;
            float x = pos.x * freqX + ox;
            float y = pos.y * freqY + oy;
            float z = pos.z * freqZ + oz;
            const float jitter = clampf(srIsFinite(n.p1.w) ? n.p1.w : 1.0f, 0.0f, 1.0f);
            if (fabsf(jitter - 1.0f) > 1.0e-4f) {
                // Approximate MaterialX rotate3d about (0.1,1,0).
                const float amount = (jitter - 1.0f) * 90000.0f * 0.017453292519943295f;
                const float ca = cosf(amount);
                const float sa = sinf(amount);
                // Rotate around Y-ish axis.
                const float rx = x * ca + z * sa;
                const float rz = -x * sa + z * ca;
                x = rx;
                z = rz;
            }
            const int type = int(floorf(n.p2.z + 0.5f));
            const int style = int(floorf(n.p2.w + 0.5f));
            const int octaves = int(floorf(n.s0 + 0.5f));
            float lac = n.s1;
            float dim = n.s2;
            if (!srIsFinite(lac) || lac < 1.0e-3f) lac = 2.0f;
            if (!srIsFinite(dim)) dim = 0.5f;
            float noise = 0.0f;
            if (type == 1)
                noise = procCell3(x, y, z);
            else if (type == 2)
                noise = procWorley3(x, y, z, jitter, style);
            else if (type == 3 || (type == 0 && octaves > 1))
                noise = procFractal3(x, y, z, octaves, lac, dim);
            else
                noise = procPerlin3(x, y, z);
            float t = (type == 1 || type == 2) ? noise : (noise * 0.5f + 0.5f);
            if (!srIsFinite(t)) t = 0.0f;
            float outMin = srIsFinite(n.p2.x) ? n.p2.x : 0.0f;
            float outMax = srIsFinite(n.p2.y) ? n.p2.y : 1.0f;
            float out = outMin * (1.0f - t) + outMax * t;
            if (n.s3 > 0.5f) out = clampf(out, outMin < outMax ? outMin : outMax, outMin < outMax ? outMax : outMin);
            if (!srIsFinite(out)) out = outMin;
            result = procSplat(out);
            break;
        }
        case kProcImage: {
            Vec2 uv = ctx.uv;
            if (n.in1 >= 0) {
                const Vec4 t = evalProceduralChild(scene, n.in1, ctx, depth, Vec4(uv.x, uv.y, 0.0f, 1.0f));
                uv = Vec2(t.x, t.y);
            }
            if (n.in0 >= 0 && n.in0 < scene.textureCount && scene.textures) {
                result = procSampleTexture(scene.textures[n.in0], uv);
            } else {
                result = n.p0;
            }
            break;
        }
        case kProcTriplanar: {
            // Arnold-style object-space triplanar: UV = (P + offset) / scale, optional rotate.
            // p0=default, p1=scale, p2=offset, s0=blend, s1=rotate degrees.
            // No lambdas — OptiX/CUDA device compile.
            float nx = fabsf(ctx.nObject.x);
            float ny = fabsf(ctx.nObject.y);
            float nz = fabsf(ctx.nObject.z);
            if (!srIsFinite(nx)) nx = 0.0f;
            if (!srIsFinite(ny)) ny = 0.0f;
            if (!srIsFinite(nz)) nz = 0.0f;
            float blend = n.s0;
            if (!srIsFinite(blend) || blend < 0.01f) blend = 0.01f;
            if (blend > 64.0f) blend = 64.0f;
            nx = powf(nx, blend);
            ny = powf(ny, blend);
            nz = powf(nz, blend);
            const float sum = nx + ny + nz;
            if (sum > 0.0f) {
                const float inv = 1.0f / sum;
                nx *= inv;
                ny *= inv;
                nz *= inv;
            } else {
                nx = 0.0f;
                ny = 1.0f;
                nz = 0.0f;
            }
            float sx = n.p1.x;
            float sy = n.p1.y;
            float sz = n.p1.z;
            if (!srIsFinite(sx) || fabsf(sx) < 1.0e-5f) sx = 1.0f;
            if (!srIsFinite(sy) || fabsf(sy) < 1.0e-5f) sy = sx;
            if (!srIsFinite(sz) || fabsf(sz) < 1.0e-5f) sz = sx;
            sx = fabsf(sx);
            sy = fabsf(sy);
            sz = fabsf(sz);
            float ox = srIsFinite(n.p2.x) ? n.p2.x : 0.0f;
            float oy = srIsFinite(n.p2.y) ? n.p2.y : 0.0f;
            float oz = srIsFinite(n.p2.z) ? n.p2.z : 0.0f;
            const float px = (ctx.pObject.x + ox) / sx;
            const float py = (ctx.pObject.y + oy) / sy;
            const float pz = (ctx.pObject.z + oz) / sz;
            float rotDeg = n.s1;
            if (!srIsFinite(rotDeg)) rotDeg = 0.0f;
            // Keep rotation in a sane range so sin/cos stay finite.
            rotDeg = fmodf(rotDeg, 360.0f);
            const float rotRad = rotDeg * 0.017453292519943295f;
            const float cr = cosf(rotRad);
            const float sn = sinf(rotRad);
            const float ux = pz * cr - py * sn;
            const float vx = pz * sn + py * cr;
            const float uy = px * cr - pz * sn;
            const float vy = px * sn + pz * cr;
            const float uz = px * cr - py * sn;
            const float vz = px * sn + py * cr;
            Vec4 cx = n.p0;
            Vec4 cy = n.p0;
            Vec4 cz = n.p0;
            if (n.in0 >= 0 && n.in0 < scene.textureCount && scene.textures)
                cx = procSampleTexture(scene.textures[n.in0], Vec2(ux, vx));
            if (n.in1 >= 0 && n.in1 < scene.textureCount && scene.textures)
                cy = procSampleTexture(scene.textures[n.in1], Vec2(uy, vy));
            if (n.in2 >= 0 && n.in2 < scene.textureCount && scene.textures)
                cz = procSampleTexture(scene.textures[n.in2], Vec2(uz, vz));
            result = Vec4(cx.x * nx + cy.x * ny + cz.x * nz, cx.y * nx + cy.y * ny + cz.y * nz,
                          cx.z * nx + cy.z * ny + cz.z * nz, 1.0f);
            if (!srIsFinite(result.x) || !srIsFinite(result.y) || !srIsFinite(result.z)) result = n.p0;
            break;
        }
        case kProcMul: {
            const Vec4 a = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
            const Vec4 b = evalProceduralChild(scene, n.in1, ctx, depth, n.p0);
            result = Vec4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
            break;
        }
        case kProcAdd: {
            const Vec4 a = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            const Vec4 b = evalProceduralChild(scene, n.in1, ctx, depth, n.p0);
            result = Vec4(a.x + b.x, a.y + b.y, a.z + b.z, 1.0f);
            break;
        }
        case kProcSub: {
            const Vec4 a = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            const Vec4 b = evalProceduralChild(scene, n.in1, ctx, depth, n.p0);
            result = Vec4(a.x - b.x, a.y - b.y, a.z - b.z, 1.0f);
            break;
        }
        case kProcDiv: {
            const Vec4 a = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            const Vec4 b = evalProceduralChild(scene, n.in1, ctx, depth, n.p0);
            result = Vec4(b.x != 0.0f ? a.x / b.x : 0.0f, b.y != 0.0f ? a.y / b.y : 0.0f,
                          b.z != 0.0f ? a.z / b.z : 0.0f, 1.0f);
            break;
        }
        case kProcMix: {
            const Vec4 bg = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            const Vec4 fg = evalProceduralChild(scene, n.in1, ctx, depth, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
            float m = n.p0.x;
            if (n.in2 >= 0) m = evalProceduralChild(scene, n.in2, ctx, depth, procSplat(m)).x;
            m = clampf(m, 0.0f, 1.0f);
            result = Vec4(bg.x * (1.0f - m) + fg.x * m, bg.y * (1.0f - m) + fg.y * m,
                          bg.z * (1.0f - m) + fg.z * m, 1.0f);
            break;
        }
        case kProcClamp: {
            const Vec4 in = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            result = Vec4(clampf(in.x, n.s1, n.s2), clampf(in.y, n.s1, n.s2), clampf(in.z, n.s1, n.s2), 1.0f);
            break;
        }
        case kProcSaturate: {
            const Vec4 in = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            result = Vec4(clampf(in.x, 0.0f, 1.0f), clampf(in.y, 0.0f, 1.0f), clampf(in.z, 0.0f, 1.0f), 1.0f);
            break;
        }
        case kProcInvert: {
            const Vec4 in = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            const float amount = n.s0;
            result = Vec4(in.x * (1.0f - amount) + (1.0f - in.x) * amount,
                          in.y * (1.0f - amount) + (1.0f - in.y) * amount,
                          in.z * (1.0f - amount) + (1.0f - in.z) * amount, 1.0f);
            break;
        }
        case kProcAbs: {
            const Vec4 in = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            result = Vec4(fabsf(in.x), fabsf(in.y), fabsf(in.z), 1.0f);
            break;
        }
        case kProcPower: {
            const Vec4 in = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            const float expv = n.s0;
            result = Vec4(powf(srMax(0.0f, in.x), expv), powf(srMax(0.0f, in.y), expv),
                          powf(srMax(0.0f, in.z), expv), 1.0f);
            break;
        }
        case kProcCombine: {
            float x = n.p0.x, y = n.p0.y, z = n.p0.z, w = n.p0.w;
            if (n.in0 >= 0) x = evalProceduralChild(scene, n.in0, ctx, depth, procSplat(x)).x;
            if (n.in1 >= 0) y = evalProceduralChild(scene, n.in1, ctx, depth, procSplat(y)).x;
            if (n.in2 >= 0) z = evalProceduralChild(scene, n.in2, ctx, depth, procSplat(z)).x;
            if (n.in3 >= 0) w = evalProceduralChild(scene, n.in3, ctx, depth, procSplat(w)).x;
            result = Vec4(x, y, z, w);
            break;
        }
        case kProcExtract: {
            const Vec4 in = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            const int idx = int(floorf(n.s0 + 0.5f));
            const float ch = idx <= 0 ? in.x : (idx == 1 ? in.y : (idx == 2 ? in.z : in.w));
            result = procSplat(ch);
            break;
        }
        case kProcRampLR: {
            const Vec4 a = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            const Vec4 b = evalProceduralChild(scene, n.in1, ctx, depth, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
            const float t = clampf(ctx.uv.x, 0.0f, 1.0f);
            result = Vec4(a.x * (1.0f - t) + b.x * t, a.y * (1.0f - t) + b.y * t, a.z * (1.0f - t) + b.z * t, 1.0f);
            break;
        }
        case kProcRampTB: {
            const Vec4 a = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            const Vec4 b = evalProceduralChild(scene, n.in1, ctx, depth, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
            const float t = clampf(ctx.uv.y, 0.0f, 1.0f);
            result = Vec4(a.x * (1.0f - t) + b.x * t, a.y * (1.0f - t) + b.y * t, a.z * (1.0f - t) + b.z * t, 1.0f);
            break;
        }
        case kProcChecker: {
            const float xf = floorf(ctx.uv.x * 8.0f);
            const float yf = floorf(ctx.uv.y * 8.0f);
            const bool on = (int(xf) + int(yf)) & 1;
            result = on ? Vec4(1.0f, 1.0f, 1.0f, 1.0f) : Vec4(0.0f, 0.0f, 0.0f, 1.0f);
            break;
        }
        default:
            result = Vec4(0.5f, 0.5f, 0.5f, 1.0f);
            break;
    }

    return procAsChannels(result, n.channels);
}

SR_INL SR_HD Vec4 evalProceduralRoot(const SceneView& scene, int root, const ProceduralCtx& ctx) {
    if (root < 0) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return remapSignedColor(evalProceduralNode(scene, root, ctx, 0));
}

}  // namespace sol
