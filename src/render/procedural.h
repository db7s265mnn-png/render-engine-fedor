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

SR_INL SR_HD float procCell2(float x, float y) {
    if (!srIsFinite(x) || !srIsFinite(y)) return 0.0f;
    x = clampf(x, -1.0e5f, 1.0e5f);
    y = clampf(y, -1.0e5f, 1.0e5f);
    const int xi = int(floorf(x));
    const int yi = int(floorf(y));
    return procHashToFloat(procHashU32(uint32_t(xi) * 374761393u + uint32_t(yi) * 668265263u)) * 2.0f - 1.0f;
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
                                       uint32_t(zi) * 2147483647u)) *
               2.0f -
           1.0f;
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
SR_INL SR_HD Vec4 procSampleTexture(const TextureView& tex, Vec2 uv) {
    if (!tex.valid()) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    float u = uv.x - floorf(uv.x);
    float v = uv.y - floorf(uv.y);
    const float x = u * float(tex.width) - 0.5f;
    const float y = v * float(tex.height) - 0.5f;
    const int x0 = int(floorf(x));
    const int y0 = int(floorf(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    auto fetch = [&](int ix, int iy) -> Vec4 {
        ix = ((ix % tex.width) + tex.width) % tex.width;
        iy = ((iy % tex.height) + tex.height) % tex.height;
        const size_t idx = (size_t(iy) * size_t(tex.width) + size_t(ix)) * 4;
        return Vec4(tex.pixels[idx + 0], tex.pixels[idx + 1], tex.pixels[idx + 2], tex.pixels[idx + 3]);
    };
    const Vec4 c00 = fetch(x0, y0);
    const Vec4 c10 = fetch(x0 + 1, y0);
    const Vec4 c01 = fetch(x0, y0 + 1);
    const Vec4 c11 = fetch(x0 + 1, y0 + 1);
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
            const int octaves = int(floorf(n.s0 + 0.5f));
            if (n.channels <= 1) {
                result = procSplat(procFractal3(pos.x, pos.y, pos.z, octaves, n.s1, n.s2) * n.p0.x);
            } else {
                const Vec3 noise = procFractal3Vec3(pos.x, pos.y, pos.z, octaves, n.s1, n.s2);
                result = Vec4(noise.x * n.p0.x, noise.y * n.p0.y, noise.z * n.p0.z, 1.0f);
            }
            break;
        }
        case kProcCell2d: {
            Vec2 tc = ctx.uv * 8.0f;
            if (n.in0 >= 0) {
                const Vec4 t = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(ctx.uv.x, ctx.uv.y, 0.0f, 1.0f));
                tc = Vec2(t.x * 8.0f, t.y * 8.0f);
            }
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
            result = procSplat(procCell3(pos.x, pos.y, pos.z));
            break;
        }
        case kProcUnified2d: {
            Vec2 tc = ctx.uv;
            if (n.in0 >= 0) {
                const Vec4 t = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(tc.x, tc.y, 0.0f, 1.0f));
                tc = Vec2(t.x, t.y);
            }
            const Vec2 freq(n.p0.x, n.p0.y);
            const Vec2 offset(n.p1.x, n.p1.y);
            const float x = tc.x * freq.x + offset.x;
            const float y = tc.y * freq.y + offset.y;
            const int type = int(floorf(n.p2.z + 0.5f));
            const int octaves = int(floorf(n.s0 + 0.5f));
            float noise = 0.0f;
            if (type == 1)
                noise = procCell2(x, y);
            else if (type == 2)
                noise = procCell2(x, y);  // worley stand-in
            else if (type == 3)
                noise = procFractal3(x, y, 0.0f, octaves, n.s1, n.s2);
            else
                noise = procPerlin2(x, y);
            // Map signed noise into [outmin, outmax].
            float t = noise * 0.5f + 0.5f;
            float out = n.p2.x * (1.0f - t) + n.p2.y * t;
            if (n.s3 > 0.5f) out = clampf(out, n.p2.x, n.p2.y);
            result = procSplat(out);
            break;
        }
        case kProcUnified3d: {
            Vec3 pos = ctx.pObject;
            if (n.in0 >= 0) {
                const Vec4 p = evalProceduralChild(scene, n.in0, ctx, depth, Vec4(pos.x, pos.y, pos.z, 1.0f));
                pos = Vec3(p.x, p.y, p.z);
            }
            const Vec3 freq(n.p0.x, n.p0.y, n.p0.z);
            const Vec3 offset(n.p1.x, n.p1.y, n.p1.z);
            const float x = pos.x * freq.x + offset.x;
            const float y = pos.y * freq.y + offset.y;
            const float z = pos.z * freq.z + offset.z;
            const int type = int(floorf(n.p2.z + 0.5f));
            const int octaves = int(floorf(n.s0 + 0.5f));
            float noise = 0.0f;
            if (type == 1)
                noise = procCell3(x, y, z);
            else if (type == 2)
                noise = procCell3(x, y, z);
            else if (type == 3)
                noise = procFractal3(x, y, z, octaves, n.s1, n.s2);
            else
                noise = procPerlin3(x, y, z);
            float t = noise * 0.5f + 0.5f;
            float out = n.p2.x * (1.0f - t) + n.p2.y * t;
            if (n.s3 > 0.5f) out = clampf(out, n.p2.x, n.p2.y);
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
            Vec3 nAbs(fabsf(ctx.nObject.x), fabsf(ctx.nObject.y), fabsf(ctx.nObject.z));
            const float blend = srMax(0.01f, n.s0);
            nAbs = Vec3(powf(nAbs.x, blend), powf(nAbs.y, blend), powf(nAbs.z, blend));
            const float sum = nAbs.x + nAbs.y + nAbs.z;
            if (sum > 0.0f) nAbs = nAbs * (1.0f / sum);
            else nAbs = Vec3(0.0f, 1.0f, 0.0f);
            const float sx = srMax(1.0e-5f, fabsf(n.p1.x) > 0.0f ? n.p1.x : 1.0f);
            const float sy = srMax(1.0e-5f, fabsf(n.p1.y) > 0.0f ? n.p1.y : sx);
            const float sz = srMax(1.0e-5f, fabsf(n.p1.z) > 0.0f ? n.p1.z : sx);
            const Vec3 p((ctx.pObject.x + n.p2.x) / sx, (ctx.pObject.y + n.p2.y) / sy,
                         (ctx.pObject.z + n.p2.z) / sz);
            const float rotRad = n.s1 * 0.017453292519943295f;
            const float cr = cosf(rotRad);
            const float sn = sinf(rotRad);
            auto rotUv = [&](float u, float v) -> Vec2 {
                return Vec2(u * cr - v * sn, u * sn + v * cr);
            };
            Vec4 cx = n.p0;
            Vec4 cy = n.p0;
            Vec4 cz = n.p0;
            if (n.in0 >= 0 && n.in0 < scene.textureCount && scene.textures)
                cx = procSampleTexture(scene.textures[n.in0], rotUv(p.z, p.y));
            if (n.in1 >= 0 && n.in1 < scene.textureCount && scene.textures)
                cy = procSampleTexture(scene.textures[n.in1], rotUv(p.x, p.z));
            if (n.in2 >= 0 && n.in2 < scene.textureCount && scene.textures)
                cz = procSampleTexture(scene.textures[n.in2], rotUv(p.x, p.y));
            result = Vec4(cx.x * nAbs.x + cy.x * nAbs.y + cz.x * nAbs.z,
                          cx.y * nAbs.x + cy.y * nAbs.y + cz.y * nAbs.z,
                          cx.z * nAbs.x + cy.z * nAbs.y + cz.z * nAbs.z, 1.0f);
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
