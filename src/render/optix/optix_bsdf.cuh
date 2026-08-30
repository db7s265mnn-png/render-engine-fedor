// GPU surface maps for OptiX. BSDF lobes come from render/shading_bsdf.h —
// the same dielectric as Embree (Snell, 1/η², anisotropy, sheen, coat).
//
// This file still must NOT include render/shading.h: that pulls MaterialX
// procedurals into the shade megakernel. Maps are bilinear 2D images in
// render/surface_maps.h (shared with CPU Exit to Diffuse dest).
#pragma once

#include "render/shading_bsdf.h"
#include "render/surface_maps.h"
#include "scene/types.h"

namespace sol {
namespace optixpt {

using ::sol::BsdfSample;
using ::sol::BsdfEval;
using ::sol::LobeWeights;
using ::sol::evaluateSurfaceMaps;
using ::sol::fetchTexelWrap;
using ::sol::sampleMapRgb;
using ::sol::sampleMapScalar;
using ::sol::sampleTextureWrap;

SR_INL SR_HD Material evaluateMaps(const SceneView& scene, const Material& base, Vec2 uv, Vec3& ns) {
    return evaluateSurfaceMaps(scene, base, uv, ns);
}

}  // namespace optixpt
}  // namespace sol
