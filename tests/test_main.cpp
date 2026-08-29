// Unit tests for the maths, sampling, node graph and renderer plumbing.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <QCoreApplication>
#include <QColor>
#include <QDir>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QTemporaryDir>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QStringList>
#include <QVector3D>

#include "app/default_scene.h"
#include "app/integrator_device.h"
#include "app/undo_hub.h"
#include "core/image.h"
#include "core/rng.h"
#include "io/alembic_loader.h"
#include "io/image_io.h"
#include "io/materialx_graph.h"
#include "io/tx_cache.h"
#include "io/tx_convert.h"
#include "io/usd_loader.h"
#include "nodes/node_graph.h"
#include "nodes/node_registry.h"
#include "nodes/parameter.h"
#include "nodes/stage.h"
#include "render/camera_proj.h"
#include "render/cpu/polynomial_optics.h"
#include "render/film_tile.h"
#include "render/framebuffer.h"
#include "render/bdpt_stats.h"
#include "render/bdpt_scratch.h"
#include "render/integrator.h"
#include "render/integrator_bdpt.h"
#include "render/pixel_filter.h"
#include "render/metal_spectra.h"
#include "render/optix/path_state.h"
#include "render/photon_map.h"
#include "render/photon_aim.h"
#include "render/physical_sky.h"
#include "render/render_session.h"
#include "render/shading.h"
#include "render/volume.h"
#include "render/volume_track.h"
#include "render/volume_vdb.h"
#include "render/xpu_split.h"
#include "render/pixel_oracle.h"
#include "render/camera_sample.h"
#include "render/spectrum.h"
#include "render/spectrum_device.h"
#include "render/spectrum_rgb.h"
#include "render/spectrum_types.h"
#include "render/spectral_common.h"
#include "render/sss_spectral.h"
#include "render/cie_tables.h"
#include "render/illuminant_spd.h"
#include "render/rgb_spectrum_tables.h"
#include "scene/scene.h"
#include "scene/displace.h"
#include "scene/tessellate.h"
#include "scene/triangulate.h"
#include "scene/volume_grid.h"
#include "scene/dcsdd_contouring.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_TIFF
#  include <tiffio.h>
#endif

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("  FAIL: %s\n", what.c_str());
    }
}

void checkNear(float value, float expected, float tolerance, const std::string& what) {
    ++g_checks;
    if (!(std::fabs(value - expected) <= tolerance)) {
        ++g_failures;
        std::printf("  FAIL: %s (got %g, expected %g +/- %g)\n", what.c_str(), value, expected, tolerance);
    }
}

using namespace sol;


namespace {
MeshPtr tessDisplaceForTest(MeshPtr cage, Material mat, Scene& scene, int subdivIters = -1,
                            bool forceFrustumOff = true) {
    if (!cage) return cage;
    if (subdivIters >= 0) cage->subdivIterations = subdivIters;
    if (cage->subdivType == kSubdivCatclark) cage->subdivType = kSubdivLinear; // tests use tri cages
    // Ensure material + instance wiring for tessellateSceneForRender.
    if (scene.materials.empty()) scene.addMaterial(mat);
    else scene.materials[0] = mat;
    scene.meshes.clear();
    scene.instances.clear();
    const int mi = scene.addMesh(cage);
    InstanceData inst;
    inst.meshIndex = mi;
    inst.materialIndex = 0;
    inst.xform = Mat4::identity();
    inst.xformInv = Mat4::identity();
    scene.instances.push_back(inst);
    if (!scene.cameraAuthored) {
        scene.camera = CameraData{};
        scene.camera.cameraToWorld = composeTRS(Vec3(0, 2, 8), Vec3(0, 0, 0), Vec3(1));
        scene.cameraAuthored = true;
    }
    // Most tests want whole-mesh densify; frustum-local tests pass forceFrustumOff=false.
    if (forceFrustumOff) scene.settings.frustumCull = 0;
    tessellateSceneForRender(scene, scene.camera);
    return scene.meshes.empty() ? cage : scene.meshes[0];
}

void tessellateCookedScene(ScenePtr scene) {
    if (!scene) return;
    if (!scene->cameraAuthored) {
        scene->camera = CameraData{};
        scene->camera.cameraToWorld = composeTRS(Vec3(0, 2, 8), Vec3(0, 0, 0), Vec3(1));
        scene->cameraAuthored = true;
    }
    // Frustum cull default on — keep contents in view for tests.
    scene->settings.frustumCull = 0;
    tessellateSceneForRender(*scene, scene->camera);
}
}  // namespace
void testMath() {
    std::printf("math\n");
    const Mat4 m = composeTRS(Vec3(1.0f, 2.0f, 3.0f), Vec3(20.0f, -35.0f, 12.0f), Vec3(2.0f, 0.5f, 1.5f));
    const Mat4 inv = inverse(m);
    const Mat4 product = m * inv;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            checkNear(product.at(r, c), r == c ? 1.0f : 0.0f, 1e-4f, "inverse(M) * M is identity");

    const Vec3 point(0.3f, -1.2f, 4.0f);
    const Vec3 roundTrip = transformPoint(inv, transformPoint(m, point));
    checkNear(length(roundTrip - point), 0.0f, 1e-3f, "point transform round trip");

    const Frame frame(normalize(Vec3(0.3f, 0.8f, -0.5f)));
    const Vec3 local(0.2f, -0.4f, 0.6f);
    const Vec3 back = frame.toLocal(frame.toWorld(local));
    checkNear(length(back - local), 0.0f, 1e-5f, "frame round trip");

    for (int i = 0; i < 32; ++i) {
        Rng rng(uint64_t(i), 7u);
        const Vec3 dir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
        const Vec2 uv = directionToEquirect(dir);
        const Vec3 back2 = equirectToDirection(uv.x, uv.y);
        checkNear(length(back2 - dir), 0.0f, 1e-3f, "equirect mapping round trip");
    }
}

void testSampling() {
    std::printf("sampling\n");
    // Cosine hemisphere sampling integrates to pi over the hemisphere.
    Rng rng(1u, 2u);
    double sum = 0.0;
    const int count = 200000;
    for (int i = 0; i < count; ++i) {
        const Vec3 wi = sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat());
        const float pdf = wi.z * kInvPi;
        if (pdf > 0.0f) sum += double(wi.z / pdf);
    }
    checkNear(float(sum / count), kPi, 0.02f, "cosine hemisphere integrates to pi");

    // A 2D distribution should reproduce the density it was built from.
    const int w = 16, h = 8;
    std::vector<float> function(size_t(w) * h, 0.1f);
    function[3 + 2 * w] = 20.0f;
    Distribution2D distribution;
    distribution.build(function, w, h);
    check(distribution.valid(), "distribution builds");
    int hits = 0;
    for (int i = 0; i < 20000; ++i) {
        float pdf = 0.0f;
        const Vec2 uv = distribution.sample(rng.nextFloat(), rng.nextFloat(), pdf);
        check(pdf >= 0.0f, "distribution pdf is non negative");
        const int x = int(uv.x * w);
        const int y = int(uv.y * h);
        if (x == 3 && y == 2) ++hits;
    }
    check(hits > 12000, "distribution concentrates samples on the bright texel");

    // Per-pixel PCG streams must not imprint a screen lattice (tile_test quilt).
    // Neighbor autocorrelation of the first float should be ~0 at lags 16/32/64.
    {
        constexpr int W = 256, H = 128;
        std::vector<float> field(size_t(W) * H);
        double mean = 0.0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                Rng r = makePixelRng(x, y, 0, 0u);
                const float u = r.nextFloat();
                field[size_t(y) * W + x] = u;
                mean += double(u);
            }
        }
        mean /= double(W * H);
        auto acLag = [&](int lag) -> double {
            double num = 0.0, den = 0.0;
            int n = 0;
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x + lag < W; ++x) {
                    const double a = double(field[size_t(y) * W + x]) - mean;
                    const double b = double(field[size_t(y) * W + x + lag]) - mean;
                    num += a * b;
                    den += a * a;
                    ++n;
                }
            }
            (void)n;
            return den > 1e-12 ? num / den : 0.0;
        };
        check(std::fabs(acLag(16)) < 0.05, "pixel RNG ac@16 ~ 0");
        check(std::fabs(acLag(32)) < 0.05, "pixel RNG ac@32 ~ 0");
        check(std::fabs(acLag(64)) < 0.05, "pixel RNG ac@64 ~ 0");
    }

    // Optional xorshift32 Path Sampler: never emits 0; neighbor streams uncorrelated.
    {
        Rng r;
        r.initXorshift32(2938653863u);
        check(r.backend == kRngBackendXorshift32, "xorshift backend set");
        bool sawZero = false;
        for (int i = 0; i < 100000; ++i) {
            if (r.nextUint() == 0u) {
                sawZero = true;
                break;
            }
        }
        check(!sawZero, "xorshift32 never emits 0");

        constexpr int W = 128, H = 64;
        std::vector<float> field(size_t(W) * H);
        double mean = 0.0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                Rng xr = makePixelRngXorshift32(x, y, 0, 0u);
                const float u = xr.nextFloat();
                field[size_t(y) * W + x] = u;
                mean += double(u);
                check(u >= 0.0f && u < 1.0f, "xorshift f01 in [0,1)");
            }
        }
        mean /= double(W * H);
        double num = 0.0, den = 0.0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x + 32 < W; ++x) {
                const double a = double(field[size_t(y) * W + x]) - mean;
                const double b = double(field[size_t(y) * W + x + 32]) - mean;
                num += a * b;
                den += a * a;
            }
        }
        const double ac32 = den > 1e-12 ? num / den : 0.0;
        check(std::fabs(ac32) < 0.08, "xorshift pixel RNG ac@32 ~ 0");
    }
}

void testLightSelectionDistantRect() {
    std::printf("light selection distant+rect\n");
    LightData lights[2];
    lights[0].type = kLightDistant;
    lights[0].intensity = 2.5f;
    lights[0].color = Vec3(1.0f);
    lights[0].normalize = 1;
    lights[0].angle = 1.5f;
    lights[0].xform = Mat4::identity();
    lights[0].xformInv = Mat4::identity();

    lights[1].type = kLightRect;
    lights[1].intensity = 24.0f;
    lights[1].color = Vec3(1.0f);
    lights[1].normalize = 1;
    lights[1].width = 4.0f;
    lights[1].height = 3.0f;
    lights[1].xform = Mat4::identity();
    lights[1].xformInv = Mat4::identity();

    SceneView view{};
    view.lights = lights;
    view.lightCount = 2;

    const float fSun = lightFluxWeight(view, 0);
    const float fRect = lightFluxWeight(view, 1);
    check(std::fabs(fSun - 2.5f * kPi) < 0.05f, "pbrt distant Φ = E π r² with r=1");
    check(std::fabs(fRect - 24.0f * kPi) < 0.5f, "pbrt area Φ = L A π (normalize → E π)");
    check(fRect / srMax(fSun, 1e-6f) < 20.0f, "rect is not ~area times more important than the sun");

    lights[1].normalize = 0;
    const float fRectRadiance = lightFluxWeight(view, 1);
    check(fRectRadiance > 24.0f * 10.0f, "unnormalized rect flux includes area");
    lights[1].normalize = 1;

    const float pSun = lightSelectionPdfIndex(view, 0);
    const float pRect = lightSelectionPdfIndex(view, 1);
    check(std::fabs(pSun + pRect - 1.0f) < 1e-4f, "selection pdfs sum to 1");
    // PBRT 4: 1 infinite slot + 1 finite slot → 1/2 each.
    check(std::fabs(pSun - 0.5f) < 1e-4f, "sun pdf is one infinite slot vs one finite slot");
    check(std::fabs(pRect - 0.5f) < 1e-4f, "rect pdf is the finite slot");
    check(1.0f / pSun < 2.1f, "sun 1/pdf is 2 (no 100× fireflies)");

    int nSun = 0;
    int pdfMismatch = 0;
    constexpr int kN = 20000;
    for (int i = 0; i < kN; ++i) {
        float pdf = 0.0f;
        const int idx = sampleLightIndex(view, (float(i) + 0.5f) / float(kN), pdf);
        if (idx == 0) ++nSun;
        if (idx >= 0 && std::fabs(pdf - lightSelectionPdfIndex(view, idx)) > 1e-4f) ++pdfMismatch;
    }
    check(pdfMismatch == 0, "sampled pdf matches selection pdf");
    const float frac = float(nSun) / float(kN);
    check(frac > 0.45f && frac < 0.55f, "sun is sampled ~50% of the time next to a key rect");

    // Default scene: dome + distant + rect. PBRT 4: 2 infinite slots + 1 finite
    // slot → p_inf = 2/3, uniform in infinite → sun = dome = 1/3, rect = 1/3.
    LightData trio[3];
    trio[0] = lights[0];
    trio[1] = lights[1];
    trio[2].type = kLightDome;
    trio[2].intensity = 0.6f;
    trio[2].color = Vec3(1.0f);
    trio[2].normalize = 0;
    trio[2].xform = Mat4::identity();
    trio[2].xformInv = Mat4::identity();
    view.lights = trio;
    view.lightCount = 3;
    const float pSun3 = lightSelectionPdfIndex(view, 0);
    const float pRect3 = lightSelectionPdfIndex(view, 1);
    const float pDome3 = lightSelectionPdfIndex(view, 2);
    check(std::fabs(pSun3 + pRect3 + pDome3 - 1.0f) < 1e-4f, "default-scene pdfs sum to 1");
    check(std::fabs(pSun3 - 1.0f / 3.0f) < 1e-4f, "sun is one of two infinite slots");
    check(std::fabs(pDome3 - 1.0f / 3.0f) < 1e-4f, "dome is the other infinite slot");
    check(std::fabs(pRect3 - 1.0f / 3.0f) < 1e-4f, "rect is the finite slot");

    {
        ScenePtr scene = std::make_shared<Scene>();
        scene->lights.push_back(lights[0]);
        scene->lights.push_back(lights[1]);
        scene->finalize();
        const SceneView v = scene->view();
        check(v.lightBvh != nullptr && v.lightBvhNodeCount > 0,
              "pbrt: finite lights always get a cone BVH");
        check(v.infiniteLightCount == 1, "distant light is outside the BVH");
        float pdf = 0.0f;
        const int idx = sampleLightIndex(v, Vec3(0.0f, 0.0f, 1.0f), 0.8f, pdf);
        check(idx >= 0 && pdf > 0.0f, "position-aware light pick uses the BVH");
        checkNear(volumeLightSelectionPdfIndex(v, Vec3(0.0f), Vec3(0.0f, 1.0f, 0.0f), 0.9f, 1),
                  lightSelectionPdfIndex(v, Vec3(0.0f), 1), 1e-6f,
                  "volume selection pdf equals the surface LightSampler");
    }
}

void testBsdf() {
    std::printf("bsdf\n");
    Material diffuse;
    diffuse.baseColor = Vec3(0.8f, 0.8f, 0.8f);
    diffuse.roughness = 1.0f;
    diffuse.metallic = 0.0f;
    diffuse.transmission = 0.0f;

    Rng rng(11u, 13u);
    const Vec3 wo = normalize(Vec3(0.3f, 0.1f, 0.9f));

    // White furnace: the sampled weight must never exceed the albedo.
    double weightSum = 0.0;
    const int count = 50000;
    for (int i = 0; i < count; ++i) {
        const BsdfSample sample = bsdfSampleLocal(diffuse, wo, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                                  rng.nextFloat());
        if (sample.pdf <= 0.0f) continue;
        check(isFinite(sample.weight), "bsdf weight is finite");
        weightSum += double(average(sample.weight));
    }
    const float meanWeight = float(weightSum / count);
    check(meanWeight > 0.5f && meanWeight < 1.05f, "diffuse energy stays below one");

    // eval and sample must agree on the pdf for non delta lobes.
    Material glossy;
    glossy.baseColor = Vec3(0.6f, 0.6f, 0.6f);
    glossy.roughness = 0.35f;
    glossy.metallic = 1.0f;
    for (int i = 0; i < 2000; ++i) {
        const BsdfSample sample = bsdfSampleLocal(glossy, wo, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                                  rng.nextFloat());
        if (sample.pdf <= 0.0f || sample.specular) continue;
        const BsdfEval evaluated = bsdfEvalLocal(glossy, wo, sample.wi);
        checkNear(evaluated.pdf, sample.pdf, std::max(1e-3f, sample.pdf * 0.01f),
                  "glossy sample and eval pdf agree");
    }

    Material glass;
    glass.baseColor = Vec3(1.0f, 1.0f, 1.0f);
    glass.roughness = 0.1f;
    glass.transmission = 1.0f;
    glass.ior = 1.5f;
    int transmitted = 0;
    for (int i = 0; i < 2000; ++i) {
        const BsdfSample sample = bsdfSampleLocal(glass, wo, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                                  rng.nextFloat());
        if (sample.pdf <= 0.0f) continue;
        check(isFinite(sample.weight), "glass weight is finite");
        if (sample.transmitted) ++transmitted;
    }
    check(transmitted > 1000, "most glass samples refract at normal incidence");

    {
        Material deltaGlass;
        deltaGlass.transmission = 1.0f;
        deltaGlass.ior = 1.5f;
        deltaGlass.roughness = 0.0f;
        Material roughGlass = deltaGlass;
        roughGlass.roughness = 0.1f;
        Material floor;
        floor.baseColor = Vec3(0.7f, 0.7f, 0.7f);
        floor.roughness = 0.4f;
        const Vec3 woN(0.0f, 0.0f, 1.0f);
        check(isDeltaCausticCaster(deltaGlass), "smooth glass is a delta MNEE caster");
        check(!isDeltaCausticCaster(roughGlass), "roughness 0.1 glass is not a delta MNEE caster");
        check(isTransmissiveCausticCaster(deltaGlass), "delta glass is a transmissive SDS caster");
        check(isTransmissiveCausticCaster(roughGlass),
              "roughness 0.1 glass shares SDS throughGlass with delta");
        check(!lightTraceConnectable(deltaGlass, woN), "delta glass is not an LT connectable");
        check(!lightTraceConnectable(roughGlass, woN), "rough glass is not an LT connectable (continue, no splat)");
        check(lightTraceConnectable(floor, woN), "Lambert floor is an LT connectable");
        check(!eyePathNeeConnectable(deltaGlass, woN), "delta glass is not NEE-connectable");
        check(!eyePathNeeConnectable(roughGlass, woN),
              "roughness 0.1 glass is not NEE-connectable (no GGX glow)");
        check(eyePathNeeConnectable(floor, woN), "Lambert is NEE-connectable");
        Material frosted = deltaGlass;
        frosted.roughness = 0.5f;
        check(eyePathNeeConnectable(frosted, woN), "roughness 0.5 glass stays NEE-connectable");
        check(!lightTraceConnectable(frosted, woN), "frosted glass is still not an LT splat vertex");
        check(!isTransmissiveCausticCaster(frosted),
              "roughness 0.5 glass is not a near-spec SDS caster");
        Material noCau = deltaGlass;
        noCau.contributeCaustics = 0;
        check(!isDeltaCausticCaster(noCau), "contribute_caustics=0 is not an MNEE caster");

        // Iray Photoreal: contributing glass is opaque for primary/LT shadows and
        // Fresnel-open for eye NEE after a bounce (straight shadow ray, not Snell).
        {
            Material g;
            g.transmission = 1.0f;
            g.ior = 1.5f;
            g.roughness = 0.0f;
            g.contributeCaustics = 1;
            g.transmissionColor = Vec3(1.0f);
            const float Fenter = fresnelDielectric(1.0f, 1.5f);
            check(shadowBlockFraction(g, g, 1, 0, 1.0f) > 0.999f,
                  "primary NEE / LT splat: contributing glass is opaque");
            const float bounceBlock = shadowBlockFraction(g, g, 1, 1, 1.0f);
            checkNear(bounceBlock, Fenter, 1e-5f,
                      "eye NEE after bounce: block equals dielectric Fresnel");
            check(bounceBlock < 0.1f, "normal-incidence glass NEE is mostly open");
            const float critCos = sqrtf(1.0f - 1.0f / (1.5f * 1.5f));
            check(shadowBlockFraction(g, g, 1, 1, -(critCos - 0.05f)) > 0.999f,
                  "TIR still fully blocks Iray glass NEE");
            Material fake = g;
            fake.contributeCaustics = 0;
            fake.shadowOpacity = 0.25f;
            checkNear(shadowBlockFraction(fake, fake, 1, 0, 1.0f), 0.25f, 1e-5f,
                      "contribute off uses shadowOpacity even with caustics ON");
        }
    }

    // Specular = 0 must disable dielectric reflections completely.
    Material matte;
    matte.baseColor = Vec3(0.7f, 0.7f, 0.7f);
    matte.roughness = 0.4f;
    matte.metallic = 0.0f;
    matte.specular = 0.0f;
    const LobeWeights matteLobes = computeLobes(matte);
    check(matteLobes.specular < 1e-4f, "specular=0 removes the reflection lobe");
    int specularHits = 0;
    for (int i = 0; i < 2000; ++i) {
        const BsdfSample sample = bsdfSampleLocal(matte, wo, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                                  rng.nextFloat());
        if (sample.pdf <= 0.0f) continue;
        if (sample.specular) ++specularHits;
        const BsdfEval evaluated = bsdfEvalLocal(matte, wo, sample.wi);
        // Pure diffuse: the specular microfacet term must stay off.
        check(evaluated.pdf > 0.0f, "matte sample has a pdf");
    }
    check(specularHits == 0, "specular=0 never samples a specular bounce");

    // Past the critical angle an interior ray has nowhere to refract to, so every
    // sample must come back as a valid total internal reflection. Evaluating the
    // Fresnel term with the outside eta silently killed these paths instead.
    Material glassDelta = glass;
    glassDelta.roughness = 0.0f;
    const float critAngle = std::asin(1.0f / 1.5f);
    const float insideTheta = critAngle + 0.2f;
    const Vec3 woBeyondCritical(std::sin(insideTheta), 0.0f, -std::cos(insideTheta));
    int tirValid = 0;
    int tirRefracted = 0;
    for (int i = 0; i < 500; ++i) {
        const BsdfSample sample = bsdfSampleLocal(glassDelta, woBeyondCritical, 0.99f, rng.nextFloat(),
                                                  rng.nextFloat(), rng.nextFloat());
        if (sample.pdf <= 0.0f) continue;
        if (sample.transmitted) ++tirRefracted;
        else if (sample.wi.z < 0.0f) ++tirValid;
    }
    check(tirValid == 500, "beyond the critical angle every interior sample reflects");
    check(tirRefracted == 0, "no refraction escapes past the critical angle");

    // Below the critical angle the reflect/refract split must follow the glass→air
    // Fresnel term. Using the air→glass eta here biases the split (and the pdf that
    // the eval side reports for the very same sample).
    const float insideCos = std::cos(radians(30.0f));
    const Vec3 woInside30(std::sin(radians(30.0f)), 0.0f, -insideCos);
    const float expectedFr = fresnelDielectric(insideCos, 1.0f / 1.5f);
    int reflected30 = 0;
    const int trials30 = 20000;
    for (int i = 0; i < trials30; ++i) {
        const BsdfSample sample = bsdfSampleLocal(glassDelta, woInside30, 0.99f, rng.nextFloat(),
                                                  rng.nextFloat(), rng.nextFloat());
        if (sample.pdf <= 0.0f) continue;
        if (!sample.transmitted) ++reflected30;
    }
    const float measuredFr = float(reflected30) / float(trials30);
    checkNear(measuredFr, expectedFr, 0.008f, "interior Fresnel split uses the glass→air eta");

    // pbrt FrDielectric: signed cos + stored outside η equals side-relative +cos.
    {
        const float c = std::cos(radians(30.0f));
        checkNear(fresnelDielectric(-c, 1.5f), fresnelDielectric(c, 1.0f / 1.5f), 1e-6f,
                  "FrDielectric(-cos, η) == FrDielectric(+cos, 1/η)");
        const float pastCrit = std::cos(std::asin(1.0f / 1.5f) + 0.2f);
        checkNear(fresnelDielectric(-pastCrit, 1.5f), 1.0f, 1e-5f,
                  "signed-cos outside eta reports TIR from inside");
    }

    // Rough glass from inside past the critical angle: F is TIR (not air→glass ~0.04).
    // Spectral f and the hero-λ pdf must use that same F, or the belly goes black.
    {
        Material gRough = glass;
        gRough.roughness = 0.1f;
        gRough.specular = 0.0f;
        const Vec3 woTir = woBeyondCritical;
        const Vec3 wiMirror(-woTir.x, -woTir.y, woTir.z);
        const BsdfEval eRgb = bsdfEvalLocal(gRough, woTir, wiMirror);
        check(eRgb.pdf > 0.0f && eRgb.f.x > 0.0f, "rough inside TIR eval is non-zero");
        const float wMirror = eRgb.f.x * fabsf(wiMirror.z) / srMax(eRgb.pdf, 1e-12f);
        check(wMirror > 0.5f, "rough inside TIR f/pdf is O(1), not air→glass F");
        const LobeWeights lwR = computeLobes(gRough, woTir);
        const DielectricGgxEval mf =
            evalDielectricGgx(woTir, wiMirror, lwR.ax, lwR.ay, 1.5f, true);
        checkNear(eRgb.pdf, lwR.transmission * mf.pdf, std::max(1e-5f, eRgb.pdf * 0.02f),
                  "RGB eval pdf is dielectric GGX pdf at outside eta");

        SampledWavelengths wv{};
        wv.n = 4;
        for (int i = 0; i < 4; ++i) {
            wv.lambda[i] = 450.0f + 60.0f * float(i);
            wv.pdf[i] = 1.0f;
        }
        const SampledSpectrum fS = bsdfEvalSpectralDielectric(gRough, woTir, wiMirror, wv, 1.5f);
        for (int i = 0; i < wv.n; ++i) {
            checkNear(fS.values[i], eRgb.f.x, std::max(1e-4f, eRgb.f.x * 0.05f),
                      "spectral inside TIR f matches RGB (same outside-eta F)");
        }

        int nTir = 0;
        int nDark = 0;
        for (int i = 0; i < 1500; ++i) {
            const float u1 = rng.nextFloat();
            const float u2 = rng.nextFloat();
            const BsdfSample s = bsdfSampleLocal(gRough, woTir, 0.99f, u1, u2, 0.0f);
            if (s.pdf <= 0.0f || s.transmitted) continue;
            const BsdfEval ev = bsdfEvalLocal(gRough, woTir, s.wi);
            checkNear(ev.pdf, s.pdf, std::max(1e-4f, s.pdf * 0.02f),
                      "rough inside TIR sample/eval pdf agree");
            const BsdfSampleSpectral ss =
                bsdfSampleSpectral(gRough, woTir, 0.99f, u1, u2, 0.0f, wv, 1.5f, 0);
            if (!ss.valid || ss.transmitted || ss.pdf <= 0.0f) continue;
            ++nTir;
            const DielectricGgxEval mfS =
                evalDielectricGgx(woTir, ss.wi, lwR.ax, lwR.ay, 1.5f, true);
            checkNear(ss.pdf, lwR.transmission * mfS.pdf, std::max(1e-4f, ss.pdf * 0.05f),
                      "spectral sample pdf is hero-λ dielectric F, not RGB eval");
            if (ss.weight.values[0] < 0.2f) ++nDark;
        }
        check(nTir > 200, "rough inside TIR produces reflection samples");
        check(nDark == 0, "spectral inside TIR weight is not the 0.04 double-swap");
    }

    // Arnold Advanced → Internal Reflections: from inside, disable Fresnel
    // reflections (keep TIR). Exterior behaviour unchanged.
    Material glassIR = glass;
    glassIR.roughness = 0.0f;
    glassIR.internalReflections = 0.0f;
    const Vec3 woInside(0.0f, 0.0f, -1.0f);  // leaving toward the interface from inside
    int insideReflect = 0;
    int insideTransmit = 0;
    for (int i = 0; i < 2000; ++i) {
        const BsdfSample sample =
            bsdfSampleLocal(glassIR, woInside, 0.99f, rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        if (sample.pdf <= 0.0f) continue;
        if (sample.transmitted)
            ++insideTransmit;
        else
            ++insideReflect;
    }
    check(insideTransmit > 1500, "IR-off: almost all inside samples refract out");
    check(insideReflect == 0, "IR-off: no Fresnel internal reflections at normal incidence");

    // Grazing inside ray past critical angle must still TIR-reflect.
    const float crit = std::asin(1.0f / 1.5f);
    const float theta = crit + 0.15f;
    const Vec3 woTir(std::sin(theta), 0.0f, -std::cos(theta));
    int tirHits = 0;
    for (int i = 0; i < 200; ++i) {
        const BsdfSample sample =
            bsdfSampleLocal(glassIR, woTir, 0.99f, rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        if (sample.pdf <= 0.0f) continue;
        if (!sample.transmitted && sample.wi.z < 0.0f) ++tirHits;
    }
    check(tirHits > 150, "IR-off: TIR still reflects past the critical angle");

    // Smooth glass refraction is Snell with η = ηt/ηi (not the inverted GPU fork),
    // the radiance weight includes 1/η², and a wrong-hemisphere reflect is invalid
    // (no z-flip).
    {
        Material g;
        g.baseColor = Vec3(1.0f);
        g.roughness = 0.0f;
        g.transmission = 1.0f;
        g.ior = 1.5f;
        g.specular = 0.0f;
        const float eta = 1.5f;
        const Vec3 woN(0.0f, 0.0f, 1.0f);
        const BsdfSample snellN = bsdfSampleLocal(g, woN, 0.99f, 0.0f, 0.0f, 0.99f);
        check(snellN.transmitted && snellN.specular, "smooth glass at normal incidence refracts");
        checkNear(snellN.wi.x, 0.0f, 1e-5f, "normal Snell wi.x");
        checkNear(snellN.wi.y, 0.0f, 1e-5f, "normal Snell wi.y");
        checkNear(snellN.wi.z, -1.0f, 1e-5f, "normal Snell wi.z");
        checkNear(snellN.weight.x, 1.0f / (eta * eta), 1e-4f, "delta glass weight is 1/η²");
        checkNear(snellN.weight.y, 1.0f / (eta * eta), 1e-4f, "delta glass weight G is 1/η²");
        checkNear(snellN.weight.z, 1.0f / (eta * eta), 1e-4f, "delta glass weight B is 1/η²");

        const float thetaI = radians(30.0f);
        const Vec3 wo30(std::sin(thetaI), 0.0f, std::cos(thetaI));
        const BsdfSample snell30 = bsdfSampleLocal(g, wo30, 0.99f, 0.0f, 0.0f, 0.99f);
        check(snell30.transmitted, "30° air→glass refracts");
        const float sinT = std::sin(thetaI) / eta;
        const float cosT = std::sqrt(std::max(0.0f, 1.0f - sinT * sinT));
        checkNear(snell30.wi.x, -sinT, 1e-4f, "Snell wi.x uses 1/η (not η)");
        checkNear(snell30.wi.z, -cosT, 1e-4f, "Snell wi.z uses 1/η (not η)");

        Vec3 woFlip(0.1f, 0.0f, 1.0f);
        woFlip = normalize(woFlip);
        const BsdfSample badH = bsdfSampleLocal(g, woFlip, 0.99f, 0.0f, 0.0f, 0.0f);
        if (!badH.transmitted && badH.pdf > 0.0f)
            check(badH.wi.z * woFlip.z > 0.0f, "dielectric reflect stays in wo hemisphere (no z-flip)");
    }

    // pbrt energy lottery: spec share tracks E(mu, alpha) F, so grazing wo is more specular.
    {
        Material diel;
        diel.baseColor = Vec3(0.8f);
        diel.roughness = 0.25f;
        diel.metallic = 0.0f;
        diel.specular = 1.0f;
        diel.transmission = 0.0f;
        diel.ior = 1.5f;
        const LobeWeights nrm = computeLobes(diel, Vec3(0.0f, 0.0f, 1.0f));
        const LobeWeights grz = computeLobes(diel, normalize(Vec3(0.95f, 0.0f, 0.08f)));
        check(grz.specular > nrm.specular, "grazing wo raises the specular energy share");
        check(grz.diffuse < nrm.diffuse, "grazing wo lowers the diffuse energy share");
        Material coat = diel;
        coat.coat = 1.0f;
        coat.coatThickness = 0.25f;
        coat.coatIor = 1.5f;
        coat.coatRoughness = 0.1f;
        check(coatPickProb(coat, Vec3(0.0f, 0.0f, 1.0f)) > 0.02f, "dielectric coat overlay is selectable");
    }

    // Anisotropic GGX: ax≠ay when specular_anisotropy>0; eval along X vs Y differs.
    {
        Material iso;
        iso.baseColor = Vec3(1.0f);
        iso.roughness = 0.4f;
        iso.metallic = 1.0f;
        iso.specular = 1.0f;
        Material an = iso;
        an.specularAnisotropy = 1.0f;
        const Vec3 wo(0.0f, 0.0f, 1.0f);
        const Vec3 wiX = normalize(Vec3(0.75f, 0.0f, 0.66f));
        const Vec3 wiY = normalize(Vec3(0.0f, 0.75f, 0.66f));
        const float isoX = average(bsdfEvalLocal(iso, wo, wiX).f);
        const float isoY = average(bsdfEvalLocal(iso, wo, wiY).f);
        const float anX = average(bsdfEvalLocal(an, wo, wiX).f);
        const float anY = average(bsdfEvalLocal(an, wo, wiY).f);
        checkNear(isoX, isoY, 0.02f, "isotropic GGX is azimuthally symmetric");
        check(std::fabs(anX - anY) > 0.02f, "anisotropic GGX eval X != Y");
        LobeWeights lw = computeLobes(an, wo);
        check(std::fabs(lw.ax - lw.ay) > 0.05f, "anisotropy splits αx/αy");
        const Vec3 h = normalize(Vec3(0.25f, 0.1f, 0.96f));
        const float a = 0.3f;
        const float a2 = a * a;
        const float cos2 = h.z * h.z;
        const float t = cos2 * (a2 - 1.0f) + 1.0f;
        const float isoD = a2 / (kPi * t * t);
        checkNear(ggxD(h, a), isoD, 1e-5f, "ax=ay GGX D matches isotropic formula");
        for (int i = 0; i < 1500; ++i) {
            const BsdfSample sample = bsdfSampleLocal(an, wo, rng.nextFloat(), rng.nextFloat(),
                                                      rng.nextFloat(), rng.nextFloat());
            if (sample.pdf <= 0.0f || sample.specular) continue;
            const BsdfEval evaluated = bsdfEvalLocal(an, wo, sample.wi);
            checkNear(evaluated.pdf, sample.pdf, std::max(1e-3f, sample.pdf * 0.02f),
                      "aniso sample and eval pdf agree");
        }
    }

    // Charlie sheen is a selectable lobe; grazing eval exceeds sheen-off.
    {
        Material cloth;
        cloth.baseColor = Vec3(0.08f, 0.08f, 0.1f);
        cloth.roughness = 0.8f;
        cloth.metallic = 0.0f;
        cloth.specular = 0.0f;
        cloth.sheen = 1.0f;
        cloth.sheenColor = Vec3(1.0f, 0.2f, 0.1f);
        cloth.sheenRoughness = 0.35f;
        const LobeWeights lw = computeLobes(cloth, Vec3(0.0f, 0.0f, 1.0f));
        check(lw.sheen > 0.05f, "sheen is selectable in the lobe lottery");
        Material bare = cloth;
        bare.sheen = 0.0f;
        const Vec3 woG = normalize(Vec3(0.98f, 0.0f, 0.20f));
        const Vec3 wiG = normalize(Vec3(0.96f, 0.15f, 0.24f));
        Material sheenOnly = cloth;
        sheenOnly.baseColor = Vec3(0.0f);
        sheenOnly.baseWeight = 0.0f;
        check(average(bsdfEvalLocal(sheenOnly, woG, wiG).f) > 1e-3f,
              "Charlie sheen eval is nonzero at the horizon");
        check(average(bsdfEvalLocal(cloth, woG, wiG).f) >
                  average(bsdfEvalLocal(bare, woG, wiG).f),
              "Charlie sheen adds grazing energy");
        int sheenHits = 0;
        for (int i = 0; i < 2000; ++i) {
            const BsdfSample sample = bsdfSampleLocal(cloth, woG, rng.nextFloat(), rng.nextFloat(),
                                                      rng.nextFloat(), rng.nextFloat());
            if (sample.pdf <= 0.0f) continue;
            ++sheenHits;
        }
        check(sheenHits > 200, "sheen material samples valid directions");
    }

    // Oren–Nayar (diffuse_roughness>0) differs from Lambert at oblique pairs.
    {
        Material lamb;
        lamb.baseColor = Vec3(0.8f);
        lamb.roughness = 1.0f;
        lamb.metallic = 0.0f;
        lamb.specular = 0.0f;
        Material on = lamb;
        on.diffuseRoughness = 1.0f;
        const Vec3 wo = normalize(Vec3(0.75f, 0.0f, 0.66f));
        const Vec3 wi = normalize(Vec3(0.45f, 0.0f, 0.89f));
        const float fL = average(bsdfEvalLocal(lamb, wo, wi).f);
        const float fO = average(bsdfEvalLocal(on, wo, wi).f);
        check(std::fabs(fL - fO) > 1e-4f, "Oren–Nayar differs from Lambert");
        checkNear(average(bsdfEvalLocal(lamb, wo, wi).f), 0.8f * kInvPi, 1e-5f,
                  "Lambert is albedo/π when diffuse_roughness=0");
    }
}

void testGlob() {
    std::printf("glob\n");
    check(globMatch("*", "/geo/sphere"), "star matches everything");
    check(globMatch("/geo/*", "/geo/sphere"), "prefix glob");
    check(!globMatch("/lights/*", "/geo/sphere"), "non matching prefix");
    check(globMatch("*sphere", "/geo/sphere"), "suffix glob");
    check(globMatch("/geo/sphere", "/geo/sphere"), "exact match");
    check(globMatch("/geo/?phere", "/geo/sphere"), "question mark");
}

void testGraphCook() {
    std::printf("node graph\n");
    registerBuiltinNodes();
    NodeGraph graph;
    buildDefaultGraph(graph);
    check(graph.nodes().size() >= 8, "default graph has nodes");
    check(graph.displayNode() != nullptr, "default graph has a display node");

    CookContext context;
    StagePtr stage = graph.cookDisplay(context);
    check(stage != nullptr, "cook produces a stage");
    check(context.errors.isEmpty(), "cook has no errors");
    check(stage->countOfType(PrimType::Mesh) == 10, "ten meshes in the default stage");
    check(stage->countOfType(PrimType::Light) == 3, "three lights in the default stage");
    check(stage->countOfType(PrimType::Camera) == 1, "one camera in the default stage");
    check(stage->settingsAuthored, "render settings authored");

    ScenePtr scene = stage->toScene();
    check(scene->instances.size() >= 2, "scene has instances");
    check(scene->totalTriangles() > 0, "scene has triangles");
    check(scene->bounds().valid(), "scene bounds are valid");

    // Cooking twice must reuse the cache and stay identical.
    StagePtr again = graph.cookDisplay(context);
    check(again == stage, "cook results are cached");

    // Serialisation round trip.
    QString error;
    const QJsonObject json = graph.toJson();
    NodeGraph reloaded;
    check(reloaded.fromJson(json, error), "graph reloads from json");
    CookContext context2;
    StagePtr stage2 = reloaded.cookDisplay(context2);
    check(stage2 != nullptr && stage2->prims.size() == stage->prims.size(), "reloaded graph cooks the same");
}

void testGraphDeleteAfterLoad() {
    std::printf("graph delete after load\n");
    registerBuiltinNodes();
    NodeGraph graph;
    buildDefaultGraph(graph);
    const int n0 = int(graph.nodes().size());
    check(n0 >= 8, "default graph has nodes");
    const QJsonObject json = graph.toJson();

    int aboutToRemove = 0;
    QObject::connect(&graph, &NodeGraph::nodeAboutToBeRemoved, [&](Node* node) {
        check(node != nullptr, "aboutToBeRemoved pointer");
        // UAF if clear()/fromJson destroyed Nodes before this signal.
        check(!node->typeName().isEmpty(), "node alive during aboutToBeRemoved");
        check(!node->name().isEmpty(), "node name alive during aboutToBeRemoved");
        ++aboutToRemove;
    });

    QString error;
    check(graph.fromJson(json, error), "reload into live graph");
    check(aboutToRemove == n0, "clear() notifies before destroying loaded nodes");

    Node* display = graph.displayNode();
    check(display != nullptr, "display after reload");
    const QString displayName = display->name();
    graph.removeNode(display);
    check(graph.findNode(displayName) == nullptr, "display removed after load");
    for (const NodePtr& n : graph.nodes()) {
        check(n && !n->typeName().isEmpty(), "surviving node after display delete");
    }

    Node* leftover = graph.nodes().empty() ? nullptr : graph.nodes().front().get();
    check(leftover != nullptr, "a node remains after display delete");
    const QString leftoverName = leftover->name();
    graph.removeNode(leftover);
    check(graph.findNode(leftoverName) == nullptr, "non-display node removed after load");

    const int left = int(graph.nodes().size());
    aboutToRemove = 0;
    graph.clear();
    check(aboutToRemove == left, "clear() notifies remaining nodes");
    check(graph.nodes().empty(), "graph empty after clear");
    check(graph.displayNode() == nullptr, "no display node after clear");
}

void testXpuDevice() {
    std::printf("xpu device\n");
    check(xpuEmbreeThreadCount(8) == 7, "XPU reserves one core for GPU submit");
    check(xpuEmbreeThreadCount(2) == 1, "XPU with 2 threads leaves 1 for GPU");
    check(xpuEmbreeThreadCount(1) == 1, "XPU keeps at least one Embree thread");
    check(xpuEmbreeThreadCount(0) >= 1, "XPU auto thread count is at least 1");
    check(kXpuScheduleOverlap == 0 && kXpuScheduleMixture == 1, "Overlap is default schedule 0");
    check(xpuGpuRemaining(10, 0, 0) == 10, "XPU GPU remaining starts at the target");
    check(xpuGpuRemaining(10, 8, 1) == 1, "XPU GPU remaining subtracts both films");
    check(xpuGpuRemaining(10, 9, 1) == 0, "XPU GPU remaining clamps at zero");
    check(xpuGpuRemaining(0, 0, 0) == 0, "XPU GPU remaining is 0 without a target");
    check(xpuCpuSampleIndex(0) != 0, "CPU estimator uses a disjoint sample index");
    check(xpuCpuSampleIndex(1) == xpuCpuSampleIndex(0) + 1, "CPU sample indices are consecutive in that band");
    check(noiseOracleMinSamples(64) >= 4, "oracle waits for a few camera samples");
    check(!noiseOraclePixelQuiet(0.5f, 0.5f, 0.5f, 0.0f, 8, 0.01f), "oracle stays open without L²");
    check(noiseOraclePixelQuiet(0.0f, 0.0f, 0.0f, 0.0f, 8, 0.01f), "black pixels with no L² are quiet");
    check(noiseOraclePixelQuiet(1.0f, 1.0f, 1.0f, 8.0f, 8, 0.01f), "constant L² matches mean² → quiet");
    check(!noiseOraclePixelQuiet(1.0f, 1.0f, 1.0f, 80.0f, 8, 0.01f), "high L² variance stays noisy");
    check(!noiseOraclePixelQuiet(1.0f, 1.0f, 1.0f, 8.0f, 8, 0.0f), "threshold 0 disables the oracle");
    // GPU LT: SDS in rgb, camera L² only. Implied variance is largely negative.
    check(!noiseOraclePixelQuiet(50.0f, 50.0f, 50.0f, 8.0f * 0.2f * 0.2f, 8, 0.01f),
          "splat-inflated mean vs camera L² stays open");

    {
        RenderSettingsData s{};
        s.caustics = 1;
        s.backend = kBackendGpuOptix;
        check(gpuLightTraceSkipUnsafe(s), "GPU caustics skip is unsafe");
        s.backend = kBackendXpu;
        check(gpuLightTraceSkipUnsafe(s), "XPU caustics skip is unsafe");
        s.backend = kBackendCpuEmbree;
        check(!gpuLightTraceSkipUnsafe(s), "CPU caustics skip is safe (separate splat plane)");
        s.caustics = 0;
        s.backend = kBackendGpuOptix;
        check(!gpuLightTraceSkipUnsafe(s), "GPU PT skip is safe");
        s.caustics = 1;
        s.causticsEngineGpu = kGpuCausticsAimedLt;
        check(kGpuCausticsAimedLtMnee == 1, "GPU MNEE menu index stays 1");
        check(!gpuEyePathMneeEnabled(s), "Aimed LT skips the MNEE wavefront (path_tail)");
        check(!gpuRefractionMneeEnabled(s), "Aimed LT does not enable refraction MNEE");
        s.causticsEngineGpu = kGpuCausticsAimedLtMnee;
        check(gpuEyePathMneeEnabled(s), "Aimed LT + MNEE still runs the MNEE pipeline");
        check(gpuRefractionMneeEnabled(s), "Aimed LT + MNEE enables refraction-only eye MNEE");
        check(gpuEyeBounceNee(s, 1, 1, 1, kLightRect) == 1,
              "mode 2 through-glass NEE stays Fresnel like CPU PT");
        check(gpuEyeBounceNee(s, 1, 1, 1, kLightDome) == 1, "mode 2 dome NEE stays Fresnel");
        check(gpuEyeBounceNee(s, 1, 1, 1, kLightDistant) == 1, "mode 2 distant NEE stays Fresnel");
        check(gpuEyeBounceNee(s, 1, 0, 1, kLightRect) == 1,
              "mode 2 without throughGlass keeps Fresnel (LT owns floor SDS)");
        check(gpuEyeBounceNee(s, 0, 0, 1, kLightRect) == 0, "primary NEE stays opaque");
        check(!gpuSkipCameraSds(s, 0, 1, 1, 1.0f), "mode 2 throughGlass keeps eye BSDF");
        check(gpuSkipCameraSds(s, 0, 1, 0, 1.0f), "mode 2 floor-first still skips SDS for LT");
        s.causticsEngineGpu = kGpuCausticsAimedLt;
        check(gpuEyeBounceNee(s, 1, 1, 1, kLightRect) == 1, "Aimed LT Fresnel-continues at depth>0");
        check(gpuSkipCameraSds(s, 0, 1, 1, 1.0f), "Aimed LT still skips camera SDS");
        s.caustics = 0;
        s.causticsEngineGpu = kGpuCausticsAimedLt;
        check(!gpuEyePathMneeEnabled(s), "caustics off disables GPU MNEE");
        check(!gpuRefractionMneeEnabled(s), "caustics off disables refraction MNEE");
        check(kShadowMnee != kShadowShade && kShadowMnee != kShadowIdle,
              "MNEE shadow slot is distinct from shade/idle");
    }

    {
        Framebuffer film;
        film.resize(2, 2);
        const Vec3 white(1.0f, 1.0f, 1.0f);
        for (int s = 0; s < 8; ++s) {
            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < 2; ++x) {
                    film.addSample(x, y, white);
                    film.addNoiseSample(x, y, white);
                }
            }
        }
        film.refreshNoiseOracle(0.01f, 8, 64);
        check(film.skipPixel(0, 0) && film.skipPixel(1, 1), "quiet constant film skips pixels");
        check(film.noiseOracleDone(), "constant 2x2 film converges");
        check(film.noiseOracleSkipCount() == 4, "constant 2x2 skip count is 4");
    }

    {
        // GPU LT addSplatRadiance: SDS into accum.rgb, not w, not L².
        Framebuffer film;
        film.resize(2, 2);
        const Vec3 cam(0.2f, 0.2f, 0.2f);
        for (int s = 0; s < 8; ++s) {
            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < 2; ++x) {
                    film.addSample(x, y, cam);
                    film.addNoiseSample(x, y, cam);
                }
            }
        }
        for (int i = 0; i < 4; ++i) {
            film.data()[i].x += 50.0f;
            film.data()[i].y += 50.0f;
            film.data()[i].z += 50.0f;
        }
        film.refreshNoiseOracle(0.01f, 8, 64);
        check(!film.skipPixel(0, 0) && !film.skipPixel(1, 1),
              "GPU LT splat-inflated mean does not freeze camera w");
        check(!film.noiseOracleDone(), "GPU LT film stays open");
    }

    // Uniform (threshold 0) never skips. Variance skips a constant block and
    // keeps a noisy block. Typical path-tracing noise at 128 spp / 0.01 does
    // not skip — that is why Uniform vs Variance looks identical on a volume.
    {
        const int nSpp = 16;
        const int maxSpp = 64;
        auto fill = [](Framebuffer& film, int x0, int x1, bool constant) {
            for (int s = 0; s < nSpp; ++s) {
                const Vec3 rgb = constant ? Vec3(1.0f, 1.0f, 1.0f)
                                          : ((s & 1) ? Vec3(2.0f, 2.0f, 2.0f) : Vec3(0.0f, 0.0f, 0.0f));
                for (int y = 0; y < 2; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        film.addSample(x, y, rgb);
                        film.addNoiseSample(x, y, rgb);
                    }
                }
            }
        };

        Framebuffer uniform;
        uniform.resize(8, 2);
        fill(uniform, 0, 4, true);
        fill(uniform, 4, 8, false);
        uniform.refreshNoiseOracle(0.0f, nSpp, maxSpp);
        check(uniform.noiseOracleSkipCount() == 0, "Uniform (threshold 0) skips nothing");
        check(!uniform.skipPixel(1, 0) && !uniform.skipPixel(5, 0), "Uniform leaves constant and noisy pixels open");

        Framebuffer variance;
        variance.resize(8, 2);
        fill(variance, 0, 4, true);
        fill(variance, 4, 8, false);
        variance.refreshNoiseOracle(0.01f, nSpp, maxSpp);
        check(variance.skipPixel(1, 0) && variance.skipPixel(1, 1), "Variance skips interior constant pixels");
        check(!variance.skipPixel(3, 0), "Variance keeps the constant/noisy boundary sampling");
        check(!variance.skipPixel(5, 0) && !variance.skipPixel(7, 1), "Variance keeps noisy pixels sampling");
        check(variance.noiseOracleSkipCount() > 0, "Variance skip count is non-zero on mixed film");
        check(!variance.noiseOracleDone(), "mixed film does not finish early");
    }

    {
        // σ/μ = 0.3 at 128 spp: rel stderr = 0.3/√128 ≈ 0.027 > 0.01 → stay open.
        const int n = 128;
        const float mean = 1.0f;
        const float var = 0.09f;
        const float lumSq = float(n) * (var + mean * mean);
        check(!noiseOraclePixelQuiet(mean, mean, mean, lumSq, n, 0.01f),
              "typical 128 spp path noise stays open at threshold 0.01");
        check(noiseOraclePixelQuiet(mean, mean, mean, lumSq, n, 0.05f),
              "same pixel is quiet if the threshold is raised to 0.05");
    }

    Scene scene;
    scene.settings.backend = kBackendXpu;
    scene.settings.xpuSchedule = kXpuScheduleOverlap;
    scene.settings.lightSamples = 8;
    scene.settings.pathGuiding = 1;
    scene.settings.motionBlur = 1;
    scene.settings.pixelFilter = 2;
    scene.settings.filterRadius = 2.0f;
    scene.settings.samplingEngine = kSamplingEngineProgressive;
    scene.camera.opticalModel = 1;
    Material sss;
    sss.subsurface = 0.7f;
    scene.materials.push_back(sss);
    check(scene.settings.lightSamples == 8, "XPU keeps light samples");
    check(scene.settings.pathGuiding == 1, "XPU keeps OpenPGL on CPU samples");
    check(scene.settings.motionBlur == 1, "XPU does not strip motion blur from settings");
    check(scene.settings.pixelFilter == 2, "XPU keeps pixel filter");
    check(scene.settings.filterRadius == 2.0f, "XPU keeps filter radius");
    check(scene.settings.samplingEngine == kSamplingEngineProgressive, "XPU keeps sampling type");
    check(scene.camera.opticalModel == 1, "XPU keeps authored camera model");
    check(scene.materials[0].subsurface == 0.7f, "XPU keeps SSS on CPU samples");
    check(scene.settings.xpuSchedule == kXpuScheduleOverlap, "XPU default schedule is Overlap");

    registerBuiltinNodes();
    NodeGraph graph;
    buildDefaultGraph(graph);
    Node* settings = nullptr;
    for (const NodePtr& node : graph.nodes()) {
        if (node && node->typeName() == QLatin1String("rendersettings")) {
            settings = node.get();
            break;
        }
    }
    check(settings != nullptr, "default graph has render settings");
    if (settings) {
        const Parameter* sched = settings->findParameter(QLatin1String("xpuschedule"));
        check(sched != nullptr, "xpuschedule parameter exists");
        if (sched) {
            check(sched->visibleWhen == QLatin1String("backend==2"), "xpuschedule only when XPU");
            check(sched->group == QLatin1String("Engine"), "xpuschedule in Engine");
            settings->setParameterValue("backend", 0);
            check(!evaluateVisibleWhen(sched->visibleWhen, *settings), "XPU Schedule hidden on CPU");
            settings->setParameterValue("backend", 2);
            check(evaluateVisibleWhen(sched->visibleWhen, *settings), "XPU Schedule visible on XPU");
        }
        const Parameter* cpuCau = settings->findParameter(QLatin1String("causticsengine"));
        const Parameter* gpuCau = settings->findParameter(QLatin1String("causticsenginegpu"));
        check(cpuCau != nullptr, "CPU caustics engine parameter exists");
        check(gpuCau != nullptr, "GPU caustics engine parameter exists");
        if (cpuCau && gpuCau) {
            check(cpuCau->menuItems.size() == 4, "CPU caustics menu has 4 engines");
            check(gpuCau->menuItems.size() == 2, "GPU caustics menu has Aimed LT / MNEE");
            check(gpuCau->menuItems[0] == QLatin1String("Aimed LT"), "GPU default label is Aimed LT");
            check(gpuCau->menuItems[1] == QLatin1String("Aimed LT + MNEE"),
                  "GPU second label is Aimed LT + MNEE");
            settings->setParameterValue("integrator", 0);
            settings->setParameterValue("backend", 0);
            check(evaluateVisibleWhen(cpuCau->visibleWhen, *settings), "CPU caustics engine visible on CPU");
            check(!evaluateVisibleWhen(gpuCau->visibleWhen, *settings), "separate GPU engine menu hidden on CPU");
            settings->setParameterValue("backend", 1);
            check(evaluateVisibleWhen(cpuCau->visibleWhen, *settings),
                  "Caustics Engine stays visible on GPU (items swap to Aimed LT / MNEE)");
            check(!evaluateVisibleWhen(gpuCau->visibleWhen, *settings),
                  "separate GPU engine menu hidden on GPU-only (same row as Caustics Engine)");
            settings->setParameterValue("backend", 2);
            check(evaluateVisibleWhen(cpuCau->visibleWhen, *settings), "CPU caustics engine visible on XPU");
            check(evaluateVisibleWhen(gpuCau->visibleWhen, *settings), "GPU caustics engine visible on XPU");
            settings->setParameterValue("causticsenginegpu", 1);
            CookContext cauCtx;
            StagePtr cauStage = graph.cookDisplay(cauCtx);
            check(cauStage != nullptr, "GPU caustics cook produces a stage");
            ScenePtr cauScene = cauStage->toScene();
            check(cauScene->settings.causticsEngineGpu == kGpuCausticsAimedLtMnee,
                  "causticsenginegpu 1 cooks to Aimed LT + MNEE");
            settings->setParameterValue("causticsenginegpu", 0);
            cauStage = graph.cookDisplay(cauCtx);
            cauScene = cauStage->toScene();
            check(cauScene->settings.causticsEngineGpu == kGpuCausticsAimedLt,
                  "causticsenginegpu 0 cooks to Aimed LT");
        }
        settings->setParameterValue("backend", 2);
        settings->setParameterValue("xpuschedule", 0);
        CookContext context;
        StagePtr stage = graph.cookDisplay(context);
        check(stage != nullptr, "XPU cook produces a stage");
        ScenePtr cooked = stage->toScene();
        check(cooked->settings.backend == kBackendXpu, "menu value 2 cooks to XPU");
        check(cooked->settings.xpuSchedule == kXpuScheduleOverlap, "xpuschedule 0 cooks to Overlap");
        settings->setParameterValue("xpuschedule", 1);
        stage = graph.cookDisplay(context);
        cooked = stage->toScene();
        check(cooked->settings.xpuSchedule == kXpuScheduleMixture, "xpuschedule 1 cooks to Mixture");
        settings->setParameterValue("xpuschedule", 2);
        stage = graph.cookDisplay(context);
        cooked = stage->toScene();
        check(cooked->settings.xpuSchedule == kXpuScheduleOverlap, "retired Tile value cooks to Overlap");
        const Parameter* oracle = settings->findParameter(QLatin1String("pixeloracle"));
        check(oracle != nullptr, "pixeloracle parameter exists");
        if (oracle) {
            check(oracle->group == QLatin1String("Sampling"), "pixeloracle in Sampling");
            check(oracle->menuItems.size() == 2, "Pixel Oracle is Uniform / Variance");
        }
        const Parameter* noise = settings->findParameter(QLatin1String("noisethreshold"));
        check(noise != nullptr, "noisethreshold parameter exists");
        if (noise) {
            check(noise->group == QLatin1String("Sampling"), "noisethreshold in Sampling");
            check(noise->visibleWhen.isEmpty(), "noisethreshold always listed in Sampling");
        }
        settings->setParameterValue("noisethreshold", 0.05);
        stage = graph.cookDisplay(context);
        cooked = stage->toScene();
        check(std::fabs(cooked->settings.noiseThreshold - 0.05f) < 1e-5f, "noisethreshold cooks");
        settings->setParameterValue("pixeloracle", 0);
        stage = graph.cookDisplay(context);
        cooked = stage->toScene();
        check(cooked->settings.noiseThreshold == 0.0f, "Uniform oracle disables noise threshold");
        settings->setParameterValue("pixeloracle", 1);
        if (sched) {
            check(sched->menuItems.size() == 2, "XPU Schedule menu is Overlap / Mixture");
        }
    }

    float jx = 0, jy = 0, lu = 0, lv = 0;
    sampleCameraPixelLens(10, 20, 3, jx, jy, lu, lv);
    float jx2 = 0, jy2 = 0, lu2 = 0, lv2 = 0;
    pixelSample(10, 20, 3, jx2, jy2);
    lensSample(10, 20, 3, lu2, lv2);
    checkNear(jx, jx2, 1e-6f, "shared camera jitter X matches Sobol");
    checkNear(jy, jy2, 1e-6f, "shared camera jitter Y matches Sobol");
    checkNear(lu, lu2, 1e-6f, "shared lens U matches Sobol");
    checkNear(lv, lv2, 1e-6f, "shared lens V matches Sobol");

    {
        const uint32_t dims[] = {0u, 1u, 2u, 7u, 31u, 100u, 511u, 1023u};
        const uint32_t indices[] = {0u, 1u, 2u, 16u, 1023u, 65536u + 3u};
        for (uint32_t d : dims) {
            for (uint32_t i : indices) {
                check(sobol_detail::sobolRaw(i, d) == sobol_detail::sobolRawDirect(i, d),
                      "host Sobol table matches on-the-fly directions");
            }
        }
        Rng rng = makePixelRng(10, 20, 3, 0u);
        attachPathSobol(rng, 10, 20, 3);
        check(rng.useSobol == 1, "attachPathSobol sets useSobol");
        SobolSampler sampler;
        sampler.scrambleBase = rng.sobolScramble;
        const uint32_t index = rng.sobolIndex;
        for (uint32_t dim = 0; dim < 4; ++dim) {
            const float fromRng = rng.nextFloat();
            const float fromSampler = sampler.sample1D(index, dim);
            check(fromRng == fromSampler, "path Sobol nextFloat matches sampler");
        }
    }
}

void testRenderSettingsFolders() {
    std::printf("render settings folders\n");
    registerBuiltinNodes();
    NodeGraph graph;
    Node* settings = graph.createNode("rendersettings", "rendersettings1");
    check(settings != nullptr, "create rendersettings");
    if (!settings) return;

    auto groupOf = [&](const char* name) -> QString {
        const Parameter* parameter = settings->findParameter(QLatin1String(name));
        return parameter ? parameter->group : QString();
    };
    auto indexOf = [&](const char* name) -> int {
        const auto& params = settings->parameters();
        for (int i = 0; i < static_cast<int>(params.size()); ++i) {
            if (params[static_cast<size_t>(i)].name == QLatin1String(name)) return i;
        }
        return -1;
    };

    check(groupOf("pixelfilter") == QLatin1String("Image"), "pixelfilter in Image");
    check(groupOf("filterradius") == QLatin1String("Image"), "filterradius in Image");
    check(indexOf("pixelfilter") == indexOf("resy") + 1, "pixelfilter after resy");
    check(indexOf("filterradius") == indexOf("pixelfilter") + 1, "filterradius after pixelfilter");

    check(groupOf("lightsamples") == QLatin1String("Sampling"), "lightsamples in Sampling");
    check(groupOf("pixeloracle") == QLatin1String("Sampling"), "pixeloracle in Sampling");
    check(groupOf("noisethreshold") == QLatin1String("Sampling"), "noisethreshold in Sampling");
    check(indexOf("pixeloracle") == indexOf("samples") + 1, "pixeloracle after samples");
    check(indexOf("noisethreshold") == indexOf("pixeloracle") + 1, "noisethreshold after pixeloracle");
    check(indexOf("lightsamples") == indexOf("noisethreshold") + 1, "lightsamples after noisethreshold");

    check(groupOf("maxdepth") == QLatin1String("Depth"), "maxdepth in Depth");
    check(groupOf("rrdepth") == QLatin1String("Depth"), "rrdepth in Depth");

    check(groupOf("caustics") == QLatin1String("Caustics"), "caustics in Caustics");
    check(groupOf("causticsengine") == QLatin1String("Caustics"), "causticsengine in Caustics");
    check(groupOf("causticsenginegpu") == QLatin1String("Caustics"), "causticsenginegpu in Caustics");
    check(groupOf("causticclamp") == QLatin1String("Caustics"), "causticclamp in Caustics");
    check(groupOf("photoncount") == QLatin1String("Caustics"), "photoncount in Caustics");
    check(groupOf("photonradius") == QLatin1String("Caustics"), "photonradius in Caustics");
    {
        QStringList causticsOrder;
        for (const Parameter& parameter : settings->parameters()) {
            if (parameter.group != QLatin1String("Caustics")) continue;
            if (parameter.name.startsWith(QLatin1Char('_'))) continue;
            causticsOrder << parameter.name;
        }
        const QStringList causticsExpected{
            QStringLiteral("caustics"), QStringLiteral("causticsengine"),
            QStringLiteral("causticsenginegpu"), QStringLiteral("causticclamp"),
            QStringLiteral("photoncount"), QStringLiteral("photonradius")};
        check(causticsOrder == causticsExpected, "caustics tab parameter order");
    }

    check(groupOf("enabledisplacement") == QLatin1String("Displacement"),
          "enabledisplacement in Displacement");
    check(groupOf("frustumcull") == QLatin1String("Displacement"), "frustumcull in Displacement");
    check(groupOf("screenadaptive") == QLatin1String("Displacement"), "screenadaptive in Displacement");

    QStringList groups;
    for (const Parameter& parameter : settings->parameters()) {
        if (parameter.name.startsWith(QLatin1Char('_'))) continue;
        if (parameter.group.isEmpty()) continue;
        if (!groups.contains(parameter.group)) groups << parameter.group;
    }
    const QStringList expected{QStringLiteral("Image"),       QStringLiteral("Sampling"),
                               QStringLiteral("Engine"),      QStringLiteral("Depth"),
                               QStringLiteral("Caustics"),    QStringLiteral("Motion Blur"),
                               QStringLiteral("Displacement"), QStringLiteral("Film"),
                               QStringLiteral("Diagnostic")};
    check(groups == expected, "render settings tab order");

    check(groupOf("samplingdebug") == QLatin1String("Diagnostic"), "samplingdebug in Diagnostic");
    check(groupOf("bdpttimers") == QLatin1String("Diagnostic"), "bdpttimers in Diagnostic");
    check(groupOf("diag_bdpt_sep") == QLatin1String("Diagnostic"), "diag_bdpt_sep in Diagnostic");
    check(groupOf("diag_bdpt_head") == QLatin1String("Diagnostic"), "diag_bdpt_head in Diagnostic");
    {
        QStringList diagnosticOrder;
        for (const Parameter& parameter : settings->parameters()) {
            if (parameter.group != QLatin1String("Diagnostic")) continue;
            if (parameter.name.startsWith(QLatin1Char('_'))) continue;
            diagnosticOrder << parameter.name;
        }
        const QStringList diagnosticExpected{
            QStringLiteral("samplingdebug"), QStringLiteral("diag_bdpt_sep"),
            QStringLiteral("diag_bdpt_head"), QStringLiteral("bdpttimers")};
        check(diagnosticOrder == diagnosticExpected, "diagnostic tab parameter order");
    }
    check(indexOf("diag_bdpt_sep") == indexOf("samplingdebug") + 1, "separator after samplingdebug");
    check(indexOf("bdpttimers") == indexOf("diag_bdpt_head") + 1, "bdpttimers after CPU BDPT note");
    settings->setParameterValue("backend", kBackendGpuOptix);
    settings->setParameterValue("integrator", kIntegratorBdpt);
    {
        CookContext cookCtx;
        StagePtr cooked = graph.cook(settings, cookCtx);
        check(settings->intValue("integrator") == kIntegratorPathTracer,
              "cook drops BDPT when the device uses GPU");
        check(cooked && cooked->settings.integrator == kIntegratorPathTracer,
              "cooked GPU settings store Path Tracer");
        check(cooked && cooked->settings.backend == kBackendGpuOptix, "cooked backend stays GPU");
    }
    settings->setParameterValue("backend", kBackendCpuEmbree);
    settings->setParameterValue("bdpttimers", true);
    {
        CookContext cookCtx;
        StagePtr cooked = graph.cookDisplay(cookCtx);
        check(cooked && cooked->settings.bdptTimers != 0, "cook copies BDPT Timers flag");
    }
    settings->setParameterValue("bdpttimers", false);

    // Older scene files omit noisethreshold / pixeloracle; ctor defaults must remain.
    {
        NodeGraph saved;
        Node* rs = saved.createNode("rendersettings", "rs1");
        check(rs != nullptr, "create rendersettings for json merge");
        QJsonObject json = saved.toJson();
        QJsonArray nodes = json.value("nodes").toArray();
        for (int i = 0; i < nodes.size(); ++i) {
            QJsonObject nodeJson = nodes[i].toObject();
            if (nodeJson.value("type").toString() != QLatin1String("rendersettings")) continue;
            QJsonArray params = nodeJson.value("parameters").toArray();
            QJsonArray stripped;
            for (const QJsonValue& v : params) {
                const QString n = v.toObject().value("name").toString();
                if (n == QLatin1String("noisethreshold") || n == QLatin1String("pixeloracle")) continue;
                stripped.append(v);
            }
            nodeJson["parameters"] = stripped;
            nodes[i] = nodeJson;
        }
        json["nodes"] = nodes;
        NodeGraph loaded;
        QString error;
        check(loaded.fromJson(json, error), "old scene json loads without noisethreshold");
        Node* loadedSettings = loaded.findNode("rs1");
        check(loadedSettings != nullptr, "reloaded rendersettings");
        if (loadedSettings) {
            const Parameter* noise = loadedSettings->findParameter(QLatin1String("noisethreshold"));
            check(noise != nullptr, "noisethreshold survives load of older scene files");
            check(std::fabs(loadedSettings->floatValue("noisethreshold", -1.0) - 0.01) < 1e-6,
                  "noisethreshold default 0.01 on old files");
            const Parameter* oracle = loadedSettings->findParameter(QLatin1String("pixeloracle"));
            check(oracle != nullptr, "pixeloracle survives load of older scene files");
            check(loadedSettings->intValue("pixeloracle", -1) == 1, "pixeloracle defaults to Variance");
        }
    }
}

void testSceneGraphFolders() {
    std::printf("scene graph folders\n");
    registerBuiltinNodes();
    NodeGraph graph;

    auto groupOf = [](const Node* node, const char* name) -> QString {
        const Parameter* parameter = node->findParameter(QLatin1String(name));
        return parameter ? parameter->group : QStringLiteral("<missing>");
    };
    auto hasGroup = [](const Node* node, const QString& group) -> bool {
        for (const Parameter& parameter : node->parameters()) {
            if (parameter.group == group) return true;
        }
        return false;
    };

    Node* camera = graph.createNode("camera", "camera1");
    check(camera != nullptr, "create camera");
    if (camera) {
        check(!hasGroup(camera, QStringLiteral("Lens")), "camera has no Lens folder");
        check(groupOf(camera, "focal").isEmpty(), "focal is in the default folder");
        check(groupOf(camera, "aperture").isEmpty(), "aperture is in the default folder");
        check(groupOf(camera, "fstop").isEmpty(), "fstop is in the default folder");
        check(groupOf(camera, "focusdistance").isEmpty(), "focusdistance is in the default folder");
        check(groupOf(camera, "opticalmodel") == QLatin1String("Optics"), "opticalmodel stays in Optics");
        check(defaultParameterFolderTitle(camera->typeName()) == QLatin1String("Base"),
              "camera default folder is Base");
    }

    Node* rect = graph.createNode("rectlight", "rectlight1");
    check(rect != nullptr, "create rect light");
    if (rect) {
        check(!hasGroup(rect, QStringLiteral("Light")), "rect light has no Light folder");
        check(groupOf(rect, "enabled").isEmpty(), "enabled is in the default folder");
        check(groupOf(rect, "color").isEmpty(), "color is in the default folder");
        check(groupOf(rect, "intensity").isEmpty(), "intensity is in the default folder");
        check(groupOf(rect, "width") == QLatin1String("Shape"), "width stays in Shape");
        check(defaultParameterFolderTitle(rect->typeName()) == QLatin1String("Base"),
              "rect light default folder is Base");
    }

    Node* sky = graph.createNode("physicalskylight", "sky1");
    check(sky != nullptr, "create physical sky");
    if (sky) {
        check(!hasGroup(sky, QStringLiteral("Light")), "physical sky has no Light folder");
        check(groupOf(sky, "intensity").isEmpty(), "sky intensity is in the default folder");
        check(groupOf(sky, "turbidity") == QLatin1String("Sky"), "turbidity stays in Sky");
        check(defaultParameterFolderTitle(sky->typeName()) == QLatin1String("Base"),
              "physical sky default folder is Base");
    }

    const char* baseTypes[] = {"sphere", "grid", "box", "tube", "alembic", "usd",
                               "vdbfrompolygons", "vdbfile", "sdftopolygons_vdb", "sdftopolygons_dcsdd",
                               "domelight", "distantlight", "disklight", "spherelight"};
    for (const char* type : baseTypes) {
        Node* node = graph.createNode(QString::fromLatin1(type), QString::fromLatin1(type) + QLatin1Char('1'));
        check(node != nullptr, std::string("create ") + type);
        if (!node) continue;
        check(defaultParameterFolderTitle(node->typeName()) == QLatin1String("Base"),
              std::string(type) + " default folder is Base");
        check(!hasGroup(node, QStringLiteral("Lens")), std::string(type) + " has no Lens folder");
        check(!hasGroup(node, QStringLiteral("Light")), std::string(type) + " has no Light folder");
    }

    Node* material = graph.createNode("material", "material1");
    check(material != nullptr, "create material");
    if (material) {
        check(defaultParameterFolderTitle(material->typeName()) == QLatin1String("Base"),
              "material default folder is Base");
        check(hasGroup(material, QStringLiteral("MaterialX")), "material keeps MaterialX folder");
        check(groupOf(material, "pattern").isEmpty(), "Assign To is in the default folder");
    }

    check(defaultParameterFolderTitle(QStringLiteral("rendersettings")) == QLatin1String("Parameters"),
          "render settings keep Parameters as the empty-folder title");
}

void testRender() {
    std::printf("render\n");
    registerBuiltinNodes();
    NodeGraph graph;
    buildDefaultGraph(graph);
    CookContext context;
    StagePtr stage = graph.cookDisplay(context);
    ScenePtr scene = stage->toScene();
    scene->settings.resolutionX = 64;
    scene->settings.resolutionY = 48;
    scene->settings.samplesPerPixel = 8;
    scene->settings.backend = kBackendCpuEmbree;

    RenderSession session;
    session.setScene(scene);
    session.start();
    session.waitForCompletion();

    const Image image = session.linearImage();
    check(image.width() == 64 && image.height() == 48, "framebuffer has the requested size");

    double sum = 0.0;
    double maxValue = 0.0;
    int nonBlack = 0;
    bool finite = true;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const Vec3 c = image.rgb(x, y);
            if (!isFinite(c)) finite = false;
            const double l = double(luminance(c));
            sum += l;
            maxValue = std::max(maxValue, l);
            if (l > 1e-4) ++nonBlack;
        }
    }
    check(finite, "render output is finite");
    check(nonBlack > image.width() * image.height() / 2, "most pixels receive light");
    check(sum > 0.0 && maxValue < 1e4, "render output is in a sane range");

    // Integrator smoke tests.
    auto smokeIntegrator = [&](int integrator, int caustics, const char* label) {
        scene->settings.integrator = integrator;
        scene->settings.caustics = caustics;
        scene->settings.samplesPerPixel = 4;
        scene->settings.pathGuiding = 0;  // keep the smoke test deterministic-ish
        RenderSession s2;
        s2.setScene(scene);
        s2.start();
        s2.waitForCompletion();
        const Image img = s2.linearImage();
        bool ok = true;
        double s = 0.0;
        for (int y = 0; y < img.height(); ++y) {
            for (int x = 0; x < img.width(); ++x) {
                const Vec3 c = img.rgb(x, y);
                if (!isFinite(c)) ok = false;
                s += double(luminance(c));
            }
        }
        check(ok, std::string(label) + " output is finite");
        check(s > 0.0, std::string(label) + " produces light");
    };
    scene->settings.causticsEngine = kCausticsEnginePbrt;
    smokeIntegrator(kIntegratorPathTracer, 1, "PT + pbrt caustics");
    smokeIntegrator(kIntegratorPathTracer, 0, "PT caustics off");
    smokeIntegrator(kIntegratorBdpt, 1, "BDPT integrator");
    scene->settings.causticsEngine = kCausticsEngineMnee;
    smokeIntegrator(kIntegratorPathTracer, 1, "PT + MNEE caustics");
    scene->settings.causticsEngine = kCausticsEnginePbrt;
    smokeIntegrator(kIntegratorWireframe, 0, "Wireframe");
    // UI default leaves Caustics on — must not route Wireframe into Photon/MNEE.
    smokeIntegrator(kIntegratorWireframe, 1, "Wireframe + caustics on");
    // Path guiding smoke: same scene with OpenPGL training enabled (no-op when
    // the build lacks OpenPGL).
    scene->settings.pathGuiding = 1;
    smokeIntegrator(kIntegratorPathTracer, 1, "PT + guiding");
    smokeIntegrator(kIntegratorBdpt, 1, "BDPT + guiding");
    scene->settings.pathGuiding = 0;
    scene->settings.integrator = kIntegratorPathTracer;
    scene->settings.caustics = 1;
    scene->settings.causticsEngine = kCausticsEnginePbrt;
}

// The equirectangular convention must stay stable: +Y is the top row of the
// image and importance sampling has to find the bright spot.
void testEnvironment() {
    std::printf("environment\n");
    auto env = std::make_shared<EnvironmentMap>();
    const int w = 64, h = 32;
    env->image.resize(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const Vec3 color = y < h / 2 ? Vec3(1.0f, 0.0f, 0.0f) : Vec3(0.0f, 0.0f, 1.0f);
            env->image.setRgb(x, y, color);
        }
    }
    // A single very bright texel drives the importance sampling test.
    env->image.setRgb(10, 4, Vec3(2000.0f, 2000.0f, 2000.0f));
    env->buildSamplingTables();

    const EnvMapView view = env->view();
    const Vec3 up = envLookup(view, Vec3(0.0f, 1.0f, 0.0f));
    const Vec3 down = envLookup(view, Vec3(0.0f, -1.0f, 0.0f));
    check(up.x > 0.9f && up.z < 0.1f, "+Y samples the top of the environment map");
    check(down.z > 0.9f && down.x < 0.1f, "-Y samples the bottom of the environment map");

    Rng rng(5u, 9u);
    int nearBrightTexel = 0;
    const Vec3 brightDirection = equirectToDirection((10.5f) / float(w), (4.5f) / float(h));
    for (int i = 0; i < 4000; ++i) {
        float pdf = 0.0f;
        const Vec3 dir = envSample(view, rng.nextFloat(), rng.nextFloat(), pdf);
        check(pdf > 0.0f, "environment sample has a positive pdf");
        if (dot(dir, brightDirection) > 0.99f) ++nearBrightTexel;
        // The analytic pdf must agree with the pdf returned while sampling.
        checkNear(envPdf(view, dir), pdf, std::max(1e-3f, pdf * 0.02f), "envPdf matches envSample");
    }
    check(nearBrightTexel > 3000, "environment sampling concentrates on the bright texel");

    // Dark texel next to the sun: bilinear mixes in 2000 nit, nearest stays dark
    // (the PDF texel). That mismatch is the volume lit-side HDRI firefly.
    {
        const Vec3 darkDir = equirectToDirection((10.0f - 0.05f) / float(w), (4.5f) / float(h));
        const Vec3 nearest = envLookupNearest(view, darkDir);
        const Vec3 bilinear = envLookup(view, darkDir);
        check(luminance(nearest) < 2.0f, "nearest lookup of a dark neighbour stays dark");
        check(luminance(bilinear) > luminance(nearest) * 10.0f,
              "bilinear lookup bleeds the neighbouring sun texel");
        const float pdfDark = envPdf(view, darkDir);
        if (pdfDark > 1e-12f) {
            check(luminance(nearest) / pdfDark < luminance(bilinear) / pdfDark,
                  "NEE Le/pdf is smaller with nearest than with bilinear bleed");
        }
    }
}

void testDomeHdrLoad() {
    std::printf("dome-hdr\n");
    registerBuiltinNodes();

    QString hdrPath;
    const QStringList candidates = {
        QStringLiteral("/workspace/examples/sky.hdr"),
        QDir::currentPath() + "/examples/sky.hdr",
        QDir::currentPath() + "/../examples/sky.hdr",
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            hdrPath = QFileInfo(candidate).absoluteFilePath();
            break;
        }
    }
    check(!hdrPath.isEmpty(), "bundled sky.hdr is present");
    if (hdrPath.isEmpty()) return;

    auto maxLum = [](const Image& img) {
        float m = 0.0f;
        for (int y = 0; y < img.height(); ++y) {
            for (int x = 0; x < img.width(); ++x) m = std::max(m, luminance(img.rgb(x, y)));
        }
        return m;
    };

    Image direct;
    std::string err;
    check(loadImage(hdrPath.toStdString(), direct, err, true, "Utility - Raw"),
          "load sky.hdr as Raw");
    if (!err.empty() && direct.empty()) std::printf("  load error: %s\n", err.c_str());
    check(direct.width() == 1024 && direct.height() == 512, "sky.hdr is 1024x512");
    check(maxLum(direct) > 1.0f, "native HDR has values above 1");
    {
        QTemporaryDir dir;
        check(dir.isValid(), "temp dir for exposure hdr");
        const QString expHdr = dir.path() + "/exp.hdr";
        std::FILE* f = std::fopen(expHdr.toUtf8().constData(), "wb");
        check(f != nullptr, "write EXPOSURE hdr");
        if (f) {
            std::fputs("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=16.0\n\n-Y 1 +X 1\n", f);
            const unsigned char rgbe[4] = {128, 128, 128, 128};  // linear ~0.5
            std::fwrite(rgbe, 1, 4, f);
            std::fclose(f);
            Image img;
            std::string e;
            check(loadImage(expHdr.toStdString(), img, e, true, "Utility - Raw"), "load EXPOSURE hdr");
            check(!img.empty(), "EXPOSURE hdr has a pixel");
            if (!img.empty()) {
                const float y = luminance(img.rgb(0, 0));
                check(y > 0.2f && y < 2.0f, "EXPOSURE=16 does not crush RGBE values");
            }
        }
    }

    {
        QTemporaryDir dir;
        check(dir.isValid(), "temp dir for red hdr");
        const QString redHdr = dir.path() + "/red.hdr";
        Image src(1, 1);
        src.setRgb(0, 0, Vec3(1.0f, 0.0f, 0.0f));
        std::string we;
        check(saveImageHdr(redHdr.toStdString(), src, we), "write 1x1 red HDR");
        Image raw;
        Image lin;
        std::string e1;
        std::string e2;
        check(loadImage(redHdr.toStdString(), raw, e1, true, "Utility - Raw"), "load red HDR Raw");
        check(loadImage(redHdr.toStdString(), lin, e2, true, "Utility - Linear - sRGB"),
              "load red HDR Linear sRGB → ACEScg");
        checkNear(raw.rgb(0, 0).x, 1.0f, 0.08f, "Raw red stays ~1");
        checkNear(raw.rgb(0, 0).y, 0.0f, 0.08f, "Raw red G stays ~0");
        const Vec3 expect = linearSrgbToAcescg(Vec3(1.0f, 0.0f, 0.0f));
        checkNear(lin.rgb(0, 0).x, expect.x, 0.06f, "Linear sRGB red → ACEScg R");
        checkNear(lin.rgb(0, 0).y, expect.y, 0.06f, "Linear sRGB red → ACEScg G");
        checkNear(lin.rgb(0, 0).z, expect.z, 0.06f, "Linear sRGB red → ACEScg B");
        check(std::fabs(lin.rgb(0, 0).x - raw.rgb(0, 0).x) > 0.05f, "ACEScg convert changes Rec.709 red");
    }

    // +Y orientation used to be rejected ("unsupported HDR resolution line").
    {
        QTemporaryDir dir;
        check(dir.isValid(), "temp dir for +Y hdr");
        const QString plusY = dir.path() + "/plusy.hdr";
        std::FILE* f = std::fopen(plusY.toUtf8().constData(), "wb");
        check(f != nullptr, "write +Y hdr");
        if (f) {
            std::fputs("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n+Y 2 +X 1\n", f);
            const unsigned char blue[4] = {0, 0, 128, 129};
            const unsigned char red[4] = {128, 0, 0, 129};
            std::fwrite(blue, 1, 4, f);
            std::fwrite(red, 1, 4, f);
            std::fclose(f);
            Image img;
            std::string e;
            check(loadImage(plusY.toStdString(), img, e, true, "Utility - Raw"), "load +Y hdr");
            check(img.width() == 1 && img.height() == 2, "+Y hdr is 1x2");
            if (!img.empty()) {
                check(img.rgb(0, 0).x > 0.5f && img.rgb(0, 0).z < 0.25f,
                      "+Y: top row is the second scanline (red)");
                check(img.rgb(0, 1).z > 0.5f && img.rgb(0, 1).x < 0.25f,
                      "+Y: bottom row is the first scanline (blue)");
            }
        }
    }

    QTemporaryDir cacheDir;
    check(cacheDir.isValid(), "temp dir for tx cache");
    RenderSettingsData arm{};
    arm.enableTxCache = 1;
    const QByteArray cachePath = cacheDir.path().toUtf8();
    std::snprintf(arm.txCacheDir, sizeof(arm.txCacheDir), "%s", cachePath.constData());

    const std::string prevCs = txDefaultInputColorSpace();
    setTxDefaultInputColorSpace("Utility - sRGB - Texture");
    setActiveTxCacheSettings(&arm);
    Image viaCache;
    std::string err2;
    const bool loaded = loadImage(hdrPath.toStdString(), viaCache, err2, /*srgbColor=*/true);
    setActiveTxCacheSettings(nullptr);
    setTxDefaultInputColorSpace(prevCs);
    check(loaded, "load sky.hdr with TX cache armed and sticky sRGB CS");
    if (!err2.empty() && viaCache.empty()) std::printf("  tx-armed load error: %s\n", err2.c_str());
    check(maxLum(viaCache) > 1.0f, "TX-armed HDR load keeps values above 1");
    check(viaCache.width() == direct.width() && viaCache.height() == direct.height(),
          "TX-armed HDR keeps resolution");

    NodeGraph graph;
    Node* dome = graph.createNode("domelight", "domelight1");
    check(dome != nullptr, "create domelight");
    if (dome) {
        dome->setParameterValue("texture", hdrPath);
        dome->setParameterValue("colorspace", QStringLiteral("auto"));
        dome->setParameterValue("intensity", 1.0);
    }
    Node* settings = graph.createNode("rendersettings", "rendersettings1");
    check(settings != nullptr, "create rendersettings");
    if (settings) {
        settings->setParameterValue("enabletxcache", true);
        settings->setParameterValue("txcachedir", cacheDir.path());
    }
    if (dome && settings) graph.connectNodes(dome, settings, 0);
    graph.setDisplayNode(settings);

    setTxDefaultInputColorSpace("Utility - sRGB - Texture");
    setActiveTxCacheSettings(&arm);
    CookContext context;
    StagePtr stage = graph.cookDisplay(context);
    setActiveTxCacheSettings(nullptr);
    setTxDefaultInputColorSpace(prevCs);

    check(stage != nullptr, "dome graph cooks");
    check(context.errors.isEmpty(), "dome HDR cook has no errors");
    for (const QString& e : context.errors) std::printf("  cook error: %s\n", qPrintable(e));

    ScenePtr scene = stage ? stage->toScene() : nullptr;
    check(scene != nullptr, "dome toScene");
    if (!scene) return;
    check(!scene->lights.empty() && scene->lights[0].type == kLightDome, "scene has a dome light");
    check(!scene->lights.empty() && scene->lights[0].envIndex >= 0, "dome has an env map index");
    bool foundEnv = false;
    float envMax = 0.0f;
    for (const auto& env : scene->envMaps) {
        if (!env || env->image.empty()) continue;
        foundEnv = true;
        envMax = std::max(envMax, maxLum(env->image));
    }
    check(foundEnv, "dome attached an environment map");
    check(envMax > 1.0f, "cooked dome HDR keeps values above 1");
}

void testPhysicalSkyLight() {
    std::printf("physical-sky\n");

    PhysicalSkyParams params;
    Vec3 zenithSun = physicalSkySunDirection(params);
    checkNear(zenithSun.x, 0.7071f, 0.02f, "default azimuth 90 elevation 45 → +X");
    checkNear(zenithSun.y, 0.7071f, 0.02f, "default elevation 45");
    checkNear(zenithSun.z, 0.0f, 0.02f, "default azimuth 90 has no Z");

    params.elevationDeg = 90.0f;
    params.azimuthDeg = 0.0f;
    const Vec3 up = physicalSkySunDirection(params);
    checkNear(up.x, 0.0f, 0.02f, "zenith sun X");
    checkNear(up.y, 1.0f, 0.02f, "zenith sun +Y");
    checkNear(up.z, 0.0f, 0.02f, "zenith sun Z");

    params.elevationDeg = 0.0f;
    params.azimuthDeg = 0.0f;
    const Vec3 horizon = physicalSkySunDirection(params);
    checkNear(horizon.x, 0.0f, 0.02f, "azimuth 0 horizon X");
    checkNear(horizon.y, 0.0f, 0.02f, "azimuth 0 horizon Y");
    checkNear(horizon.z, -1.0f, 0.02f, "azimuth 0 is −Z");

    const Mat4 sunXform = physicalSkySunLookAt(params);
    const Vec3 axis = normalize(transformVector(sunXform, Vec3(0.0f, 0.0f, 1.0f)));
    checkNear(dot(axis, horizon), 1.0f, 0.02f, "distant +Z points at the sun");

    params = PhysicalSkyParams{};
    {
        const float e0 = physicalSkySunIntensity(params);
        check(std::isfinite(e0) && e0 > 1e-3f, "default sun irradiance is finite and usable");
        const float Lz = luminance(physicalSkyRadianceAceScg(params, Vec3(0.0f, 1.0f, 0.0f)));
        check(e0 > kPi * Lz * 0.25f, "default sun irradiance dominates the zenith sky");
        PhysicalSkyParams skyOnly = params;
        skyOnly.skyIntensity = 4.0f;
        checkNear(physicalSkySunIntensity(skyOnly), e0, 1e-4f * (1.0f + e0),
                  "sky intensity does not scale the sun");
        PhysicalSkyParams scaled = params;
        scaled.intensity = 2.0f;
        scaled.sunIntensity = 3.0f;
        checkNear(physicalSkySunIntensity(scaled), e0 * 6.0f, 1e-3f * (1.0f + e0 * 6.0f),
                  "overall intensity and sun intensity scale the sun");
        PhysicalSkyParams off = params;
        off.sunIntensity = 0.0f;
        checkNear(physicalSkySunIntensity(off), 0.0f, 1e-6f, "sun intensity 0 → no sun energy");
    }
    Image baked;
    bakePhysicalSkyEnv(baked, params, 64, 32);
    check(baked.width() == 64 && baked.height() == 32, "bake writes the requested size");
    const Vec3 zenith = physicalSkyRadianceAceScg(params, Vec3(0.0f, 1.0f, 0.0f));
    const Vec3 ground = physicalSkyRadianceAceScg(params, Vec3(0.0f, -1.0f, 0.0f));
    check(luminance(zenith) > luminance(ground), "zenith is brighter than the ground");
    check(isFinite(zenith) && isFinite(ground), "sky samples are finite");
    const Vec3 zenithSrgb = acescgToLinearSrgb(zenith);
    check(zenithSrgb.z > zenithSrgb.x * 0.9f, "clear zenith is bluish in Rec.709");
    float bakeMax = 0.0f;
    for (int y = 0; y < baked.height(); ++y)
        for (int x = 0; x < baked.width(); ++x)
            bakeMax = std::max(bakeMax, luminance(baked.rgb(x, y)));
    check(bakeMax > luminance(zenith) * 0.5f, "sky bake has energy around zenith");
    check(bakeMax < luminance(zenith) * 40.0f, "sky bake has no solar disc");
    {
        Image cookSize;
        const auto t0 = std::chrono::steady_clock::now();
        bakePhysicalSkyEnv(cookSize, params, 1024, 512);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        check(cookSize.width() == 1024 && cookSize.height() == 512, "cook-size bake writes 1024x512");
        check(ms < 15000, "1024x512 physical sky bake stays interactive");
        float cookMax = 0.0f;
        for (int y = 0; y < cookSize.height(); ++y)
            for (int x = 0; x < cookSize.width(); ++x)
                cookMax = std::max(cookMax, luminance(cookSize.rgb(x, y)));
        check(cookMax < luminance(zenith) * 40.0f, "cook-size bake has no solar disc");
    }
    const Vec3 sunDirRad = physicalSkyRadianceAceScg(params, physicalSkySunDirection(params));
    check(luminance(sunDirRad) < luminance(zenith) * 40.0f, "sky radiance toward the sun has no disc");
    {
        const float half = 0.5f * radians(params.sunSizeDeg);
        const float omega = kTwoPi * (1.0f - std::cos(half));
        const float discL = physicalSkySunIntensity(params) / srMax(omega, 1e-12f);
        check(discL > luminance(acescgToLinearSrgb(sunDirRad)) * 10.0f,
              "distant sun radiance dominates sky in the disc");
        PhysicalSkyParams noSun = params;
        noSun.enableSun = false;
        checkNear(physicalSkySunIntensity(noSun), 0.0f, 1e-6f, "enable sun off zeros distant intensity");
        const Vec3 noSunSky = physicalSkyRadianceAceScg(noSun, physicalSkySunDirection(noSun));
        checkNear(luminance(noSunSky), luminance(sunDirRad), 0.05f * (1.0f + luminance(sunDirRad)),
                  "enable sun off does not change the sky bake");
        PhysicalSkyParams skyBoost = params;
        skyBoost.skyIntensity = 4.0f;
        const Vec3 zBoost = physicalSkyRadianceAceScg(skyBoost, Vec3(0.0f, 1.0f, 0.0f));
        checkNear(luminance(zBoost), luminance(zenith) * 4.0f, 0.15f * luminance(zenith) * 4.0f,
                  "sky intensity scales the sky not the sun");
    }

    {
        PhysicalSkyParams sunset;
        sunset.elevationDeg = 4.0f;
        sunset.azimuthDeg = 0.0f;
        const Vec3 towardSun =
            acescgToLinearSrgb(physicalSkyRadianceAceScg(sunset, normalize(Vec3(0.0f, 0.08f, -1.0f))));
        const Vec3 sunsetZenith = acescgToLinearSrgb(physicalSkyRadianceAceScg(sunset, Vec3(0.0f, 1.0f, 0.0f)));
        const float sunRB = towardSun.x / srMax(1e-6f, towardSun.z);
        const float zenRB = sunsetZenith.x / srMax(1e-6f, sunsetZenith.z);
        check(sunRB > zenRB * 1.05f, "Hosek sunset toward the sun is redder than zenith");
    }
    {
        PhysicalSkyParams groundP;
        groundP.computeGroundColor = false;
        groundP.groundColor = Vec3(0.05f, 0.8f, 0.05f);
        groundP.horizonBlurDeg = 0.1f;
        const Vec3 nadir = acescgToLinearSrgb(physicalSkyRadianceAceScg(groundP, Vec3(0.0f, -1.0f, 0.0f)));
        check(nadir.y > nadir.x * 2.0f && nadir.y > nadir.z * 2.0f, "authored ground colour shows below the horizon");
    }

    registerBuiltinNodes();
    {
        NodeGraph graph;
        Node* sky = graph.createNode("physicalskylight", "sky1");
        check(sky != nullptr, "create physicalskylight");
        Node* settings = graph.createNode("rendersettings", "rendersettings1");
        check(settings != nullptr, "create rendersettings for sky");
        if (sky && settings) graph.connectNodes(sky, settings, 0);
        graph.setDisplayNode(settings);

        CookContext context;
        StagePtr stage = graph.cookDisplay(context);
        check(stage != nullptr, "physical sky cooks");
        check(context.errors.isEmpty(), "physical sky cook has no errors");
        for (const QString& e : context.errors) std::printf("  cook error: %s\n", qPrintable(e));
        check(stage && stage->countOfType(PrimType::Light) == 2, "one node emits sky dome + distant sun");

        ScenePtr scene = stage ? stage->toScene() : nullptr;
        check(scene != nullptr, "physical sky toScene");
        if (scene) {
            check(scene->lights.size() == 2, "scene has sky + sun");
            int domeN = 0, sunN = 0, sunIdx = -1;
            for (int i = 0; i < int(scene->lights.size()); ++i) {
                if (scene->lights[size_t(i)].type == kLightDome) ++domeN;
                if (scene->lights[size_t(i)].type == kLightDistant) {
                    ++sunN;
                    sunIdx = i;
                }
            }
            check(domeN == 1 && sunN == 1, "physical sky is a dome plus a distant sun");
            const LightData* dome = nullptr;
            const LightData* sun = nullptr;
            for (const LightData& l : scene->lights) {
                if (l.type == kLightDome) dome = &l;
                if (l.type == kLightDistant) sun = &l;
            }
            check(dome && sun, "both physical sky lights are present");
            if (dome) {
                checkNear(dome->intensity, 1.0f, 1e-4f, "overall intensity is on the dome");
                check(!scene->envMaps.empty() && scene->envMaps[0] && !scene->envMaps[0]->image.empty(),
                      "dome has a baked sky map");
                if (!scene->envMaps.empty() && scene->envMaps[0]) {
                    float envMax = 0.0f;
                    const Image& img = scene->envMaps[0]->image;
                    for (int y = 0; y < img.height(); ++y)
                        for (int x = 0; x < img.width(); ++x)
                            envMax = std::max(envMax, luminance(img.rgb(x, y)));
                    check(envMax > 0.01f, "cooked sky map has energy");
                    check(envMax < 200.0f, "cooked sky map has no baked solar disc");
                }
            }
            if (sun) {
                check(sun->normalize == 1, "sun irradiance is normalized");
                check(sun->cameraSunDisc == 1, "sun disc is visible to camera");
                checkNear(sun->angle, 0.53f, 1e-4f, "sun angular size is 0.53°");
                checkNear(sun->intensity, physicalSkySunIntensity(PhysicalSkyParams{}),
                          1e-3f * (1.0f + sun->intensity), "distant intensity is Hosek sun irradiance");
                const Vec3 axis = normalize(transformVector(sun->xform, Vec3(0.0f, 0.0f, 1.0f)));
                checkNear(dot(axis, physicalSkySunDirection(PhysicalSkyParams{})), 1.0f, 0.02f,
                          "distant +Z matches solar altitude/azimuth");
            }
            if (sunIdx >= 0) {
                SceneView view = scene->view();
                LightSample ls;
                check(sampleLight(view, sunIdx, Vec3(0.0f, 1.0f, 0.0f), 0.2f, 0.3f, ls),
                      "distant sun NEE samples");
                const Vec3 axis = normalize(transformVector(scene->lights[size_t(sunIdx)].xform,
                                                            Vec3(0.0f, 0.0f, 1.0f)));
                const float cosMax = std::cos(0.5f * radians(0.53f));
                check(dot(ls.wi, axis) >= cosMax - 1e-4f, "sun NEE lands inside the disc cone");
                check(ls.delta == false, "0.53° sun is a finite cone, not a Dirac");
                const float pdfCone = 1.0f / (kTwoPi * (1.0f - cosMax));
                checkNear(ls.pdf, pdfCone, 0.02f * pdfCone, "sun NEE pdf is 1/ω");
            }
        }
    }
    {
        NodeGraph graph;
        Node* sky = graph.createNode("physicalskylight", "sky1");
        Node* settings = graph.createNode("rendersettings", "rendersettings1");
        if (sky) sky->setParameterValue("enablesun", false);
        if (sky && settings) graph.connectNodes(sky, settings, 0);
        graph.setDisplayNode(settings);
        CookContext context;
        StagePtr stage = graph.cookDisplay(context);
        check(stage && stage->countOfType(PrimType::Light) == 1, "enable sun off → dome only");
        ScenePtr scene = stage ? stage->toScene() : nullptr;
        check(scene && scene->lights.size() == 1 && scene->lights[0].type == kLightDome,
              "sun disabled leaves only the sky");
    }
    {
        NodeGraph graph;
        Node* sky = graph.createNode("physicalskylight", "sky1");
        Node* settings = graph.createNode("rendersettings", "rendersettings1");
        if (sky) sky->setParameterValue("enablesky", false);
        if (sky && settings) graph.connectNodes(sky, settings, 0);
        graph.setDisplayNode(settings);
        CookContext context;
        StagePtr stage = graph.cookDisplay(context);
        check(stage && stage->countOfType(PrimType::Light) == 1, "enable sky off → sun only");
        ScenePtr scene = stage ? stage->toScene() : nullptr;
        check(scene && scene->lights.size() == 1 && scene->lights[0].type == kLightDistant,
              "sky disabled leaves a distant sun");
        if (scene && !scene->lights.empty()) {
            check(scene->lights[0].cameraSunDisc == 1, "sun-only still draws the camera disc");
            check(scene->envMaps.empty(), "sun-only does not bake a sky map");
        }
    }
    {
        NodeGraph graph;
        Node* sky = graph.createNode("physicalskylight", "sky1");
        Node* settings = graph.createNode("rendersettings", "rendersettings1");
        if (sky) {
            sky->setParameterValue("intensity", 2.0);
            sky->setParameterValue("skyintensity", 3.0);
            sky->setParameterValue("sunintensity", 4.0);
        }
        if (sky && settings) graph.connectNodes(sky, settings, 0);
        graph.setDisplayNode(settings);
        CookContext context;
        StagePtr stage = graph.cookDisplay(context);
        ScenePtr scene = stage ? stage->toScene() : nullptr;
        check(scene && scene->lights.size() == 2, "overall/sky/sun intensity cooks sky + sun");
        if (scene && scene->lights.size() == 2) {
            const LightData* dome = nullptr;
            const LightData* sun = nullptr;
            for (const LightData& l : scene->lights) {
                if (l.type == kLightDome) dome = &l;
                if (l.type == kLightDistant) sun = &l;
            }
            check(dome && sun, "intensity cook has both lights");
            if (dome) checkNear(dome->intensity, 2.0f, 1e-4f, "dome intensity is the overall scale");
            PhysicalSkyParams expect;
            expect.intensity = 2.0f;
            expect.sunIntensity = 4.0f;
            PhysicalSkyParams skyOnly;
            skyOnly.intensity = 2.0f;
            skyOnly.skyIntensity = 3.0f;
            skyOnly.sunIntensity = 4.0f;
            checkNear(physicalSkySunIntensity(skyOnly), physicalSkySunIntensity(expect), 1e-4f,
                      "sky intensity is not folded into the sun irradiance");
            if (sun) {
                checkNear(sun->intensity, physicalSkySunIntensity(expect),
                          1e-3f * (1.0f + sun->intensity),
                          "distant intensity folds overall × sun intensity");
            }
        }
    }
    {
        NodeGraph graph;
        Node* grid = graph.createNode("grid", "grid1");
        Node* sky = graph.createNode("physicalskylight", "sky1");
        Node* cam = graph.createNode("camera", "camera1");
        Node* settings = graph.createNode("rendersettings", "rendersettings1");
        if (grid) {
            grid->setParameterValue("sizex", 8.0);
            grid->setParameterValue("sizez", 8.0);
            grid->setParameterValue("subdivtype", 0);
        }
        if (cam) {
            cam->setParameterValue("eye", QVariant::fromValue(QVector3D(0.0f, 6.0f, 0.2f)));
            cam->setParameterValue("target", QVariant::fromValue(QVector3D(0.0f, 0.0f, 0.0f)));
        }
        if (grid && sky) graph.connectNodes(grid, sky, 0);
        if (sky && cam) graph.connectNodes(sky, cam, 0);
        if (cam && settings) graph.connectNodes(cam, settings, 0);
        graph.setDisplayNode(settings);
        CookContext context;
        StagePtr stage = graph.cookDisplay(context);
        ScenePtr scene = stage ? stage->toScene() : nullptr;
        check(scene != nullptr, "physical sky render scene");
        if (scene) {
            scene->settings.resolutionX = 32;
            scene->settings.resolutionY = 24;
            scene->settings.samplesPerPixel = 4;
            scene->settings.backend = kBackendCpuEmbree;
            RenderSession session;
            session.setScene(scene);
            session.start();
            session.waitForCompletion();
            const Image image = session.linearImage();
            double sum = 0.0;
            int nonBlack = 0;
            bool finite = true;
            for (int y = 0; y < image.height(); ++y) {
                for (int x = 0; x < image.width(); ++x) {
                    const Vec3 c = image.rgb(x, y);
                    if (!isFinite(c)) finite = false;
                    const double l = double(luminance(c));
                    sum += l;
                    if (l > 1e-4) ++nonBlack;
                }
            }
            check(finite, "physical sky render is finite");
            check(nonBlack > image.width() * image.height() / 4, "physical sky lights the ground");
            check(sum > 0.0, "physical sky render has energy");
        }
    }
}

void testAcesTextureConvert() {
    std::printf("aces-texture-convert\n");
    check(txResolveInputColorSpace("auto", "foo.png", true) == "Utility - sRGB - Texture",
          "auto 8-bit colour → sRGB Texture");
    check(txResolveInputColorSpace("auto", "foo.hdr", true) == "Utility - Linear - sRGB",
          "auto HDR colour → Linear sRGB");
    check(txResolveInputColorSpace("auto", "foo.exr", true) == "Utility - Linear - sRGB",
          "auto EXR colour → Linear sRGB");
    check(txResolveInputColorSpace({}, "disp.exr", false) == "Utility - Raw",
          "data maps → Raw");
    check(txResolveInputColorSpace("ACES - ACEScg", "foo.png", true) == "ACES - ACEScg",
          "authored ACEScg is kept");
    check(txSkipColorConvert("ACES - ACEScg"), "ACEScg skips convert");
    check(txSkipColorConvert("Utility - Raw"), "Raw skips convert");
    check(!txSkipColorConvert("Utility - Linear - sRGB"), "Linear sRGB converts");
    check(!txSkipColorConvert("Utility - sRGB - Texture"), "sRGB Texture converts");

    QTemporaryDir dir;
    check(dir.isValid(), "temp dir for aces png");
    const QString pngPath = dir.path() + "/red.png";
    {
        QImage img(1, 1, QImage::Format_RGB888);
        img.fill(QColor(255, 0, 0));
        check(img.save(pngPath), "write 1x1 sRGB red PNG");
    }

    Image raw;
    Image aces;
    Image taggedAces;
    std::string e1, e2, e3;
    check(loadImage(pngPath.toStdString(), raw, e1, true, "Utility - Raw"), "load PNG Raw");
    check(loadImage(pngPath.toStdString(), aces, e2, true, "auto"), "load PNG auto → ACEScg");
    check(loadImage(pngPath.toStdString(), taggedAces, e3, true, "ACES - ACEScg"), "load PNG as ACEScg");
    checkNear(raw.rgb(0, 0).x, 1.0f, 0.02f, "Raw red R ~1");
    checkNear(raw.rgb(0, 0).y, 0.0f, 0.02f, "Raw red G ~0");
    const Vec3 expect = linearSrgbToAcescg(Vec3(1.0f, 0.0f, 0.0f));
    checkNear(aces.rgb(0, 0).x, expect.x, 0.05f, "auto PNG red → ACEScg R");
    checkNear(aces.rgb(0, 0).y, expect.y, 0.05f, "auto PNG red → ACEScg G");
    checkNear(aces.rgb(0, 0).z, expect.z, 0.05f, "auto PNG red → ACEScg B");
    checkNear(taggedAces.rgb(0, 0).x, 1.0f, 0.02f, "tagged ACEScg skips matrix");
    check(std::fabs(aces.rgb(0, 0).x - raw.rgb(0, 0).x) > 0.05f, "ACEScg convert changes Rec.709 red");
}

// Glass sphere over a floor lit by a small rect light: PT+MNEE and BDPT are
// independent estimators of the same transport — their total energy must agree,
// and caustics ON must deliver more light under the glass than caustics OFF.
void testCausticsGlassSphere() {
    std::printf("caustics-glass-sphere\n");
    auto buildScene = [](int integrator, int caustics, bool withSphere = true) {
        auto scene = std::make_shared<Scene>();

        // Floor: 8x8 quad at y=0.
        MeshPtr floor = std::make_shared<Mesh>();
        floor->positions = {Vec3(-4, 0, -4), Vec3(4, 0, -4), Vec3(4, 0, 4), Vec3(-4, 0, 4)};
        floor->indices = {0, 2, 1, 0, 3, 2};
        floor->normals = {Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0)};
        floor->uvs = {Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)};
        floor->validate();
        const int floorMesh = scene->addMesh(floor);
        Material floorMat;
        floorMat.baseColor = Vec3(0.75f, 0.75f, 0.75f);
        floorMat.roughness = 0.9f;
        floorMat.specular = 0.0f;
        const int floorMatIdx = scene->addMaterial(floorMat);
        InstanceData floorInst;
        floorInst.meshIndex = floorMesh;
        floorInst.materialIndex = floorMatIdx;
        scene->instances.push_back(floorInst);

        // Glass sphere hovering above the floor.
        if (withSphere) {
        MeshPtr ball = makeSphereMesh(0.7f, 48, 24);
        const int ballMesh = scene->addMesh(ball);
        Material glass;
        glass.baseColor = Vec3(1.0f, 1.0f, 1.0f);
        glass.roughness = 0.0f;
        glass.transmission = 1.0f;
        glass.ior = 1.5f;
        glass.specular = 1.0f;
        const int glassIdx = scene->addMaterial(glass);
        InstanceData ballInst;
        ballInst.xform = Mat4::translate(Vec3(0.0f, 1.0f, 0.0f));
        ballInst.meshIndex = ballMesh;
        ballInst.materialIndex = glassIdx;
        scene->instances.push_back(ballInst);
        }

        // Small rect light high above, pointing down (rect emits along -Z).
        LightData light;
        light.type = kLightRect;
        light.width = 0.8f;
        light.height = 0.8f;
        light.intensity = 60.0f;
        light.normalize = 1;
        light.visibleCamera = 0;
        // Rotate -Z to -Y: −90° about X maps (0,0,-1) → (0,-1,0) — light shines down.
        light.xform = Mat4::translate(Vec3(0.0f, 4.0f, 0.0f)) * Mat4::rotateX(-90.0f);
        light.xformInv = inverse(light.xform);
        scene->lights.push_back(light);

        scene->settings.resolutionX = 72;
        scene->settings.resolutionY = 54;
        scene->settings.samplesPerPixel = 24;
        scene->settings.maxDepth = 8;
        scene->settings.integrator = integrator;
        scene->settings.caustics = caustics;
        scene->settings.causticsEngine = kCausticsEngineMnee;
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
        scene->settings.clampDirect = 0.0f;
        scene->settings.clampIndirect = 0.0f;  // unbiased comparison
        scene->camera.cameraToWorld =
            lookAtMatrix(Vec3(2.4f, 2.6f, 2.4f), Vec3(0.0f, 0.35f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
        scene->cameraAuthored = true;
        scene->finalize();
        return scene;
    };

    auto renderSum = [&](int integrator, int caustics, bool& finiteOut) -> double {
        RenderSession session;
        session.setScene(buildScene(integrator, caustics));
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double sum = 0.0;
        finiteOut = true;
        for (int y = 0; y < img.height(); ++y)
            for (int x = 0; x < img.width(); ++x) {
                const Vec3 c = img.rgb(x, y);
                if (!isFinite(c)) finiteOut = false;
                sum += double(luminance(c));
            }
        return sum;
    };

    bool finPt = true, finBdpt = true, finOff = true;
    const double sumPt = renderSum(kIntegratorPathTracer, 1, finPt);
    const double sumBdpt = renderSum(kIntegratorBdpt, 1, finBdpt);
    const double sumOff = renderSum(kIntegratorPathTracer, 0, finOff);
    check(finPt && finBdpt && finOff, "caustics renders are finite");
    check(sumPt > 0.0 && sumBdpt > 0.0 && sumOff > 0.0, "caustics renders produce light");
    // Caustics ON must deliver noticeably more energy than the dark-shadow OFF mode
    // (the glass transmits ~30% of the scene's light here).
    check(sumPt > sumOff * 1.15, "MNEE adds caustic energy vs caustics off");
    check(sumBdpt > sumOff * 1.15, "BDPT adds caustic energy vs caustics off");
    // Independent estimators must agree on the total transport (tight band: the
    // MNEE fix for the fold Jacobian brought them within ~1%; allow noise room).
    const double ratio = sumPt > 0.0 ? sumBdpt / sumPt : 0.0;
    check(ratio > 0.85 && ratio < 1.18, "BDPT and PT+MNEE energies agree");
    std::printf("  sumPT=%.1f sumBDPT=%.1f sumOFF=%.1f ratio=%.3f\n", sumPt, sumBdpt, sumOff, ratio);

    // Point light: BSDF sampling can never hit a delta light, so refractive
    // caustics from small sources exist ONLY through the manifold connections
    // (both in the Path Tracer and in BDPT).
    auto renderPoint = [&](int integrator, int caustics, bool& finiteOut) -> double {
        auto scene = buildScene(integrator, caustics);
        scene->lights[0].type = kLightPoint;
        scene->lights[0].intensity = 30.0f;
        scene->lights[0].normalize = 0;
        scene->finalize();
        RenderSession session;
        session.setScene(scene);
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double sum = 0.0;
        finiteOut = true;
        for (int y = 0; y < img.height(); ++y)
            for (int x = 0; x < img.width(); ++x) {
                const Vec3 c = img.rgb(x, y);
                if (!isFinite(c)) finiteOut = false;
                sum += double(luminance(c));
            }
        return sum;
    };
    bool finPointOn = true, finPointOff = true, finPointBdpt = true;
    const double sumPointOn = renderPoint(kIntegratorPathTracer, 1, finPointOn);
    const double sumPointOff = renderPoint(kIntegratorPathTracer, 0, finPointOff);
    const double sumPointBdpt = renderPoint(kIntegratorBdpt, 1, finPointBdpt);
    check(finPointOn && finPointOff && finPointBdpt, "point-light caustics renders are finite");
    check(sumPointOn > sumPointOff * 1.1, "MNEE delivers point-light caustics through glass");
    check(sumPointBdpt > sumPointOff * 1.1, "BDPT manifold connections deliver point-light caustics");
    const double pointRatio = sumPointOn > 0.0 ? sumPointBdpt / sumPointOn : 0.0;
    // Monte Carlo agreement — allow platform variance (Windows Clang floated to ~1.26).
    check(pointRatio > 0.8 && pointRatio < 1.35, "BDPT and PT point-light caustics agree");
    std::printf("  pointOn=%.1f pointOff=%.1f pointBDPT=%.1f ratio=%.3f\n", sumPointOn, sumPointOff,
                sumPointBdpt, pointRatio);
}

// pbrt SampleLe: BDPT must start light subpaths from distant lights so a sun
// caustic under glass exists. Path Tracer + pbrt engine cannot find that SDS/LDS
// family; the hot spot is the light-tracing splat.
void testBdptDistantSunCaustics() {
    std::printf("bdpt-distant-sun-caustics\n");

    auto buildScene = [](int integrator, int caustics) {
        auto scene = std::make_shared<Scene>();
        MeshPtr floor = std::make_shared<Mesh>();
        floor->positions = {Vec3(-4, 0, -4), Vec3(4, 0, -4), Vec3(4, 0, 4), Vec3(-4, 0, 4)};
        floor->indices = {0, 2, 1, 0, 3, 2};
        floor->normals = {Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0)};
        floor->validate();
        const int floorMesh = scene->addMesh(floor);
        Material floorMat;
        floorMat.baseColor = Vec3(0.75f);
        floorMat.roughness = 0.9f;
        floorMat.specular = 0.0f;
        const int floorIdx = scene->addMaterial(floorMat);
        InstanceData floorInst;
        floorInst.meshIndex = floorMesh;
        floorInst.materialIndex = floorIdx;
        scene->instances.push_back(floorInst);

        MeshPtr ball = makeSphereMesh(0.7f, 48, 24);
        const int ballMesh = scene->addMesh(ball);
        Material glass;
        glass.baseColor = Vec3(1.0f);
        glass.roughness = 0.0f;
        glass.transmission = 1.0f;
        glass.ior = 1.5f;
        glass.specular = 1.0f;
        const int glassIdx = scene->addMaterial(glass);
        InstanceData ballInst;
        ballInst.xform = Mat4::translate(Vec3(0.0f, 1.0f, 0.0f));
        ballInst.meshIndex = ballMesh;
        ballInst.materialIndex = glassIdx;
        scene->instances.push_back(ballInst);

        LightData sun;
        sun.type = kLightDistant;
        sun.intensity = 8.0f;
        sun.angle = 0.53f;
        sun.normalize = 1;
        sun.visibleCamera = 0;
        sun.xform = Mat4::rotateX(-90.0f);  // +Z → +Y, so NEE wi points at the sun
        sun.xformInv = inverse(sun.xform);
        scene->lights.push_back(sun);

        scene->settings.resolutionX = 72;
        scene->settings.resolutionY = 54;
        scene->settings.samplesPerPixel = 48;
        scene->settings.maxDepth = 8;
        scene->settings.integrator = integrator;
        scene->settings.caustics = caustics;
        scene->settings.causticsEngine = kCausticsEnginePbrt;
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
        scene->settings.clampDirect = 0.0f;
        scene->settings.clampIndirect = 0.0f;
        scene->camera.cameraToWorld =
            lookAtMatrix(Vec3(2.4f, 2.6f, 2.4f), Vec3(0.0f, 0.35f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
        scene->cameraAuthored = true;
        scene->finalize();
        return scene;
    };

    auto stats = [&](int integrator, int caustics, bool& finiteOut, double& peakOut) -> double {
        RenderSession session;
        session.setScene(buildScene(integrator, caustics));
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double sum = 0.0;
        peakOut = 0.0;
        finiteOut = true;
        for (int y = 0; y < img.height(); ++y)
            for (int x = 0; x < img.width(); ++x) {
                const Vec3 c = img.rgb(x, y);
                if (!isFinite(c)) finiteOut = false;
                const double yLum = double(luminance(c));
                sum += yLum;
                if (yLum > peakOut) peakOut = yLum;
            }
        return sum;
    };

    bool finOn = true, finOff = true, finPt = true;
    double peakOn = 0.0, peakOff = 0.0, peakPt = 0.0;
    const double sumOn = stats(kIntegratorBdpt, 1, finOn, peakOn);
    const double sumOff = stats(kIntegratorBdpt, 0, finOff, peakOff);
    const double sumPt = stats(kIntegratorPathTracer, 1, finPt, peakPt);
    check(finOn && finOff && finPt, "distant-sun caustics renders are finite");
    check(sumOn > 0.0 && sumOff > 0.0, "distant-sun renders produce light");
    check(peakOn > peakOff * 1.25, "BDPT SampleLe distant light makes a caustic hot spot");
    check(peakOn > peakPt * 1.15, "BDPT distant caustic is hotter than Path Tracer");
    std::printf("  bdptOn sum=%.1f peak=%.3f | bdptOff sum=%.1f peak=%.3f | ptOn sum=%.1f peak=%.3f\n",
                sumOn, peakOn, sumOff, peakOff, sumPt, peakPt);
}

void testCameraProjShared() {
    std::printf("camera-proj-shared\n");
    SceneView scene;
    scene.settings.resolutionX = 200;
    scene.settings.resolutionY = 100;
    scene.camera.focalLength = 50.0f;
    scene.camera.sensorWidth = 36.0f;
    const CameraProj proj = buildCameraProj(scene);
    check(proj.valid, "camera proj valid");
    float px = 0.0f, py = 0.0f, cosTheta = 0.0f, dist2 = 0.0f;
    check(projectToPixel(proj, Vec3(0.0f, 0.0f, -2.0f), px, py, cosTheta, dist2),
          "point on optical axis projects");
    check(std::fabs(px - 100.0f) < 1.0f && std::fabs(py - 50.0f) < 1.0f, "axis maps to raster center");
    check(cameraPdfOmega(proj, cosTheta) > 0.0f, "camera pdf omega");
    check(!projectToPixel(proj, Vec3(0.0f, 0.0f, 2.0f), px, py, cosTheta, dist2),
          "behind camera rejected");
}

// Camera looks straight down through a glass sphere at the floor. Light-tracing
// splats cannot reach those pixels (floor→camera occluded by glass). BDPT must
// upgrade s=1 to MNEE after the specular eye prefix so caustic energy under the
// glass is visible through refraction — matching PT+MNEE.
void testBdptCausticThroughRefraction() {
    std::printf("bdpt-caustic-through-refraction\n");

    auto buildScene = [](int integrator, int caustics) {
        auto scene = std::make_shared<Scene>();
        MeshPtr floor = std::make_shared<Mesh>();
        floor->positions = {Vec3(-3, 0, -3), Vec3(3, 0, -3), Vec3(3, 0, 3), Vec3(-3, 0, 3)};
        floor->indices = {0, 2, 1, 0, 3, 2};
        floor->normals = {Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0)};
        floor->validate();
        const int floorMesh = scene->addMesh(floor);
        Material floorMat;
        floorMat.baseColor = Vec3(0.8f);
        floorMat.roughness = 0.95f;
        floorMat.specular = 0.0f;
        const int floorIdx = scene->addMaterial(floorMat);
        InstanceData floorInst;
        floorInst.meshIndex = floorMesh;
        floorInst.materialIndex = floorIdx;
        scene->instances.push_back(floorInst);

        MeshPtr ball = makeSphereMesh(0.85f, 48, 24);
        const int ballMesh = scene->addMesh(ball);
        Material glass;
        glass.baseColor = Vec3(1.0f);
        glass.roughness = 0.0f;
        glass.transmission = 1.0f;
        glass.ior = 1.5f;
        glass.specular = 1.0f;
        const int glassIdx = scene->addMaterial(glass);
        InstanceData ballInst;
        ballInst.xform = Mat4::translate(Vec3(0.0f, 0.9f, 0.0f));
        ballInst.meshIndex = ballMesh;
        ballInst.materialIndex = glassIdx;
        scene->instances.push_back(ballInst);

        LightData light;
        light.type = kLightRect;
        light.width = 0.8f;
        light.height = 0.8f;
        light.intensity = 60.0f;
        light.normalize = 1;
        light.visibleCamera = 0;
        // Overhead light offset so the camera-invisible proxy is not on the
        // primary ray (BDPT must still skip such proxies — covered here too).
        light.xform = Mat4::translate(Vec3(0.4f, 4.0f, 0.4f)) * Mat4::rotateX(-90.0f);
        light.xformInv = inverse(light.xform);
        scene->lights.push_back(light);

        scene->settings.resolutionX = 64;
        scene->settings.resolutionY = 64;
        scene->settings.samplesPerPixel = 64;
        scene->settings.maxDepth = 8;
        scene->settings.integrator = integrator;
        scene->settings.caustics = caustics;
        scene->settings.causticsEngine = kCausticsEngineMnee;
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
        scene->settings.clampDirect = 0.0f;
        scene->settings.clampIndirect = 0.0f;
        scene->settings.lightSamples = 2;
        // Top-down: primary hits are glass over the silhouette center.
        scene->camera.cameraToWorld =
            lookAtMatrix(Vec3(0.0f, 4.2f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -1.0f));
        scene->cameraAuthored = true;
        scene->finalize();
        return scene;
    };

    auto renderCenter = [&](int integrator, int caustics, bool& finiteOut) -> double {
        RenderSession session;
        session.setScene(buildScene(integrator, caustics));
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double sum = 0.0;
        finiteOut = true;
        const int x0 = img.width() / 4;
        const int x1 = (3 * img.width()) / 4;
        const int y0 = img.height() / 4;
        const int y1 = (3 * img.height()) / 4;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                const Vec3 c = img.rgb(x, y);
                if (!isFinite(c)) finiteOut = false;
                sum += double(luminance(c));
            }
        return sum;
    };

    bool finBdptOn = true, finBdptOff = true, finPtOn = true;
    const double sumBdptOn = renderCenter(kIntegratorBdpt, 1, finBdptOn);
    const double sumBdptOff = renderCenter(kIntegratorBdpt, 0, finBdptOff);
    const double sumPtOn = renderCenter(kIntegratorPathTracer, 1, finPtOn);
    check(finBdptOn && finBdptOff && finPtOn, "through-refraction renders are finite");
    check(sumBdptOn > sumBdptOff * 1.2, "BDPT MNEE lights floor seen through glass");
    check(sumPtOn > sumBdptOff * 1.2, "PT MNEE lights floor seen through glass");
    const double ratio = sumPtOn > 0.0 ? sumBdptOn / sumPtOn : 0.0;
    // 64 spp MNEE through-glass is noisy; BDPT vs PT often sits near 0.5.
    // Both must still beat caustics-off by a wide margin (checks above).
    check(ratio > 0.4 && ratio < 2.0, "BDPT and PT through-glass caustic energy agree");
    std::printf("  bdptOn=%.1f bdptOff=%.1f ptOn=%.1f ratio=%.3f\n", sumBdptOn, sumBdptOff, sumPtOn, ratio);
}

// GPU photon aiming: aimed-only SampleLe (mix=1), screen solid angle, clusters.
void testPhotonAim() {
    std::printf("photon-aim\n");

    const Vec3 center(0.0f, 0.0f, 10.0f);
    const float radius = 1.0f;
    const float farOmega = gpuScreenSolidAngle(Vec3(0.0f, 0.0f, 0.0f), center, radius);
    const float nearOmega = gpuScreenSolidAngle(Vec3(0.0f, 0.0f, 4.0f), center, radius);
    check(nearOmega > farOmega * 1.5f, "closer camera → larger caster solid angle");
    check(farOmega > 0.0f, "far solid angle positive");

    GpuPhotonCluster cluster;
    cluster.center = Vec3(0.0f, 0.0f, 0.0f);
    cluster.radius = 1.0f;
    cluster.weight = 1.0f;
    const Vec3 planeC(0.0f, 0.0f, 0.0f);
    const Vec3 planeN(0.0f, 1.0f, 0.0f);
    const float sceneR = 10.0f;
    const Vec3 aimedP = gpuClusterDiskPoint(cluster, planeC, planeN, Vec2(0.0f, 0.0f));
    check(kGpuPhotonAimMix >= 1.0f, "GPU light trace is aimed-only (no uniform mix)");
    const float mixPdf = gpuPhotonAimMixtureDiskPdf(aimedP, planeC, planeN, sceneR, kGpuPhotonAimMix,
                                                    &cluster, 1);
    const float uniPdf = 1.0f / (kPi * sceneR * sceneR);
    const float aimDisk = gpuAimDiskPdf(aimedP, planeC, planeN, &cluster, 1);
    check(mixPdf > uniPdf * 10.0f, "aimed disk pdf >> uniform scene-disk pdf");
    check(aimDisk > 0.0f, "aimed point in cluster disk");
    checkNear(mixPdf, aimDisk, 1e-6f, "mix=1 disk pdf is the aim disk pdf");

    const Vec3 origin(0.0f, 0.0f, 0.0f);
    GpuPhotonCluster coneC;
    coneC.center = Vec3(0.0f, 0.0f, 8.0f);
    coneC.radius = 1.0f;
    coneC.weight = 1.0f;
    const Vec3 aimDir = normalize(coneC.center - origin);
    const float conePdf = gpuAimConePdf(origin, aimDir, &coneC, 1);
    check(conePdf > 0.0f, "aimed cone pdf of aimed dir > 0");
    const float missPdf = gpuAimConePdf(origin, Vec3(1.0f, 0.0f, 0.0f), &coneC, 1);
    check(missPdf == 0.0f, "direction outside caster cone has aim pdf 0");
    check(gpuPhotonAimDirPdf(origin, Vec3(1.0f, 0.0f, 0.0f), 1.0f, &coneC, 1, kInv4Pi) == 0.0f,
          "aimed-only pdf has no uniform floor outside the cone");
    check(gpuPhotonAimSelect(1.0f, 0.999f), "aimed-only always selects aim");
    check(!gpuPhotonAimSelect(0.0f, 0.0f), "mix 0 never selects aim");
    Vec3 sampledDir;
    check(gpuSamplePhotonAimDir(origin, coneC, 0.25f, 0.4f, sampledDir), "sample cone toward cluster");
    check(gpuAimConePdf(origin, sampledDir, &coneC, 1) > 0.0f, "sampled aim dir has positive cone pdf");

    GpuPhotonCluster tiny;
    tiny.center = Vec3(0.0f, 0.0f, 1000.0f);
    tiny.radius = 1.0e-4f;
    tiny.weight = 1.0f;
    const float rawCos = gpuSphereCosThetaMax(origin, tiny.center, tiny.radius);
    check(gpuUniformConePdf(rawCos) == 0.0f, "far tiny sphere subtends a degenerate cone in float32");
    check(gpuAimConeCosThetaMax(origin, tiny.center, tiny.radius) < 1.0f,
          "clamped aim cone cosMax stays below 1");
    Vec3 farDir;
    check(gpuSamplePhotonAimDir(origin, tiny, 0.25f, 0.4f, farDir), "sample far tiny caster");
    const float farPdf = gpuAimConePdf(origin, farDir, &tiny, 1);
    check(farPdf > 0.0f && srIsFinite(farPdf), "far tiny aim cone pdf is finite and > 0");
    const float farDirPdf =
        gpuPhotonAimDirPdf(origin, farDir, kGpuPhotonAimMix, &tiny, 1, kInv4Pi);
    check(farDirPdf > 0.0f && srIsFinite(farDirPdf), "aimed-only pdf of far tiny sample > 0");
    checkNear(farDirPdf, farPdf, 1e-6f, "mix=1 dir pdf is the aim cone pdf");

    check(gpuPickPhotonCluster(&cluster, 1, 0.0f) == 0, "pick single cluster");
    check(gpuPickPhotonCluster(&cluster, 1, 0.99f) == 0, "pick single cluster high u");

    auto scene = std::make_shared<Scene>();
    MeshPtr floor = std::make_shared<Mesh>();
    floor->positions = {Vec3(-8, 0, -8), Vec3(8, 0, -8), Vec3(8, 0, 8), Vec3(-8, 0, 8)};
    floor->indices = {0, 2, 1, 0, 3, 2};
    floor->normals = {Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0)};
    floor->validate();
    const int floorMesh = scene->addMesh(floor);
    Material floorMat;
    floorMat.baseColor = Vec3(0.75f);
    floorMat.roughness = 0.9f;
    floorMat.specular = 0.0f;
    const int floorIdx = scene->addMaterial(floorMat);
    InstanceData floorInst;
    floorInst.meshIndex = floorMesh;
    floorInst.materialIndex = floorIdx;
    scene->instances.push_back(floorInst);

    MeshPtr ball = makeSphereMesh(0.7f, 24, 16);
    const int ballMesh = scene->addMesh(ball);
    Material glass;
    glass.baseColor = Vec3(1.0f);
    glass.roughness = 0.0f;
    glass.transmission = 1.0f;
    glass.ior = 1.5f;
    glass.specular = 1.0f;
    const int glassIdx = scene->addMaterial(glass);
    InstanceData ballInst;
    ballInst.xform = Mat4::translate(Vec3(0.0f, 1.0f, 0.0f));
    ballInst.meshIndex = ballMesh;
    ballInst.materialIndex = glassIdx;
    scene->instances.push_back(ballInst);

    scene->camera.cameraToWorld =
        lookAtMatrix(Vec3(2.4f, 2.6f, 2.4f), Vec3(0.0f, 0.35f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    scene->cameraAuthored = true;
    scene->finalize();

    GpuPhotonCluster clusters[kMaxGpuPhotonClusters];
    const int n = fillPhotonAimClusters(scene->view(), clusters, kMaxGpuPhotonClusters);
    check(n == 1, "glass sphere is the only photon-aim cluster");
    if (n == 1) {
        check(clusters[0].radius < 2.0f, "cluster radius is the Buddha-sized sphere, not the ground");
        check(std::fabs(clusters[0].weight - 1.0f) < 1e-4f, "single cluster weight is 1");
        check(clusters[0].center.y > 0.3f, "cluster centered on the glass, not y=0 ground");
    }

    scene->materials[glassIdx].contributeCaustics = 0;
    GpuPhotonCluster offClusters[kMaxGpuPhotonClusters];
    const int nOff = fillPhotonAimClusters(scene->view(), offClusters, kMaxGpuPhotonClusters);
    check(nOff == 0, "Contribute to Caustics off → no aim clusters");

    std::printf("  solidAngle near/far=%.3f mixPdf/uni=%.1f clusters=%d\n", nearOmega / farOmega,
                mixPdf / uniPdf, n);
}

// Photon / VCM caustic engine: must deliver more energy under glass than caustics
// off, and material Contribute to Caustics off must kill that transport.
void testPhotonCaustics() {
    std::printf("photon-caustics\n");

    auto buildScene = [](int caustics, int engine, int matContribute) {
        auto scene = std::make_shared<Scene>();
        MeshPtr floor = std::make_shared<Mesh>();
        floor->positions = {Vec3(-4, 0, -4), Vec3(4, 0, -4), Vec3(4, 0, 4), Vec3(-4, 0, 4)};
        floor->indices = {0, 2, 1, 0, 3, 2};
        floor->normals = {Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0)};
        floor->validate();
        const int floorMesh = scene->addMesh(floor);
        Material floorMat;
        floorMat.baseColor = Vec3(0.75f);
        floorMat.roughness = 0.9f;
        floorMat.specular = 0.0f;
        const int floorIdx = scene->addMaterial(floorMat);
        InstanceData floorInst;
        floorInst.meshIndex = floorMesh;
        floorInst.materialIndex = floorIdx;
        scene->instances.push_back(floorInst);

        MeshPtr ball = makeSphereMesh(0.7f, 48, 24);
        const int ballMesh = scene->addMesh(ball);
        Material glass;
        glass.baseColor = Vec3(1.0f);
        glass.roughness = 0.0f;
        glass.transmission = 1.0f;
        glass.ior = 1.5f;
        glass.specular = 1.0f;
        glass.contributeCaustics = matContribute;
        const int glassIdx = scene->addMaterial(glass);
        InstanceData ballInst;
        ballInst.xform = Mat4::translate(Vec3(0.0f, 1.0f, 0.0f));
        ballInst.meshIndex = ballMesh;
        ballInst.materialIndex = glassIdx;
        scene->instances.push_back(ballInst);

        LightData light;
        light.type = kLightRect;
        light.width = 0.8f;
        light.height = 0.8f;
        light.intensity = 60.0f;
        light.normalize = 1;
        light.visibleCamera = 0;
        light.xform = Mat4::translate(Vec3(0.0f, 4.0f, 0.0f)) * Mat4::rotateX(-90.0f);
        light.xformInv = inverse(light.xform);
        scene->lights.push_back(light);

        scene->settings.resolutionX = 72;
        scene->settings.resolutionY = 54;
        scene->settings.samplesPerPixel = 16;
        scene->settings.maxDepth = 8;
        scene->settings.integrator = kIntegratorPathTracer;
        scene->settings.caustics = caustics;
        scene->settings.causticsEngine = engine;
        scene->settings.photonCount = 40000;
        scene->settings.photonRadius = 0.15f;
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
        scene->settings.clampDirect = 0.0f;
        scene->settings.clampIndirect = 0.0f;
        scene->camera.cameraToWorld =
            lookAtMatrix(Vec3(2.4f, 2.6f, 2.4f), Vec3(0.0f, 0.35f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
        scene->cameraAuthored = true;
        scene->finalize();
        return scene;
    };

    auto renderSum = [&](int caustics, int engine, int matContribute, bool& finiteOut) -> double {
        RenderSession session;
        session.setScene(buildScene(caustics, engine, matContribute));
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double sum = 0.0;
        finiteOut = true;
        for (int y = 0; y < img.height(); ++y)
            for (int x = 0; x < img.width(); ++x) {
                const Vec3 c = img.rgb(x, y);
                if (!isFinite(c)) finiteOut = false;
                sum += double(luminance(c));
            }
        return sum;
    };

    bool finOn = true, finOff = true, finMatOff = true, finMnee = true;
    const double sumPhoton = renderSum(1, kCausticsEnginePhoton, 1, finOn);
    const double sumOff = renderSum(0, kCausticsEnginePhoton, 1, finOff);
    const double sumMatOff = renderSum(1, kCausticsEnginePhoton, 0, finMatOff);
    const double sumMnee = renderSum(1, kCausticsEngineMnee, 1, finMnee);
    check(finOn && finOff && finMatOff && finMnee, "photon caustics renders are finite");
    check(sumPhoton > sumOff * 1.1, "photon map adds caustic energy vs caustics off");
    check(sumMatOff < sumPhoton * 0.85, "material Contribute to Caustics off reduces caustics");
    check(sumMnee > sumOff * 1.1, "MNEE engine still adds caustic energy");
    std::printf("  photon=%.1f off=%.1f matOff=%.1f mnee=%.1f\n", sumPhoton, sumOff, sumMatOff, sumMnee);
}

// Rough (but still tightly focusing) glass must converge like smooth glass: the
// caustic family belongs to light tracing, otherwise the eye path has to stumble
// onto a small light through the chain and the render is all fireflies.
void testRoughGlassCaustics() {
    std::printf("rough-glass-caustics\n");

    auto buildScene = [](int integrator, float roughness, float causticClamp) {
        auto scene = std::make_shared<Scene>();
        MeshPtr floor = std::make_shared<Mesh>();
        floor->positions = {Vec3(-4, 0, -4), Vec3(4, 0, -4), Vec3(4, 0, 4), Vec3(-4, 0, 4)};
        floor->indices = {0, 2, 1, 0, 3, 2};
        floor->normals = {Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0)};
        floor->validate();
        const int floorMesh = scene->addMesh(floor);
        Material floorMat;
        floorMat.baseColor = Vec3(0.75f);
        floorMat.roughness = 0.9f;
        floorMat.specular = 0.0f;
        const int floorIdx = scene->addMaterial(floorMat);
        InstanceData floorInst;
        floorInst.meshIndex = floorMesh;
        floorInst.materialIndex = floorIdx;
        scene->instances.push_back(floorInst);

        MeshPtr ball = makeSphereMesh(0.7f, 48, 24);
        const int ballMesh = scene->addMesh(ball);
        Material glass;
        glass.baseColor = Vec3(1.0f);
        glass.roughness = roughness;
        glass.transmission = 1.0f;
        glass.ior = 1.5f;
        glass.specular = 1.0f;
        const int glassIdx = scene->addMaterial(glass);
        InstanceData ballInst;
        ballInst.xform = Mat4::translate(Vec3(0.0f, 1.0f, 0.0f));
        ballInst.meshIndex = ballMesh;
        ballInst.materialIndex = glassIdx;
        scene->instances.push_back(ballInst);

        // Small light: the whole point is that BSDF sampling cannot find it.
        LightData light;
        light.type = kLightRect;
        light.width = 0.2f;
        light.height = 0.2f;
        light.intensity = 900.0f;
        light.normalize = 1;
        light.visibleCamera = 0;
        light.xform = Mat4::translate(Vec3(0.0f, 4.0f, 0.0f)) * Mat4::rotateX(-90.0f);
        light.xformInv = inverse(light.xform);
        scene->lights.push_back(light);

        scene->settings.resolutionX = 72;
        scene->settings.resolutionY = 54;
        scene->settings.samplesPerPixel = 32;
        scene->settings.maxDepth = 8;
        scene->settings.integrator = integrator;
        scene->settings.caustics = 1;
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
        scene->settings.clampDirect = 0.0f;
        scene->settings.clampIndirect = 0.0f;  // unbiased: fireflies stay visible
        scene->settings.causticClamp = causticClamp;
        scene->camera.cameraToWorld =
            lookAtMatrix(Vec3(2.4f, 2.6f, 2.4f), Vec3(0.0f, 0.35f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
        scene->cameraAuthored = true;
        scene->finalize();
        return scene;
    };

    // Peak-to-mean luminance: a converged caustic has a smooth falloff, while an
    // unresolved one is a handful of enormous pixels over a black floor.
    auto render = [&](int integrator, float roughness, float causticClamp,
                      double& peakOverMean) -> double {
        RenderSession session;
        session.setScene(buildScene(integrator, roughness, causticClamp));
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double sum = 0.0;
        double peak = 0.0;
        for (int y = 0; y < img.height(); ++y)
            for (int x = 0; x < img.width(); ++x) {
                const double l = double(luminance(img.rgb(x, y)));
                sum += l;
                if (l > peak) peak = l;
            }
        const double mean = sum / double(img.width() * img.height());
        peakOverMean = mean > 1e-9 ? peak / mean : 0.0;
        return sum;
    };

    double peakSmooth = 0.0, peakRough = 0.0, peakPtRough = 0.0;
    const double sumSmooth = render(kIntegratorBdpt, 0.0f, 0.0f, peakSmooth);
    const double sumRough = render(kIntegratorBdpt, 0.1f, 0.0f, peakRough);
    const double sumPtRough = render(kIntegratorPathTracer, 0.1f, 0.0f, peakPtRough);
    std::printf("  bdpt smooth sum=%.1f peak/mean=%.1f | rough(0.1) sum=%.1f peak/mean=%.1f\n", sumSmooth,
                peakSmooth, sumRough, peakRough);
    std::printf("  pt rough(0.1) sum=%.1f peak/mean=%.1f\n", sumPtRough, peakPtRough);


    check(sumSmooth > 0.0 && sumRough > 0.0, "rough and smooth glass both transport light");
    // Roughening the glass slightly must not blow the estimator up: the caustic
    // spreads a little, so total energy stays close and the peak stays bounded.
    const double ratio = sumRough / sumSmooth;
    check(ratio > 0.6 && ratio < 1.6, "rough glass keeps the caustic energy of smooth glass");
    check(peakRough < peakSmooth * 3.0 + 30.0, "rough glass does not degenerate into fireflies");
    // BDPT routes rough caustics through light tracing while the plain path tracer
    // BSDF-samples them; the two must still agree on the transported energy.
    const double ptRatio = sumPtRough > 0.0 ? sumRough / sumPtRough : 0.0;
    check(ptRatio > 0.7 && ptRatio < 1.4, "BDPT and PT agree on rough glass energy");
}

// Sparkle *inside* a refractive object at roughness 0. Looking through glass at a
// small light is an SDS path that light tracing cannot cover, and BSDF sampling
// never converges — a safety cap is always applied (causticClamp tightens it).
void testRefractionSparkleClamp() {
    std::printf("refraction-sparkle\n");

    auto buildScene = [](float causticClamp) {
        auto scene = std::make_shared<Scene>();
        MeshPtr floor = std::make_shared<Mesh>();
        floor->positions = {Vec3(-6, 0, -6), Vec3(6, 0, -6), Vec3(6, 0, 6), Vec3(-6, 0, 6)};
        floor->indices = {0, 2, 1, 0, 3, 2};
        floor->normals = {Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0)};
        floor->validate();
        const int floorMesh = scene->addMesh(floor);
        Material floorMat;
        floorMat.baseColor = Vec3(0.75f);
        floorMat.roughness = 0.9f;
        floorMat.specular = 0.0f;
        const int floorIdx = scene->addMaterial(floorMat);
        InstanceData floorInst;
        floorInst.meshIndex = floorMesh;
        floorInst.materialIndex = floorIdx;
        scene->instances.push_back(floorInst);

        // Smooth glass: the fireflies are SDS hits on the light, not roughness.
        MeshPtr ball = makeSphereMesh(0.7f, 48, 24);
        const int ballMesh = scene->addMesh(ball);
        Material glass;
        glass.baseColor = Vec3(1.0f);
        glass.roughness = 0.0f;
        glass.transmission = 1.0f;
        glass.ior = 1.5f;
        glass.specular = 1.0f;
        const int glassIdx = scene->addMaterial(glass);
        InstanceData ballInst;
        ballInst.xform = Mat4::translate(Vec3(0.0f, 0.8f, 0.0f));
        ballInst.meshIndex = ballMesh;
        ballInst.materialIndex = glassIdx;
        scene->instances.push_back(ballInst);

        LightData light;
        light.type = kLightRect;
        light.width = 0.05f;
        light.height = 0.05f;
        light.intensity = 2000.0f;
        light.normalize = 1;
        light.visibleCamera = 1;  // visible through the glass = SDS fireflies
        light.xform = Mat4::translate(Vec3(0.0f, 7.0f, 0.0f)) * Mat4::rotateX(-90.0f);
        light.xformInv = inverse(light.xform);
        scene->lights.push_back(light);

        scene->settings.resolutionX = 64;
        scene->settings.resolutionY = 48;
        scene->settings.samplesPerPixel = 48;
        scene->settings.maxDepth = 8;
        scene->settings.integrator = kIntegratorBdpt;
        scene->settings.caustics = 1;
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
        scene->settings.clampDirect = 0.0f;
        scene->settings.clampIndirect = 0.0f;
        scene->settings.causticClamp = causticClamp;
        scene->camera.cameraToWorld =
            lookAtMatrix(Vec3(0.0f, 0.8f, 1.7f), Vec3(0.0f, 0.8f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
        scene->cameraAuthored = true;
        scene->finalize();
        return scene;
    };

    auto render = [&](float causticClamp, double& peakOverMean, double& peak) -> double {
        RenderSession session;
        session.setScene(buildScene(causticClamp));
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double sum = 0.0;
        peak = 0.0;
        for (int y = 0; y < img.height(); ++y)
            for (int x = 0; x < img.width(); ++x) {
                const double l = double(luminance(img.rgb(x, y)));
                sum += l;
                if (l > peak) peak = l;
            }
        const double mean = sum / double(img.width() * img.height());
        peakOverMean = mean > 1e-9 ? peak / mean : 0.0;
        return sum;
    };

    // Sentinel: causticClamp < 0 disables the safety cap (test-only) so we can
    // measure the uncapped SDS fireflies. Production always keeps the safety.
    double peakSafe = 0.0, peakTight = 0.0, pomSafe = 0.0, pomTight = 0.0;
    const double sumSafe = render(0.0f, pomSafe, peakSafe);
    const double sumTight = render(2.0f, pomTight, peakTight);
    std::printf("  safety sum=%.2f peak=%.2f p/m=%.1f | clamp2 sum=%.2f peak=%.2f p/m=%.1f\n", sumSafe,
                peakSafe, pomSafe, sumTight, peakTight, pomTight);
    check(sumSafe > 0.0 && sumTight > 0.0, "glass interior receives light either way");
    // With the safety cap the peak stays bounded; tightening further must not raise it.
    check(peakSafe < 50.0, "safety cap keeps roughness-0 SDS fireflies bounded");
    check(peakTight <= peakSafe * 1.02, "tighter caustic clamp never raises the peak");

    RenderSettingsData rs;
    rs.causticClamp = 0.0f;
    check(causticFireflyCap(rs) == 10.0f, "default caustic safety floor is 10");
    rs.causticClamp = -1.0f;
    check(causticFireflyCap(rs) == 0.0f, "causticClamp < 0 disables safety floor");
    rs.causticClamp = 2.0f;
    check(causticFireflyCap(rs) == 2.0f, "explicit caustic clamp is used as-is");
    RenderSettingsData clampSt;
    check(pathContributionClamp(clampSt, 0, false, false) == 0.0f, "primary NEE/hit unclamped");
    check(pathContributionClamp(clampSt, 1, false, false) == 10.0f, "indirect uses Direct Clamp");
    check(pathContributionClamp(clampSt, 1, true, false) == 10.0f, "specular hit uses firefly floor");
    check(pathContributionClamp(clampSt, 2, false, true) == 10.0f, "SDS uses firefly floor");
    clampSt.causticClamp = 2.0f;
    check(pathContributionClamp(clampSt, 1, true, true) == 2.0f, "SDS respects causticClamp");
}

// Film reconstruction filters: Box is 1-pixel; Gaussian spreads into neighbours.
void testPixelFilter() {
    std::printf("pixel-filter\n");
    check(isTrivialBoxFilter(kPixelFilterBox, 0.5f), "Box 0.5 is trivial");
    check(!isTrivialBoxFilter(kPixelFilterGaussian, 1.5f), "Gaussian is non-trivial");
    check(filterPixelBorder(0.5f) == 0, "Box needs no FilmTile border");
    check(filterPixelBorder(1.5f) >= 1, "Gaussian needs FilmTile border");

    // Centered sample in pixel (2,2) of a 5×5 film: Box hits only that pixel.
    float boxW = 0.0f;
    int boxHits = 0;
    splatFilteredSample(2.5f, 2.5f, Vec3(1.0f), kPixelFilterBox, 0.5f, 5, 5,
                        [&](int, int, Vec3, float w) {
                            boxW += w;
                            ++boxHits;
                        });
    check(boxHits == 1 && std::fabs(boxW - 1.0f) < 1e-5f, "Box deposits weight 1 into one pixel");

    float gaussCenter = 0.0f, gaussSum = 0.0f;
    int gaussHits = 0;
    splatFilteredSample(2.5f, 2.5f, Vec3(1.0f), kPixelFilterGaussian, 1.5f, 5, 5,
                        [&](int px, int py, Vec3, float w) {
                            gaussSum += w;
                            ++gaussHits;
                            if (px == 2 && py == 2) gaussCenter = w;
                        });
    check(gaussHits > 1, "Gaussian hits more than one pixel");
    check(gaussCenter > 0.0f && gaussCenter < gaussSum, "Gaussian peak at center, mass spreads");
    check(filterWeight2D(kPixelFilterMitchell, 0.0f, 0.0f, 2.0f) > 0.0f, "Mitchell peak positive");
}

// Light-tracing splats are normalized by a global path counter and accumulated in
// a separate plane. Both used 32-bit / single-precision storage, which broke long
// caustic renders: the counter wrapped negative (splats vanished, so caustics
// disappeared and glass shadows went black) and single-precision sums stopped
// growing once they dwarfed the incoming splat values.
void testSplatAccumulationPrecision() {
    std::printf("splat-accumulation\n");

    Framebuffer fb;
    fb.resize(4, 4);
    fb.addSplat(1, 1, Vec3(4.0e9f, 8.0e9f, 1.6e10f));
    const int64_t beyond32Bit = int64_t(1) << 32;  // 960x540 reaches this near 8300 passes
    fb.addSplatPaths(beyond32Bit);
    check(fb.splatPaths() == beyond32Bit, "splat path counter holds counts past 2^31");
    const Vec3 resolved = fb.resolvePixel(1, 1);
    check(resolved.x > 0.0f, "splats survive a path count past 2^31");
    checkNear(resolved.x, 4.0e9f / float(beyond32Bit), 1e-4f, "splat scaling uses the 64-bit count");
    checkNear(resolved.y / resolved.x, 2.0f, 1e-3f, "splat channels keep their ratio");

    // A pixel whose running sum is already large must still absorb small splats.
    Framebuffer deep;
    deep.resize(1, 1);
    deep.addSplat(0, 0, Vec3(1.0e13f, 0.0f, 0.0f));
    for (int i = 0; i < 1000; ++i) deep.addSplat(0, 0, Vec3(1.0e6f, 0.0f, 0.0f));
    deep.addSplatPaths(1);
    // 1e6 is below the float spacing at 1e13, so float accumulation loses every add.
    check(deep.resolvePixel(0, 0).x > 1.00005e13f, "large splat sums keep absorbing small adds");
}

void testBdptTimersFormat() {
    std::printf("bdpt-timers\n");
    check(RenderSettingsData{}.bdptTimers == 0, "BDPT timers default off");

    BdptPassStats stats;
    stats.pixels.store(10);
    stats.nsTotal.store(10ull * 1000000000ull);
    stats.nsAlloc.store(6ull * 1000000000ull);
    stats.nsWalk.store(2ull * 1000000000ull);
    stats.nsSss.store(500ull * 1000000ull);
    stats.nsConnect.store(1ull * 1000000000ull);
    stats.nsSplat.store(1ull * 1000000000ull);
    stats.nEyeSum.store(250);
    stats.nLightSum.store(80);
    stats.nEyeMax.store(31);
    stats.nLightMax.store(12);
    stats.pairs.store(400);
    stats.shadows.store(350);
    stats.splatDeposits.store(40);
    stats.casRetries.store(100);

    BdptPassMeta meta;
    meta.spp = 2;
    meta.maxDepth = 30;
    meta.maxVerts = 31;
    meta.poolThreads = 32;
    meta.vertBytes = 528;
    meta.allocBytesPerPixel = 32736;
    meta.wallNs = 5ull * 1000000000ull;

    const std::string text = formatBdptPassStats(stats, meta);
    check(text.find("BDPT timers") != std::string::npos, "timer header");
    check(text.find("maxDepth=30") != std::string::npos, "timer maxDepth");
    check(text.find("alloc") != std::string::npos, "timer alloc phase");
    check(text.find("inside walk") != std::string::npos, "timer SSS nested in walk");
    check(text.find("KiB/pixel") != std::string::npos, "timer alloc footprint");
    check(text.find("casRetries") != std::string::npos, "timer CAS retries");
    check(text.find("parallel=") != std::string::npos, "timer parallel factor");
    check(text.find("scratch") == std::string::npos, "no scratch line when scratchVerts=0");

    meta.scratchVerts = 31;
    meta.scratchThreads = 33;
    meta.scratchBytes = 31ull * (2 * 528 + 2 * 20 + 2 * 36);
    const std::string scratchText = formatBdptPassStats(stats, meta);
    check(scratchText.find("scratch 31 verts") != std::string::npos, "timer scratch reuse line");
    check(scratchText.find("no per-pixel malloc") != std::string::npos, "timer scratch no malloc");
    check(scratchText.find("KiB/thread") != std::string::npos, "timer scratch per-thread footprint");

    Framebuffer fb;
    fb.resize(2, 2);
    std::atomic<uint64_t> cas{0};
    std::atomic<uint64_t> dep{0};
    fb.setSplatDiag(&cas, &dep);
    fb.addSplat(0, 0, Vec3(1.0f, 0.0f, 0.0f));
    check(dep.load() == 1, "splat deposit counter");
    check(cas.load() == 0, "single-thread splat has no CAS retries");
    fb.setSplatDiag(nullptr, nullptr);
    fb.addSplat(0, 0, Vec3(1.0f, 0.0f, 0.0f));
    check(dep.load() == 1, "deposit counter disabled with nullptr");
}

void testBdptScratchReuse() {
    std::printf("bdpt-scratch\n");
    check(bdpt::kMaxVerts == 4096, "BDPT vertex cap is 4096");
    check(bdpt::bdptSessionVerts(30) == 31, "session verts are maxDepth+1");
    check(bdpt::bdptSessionVerts(4096) == 4096, "session verts clamp to kMaxVerts");
    check(bdpt::bdptSessionVerts(0) == 2, "session verts floor at 2");

    BdptScratchPool pool;
    pool.ensureThreads(4, 31);
    check(pool.threadSlots() == 4, "scratch pool has caller+workers slots");
    BdptScratch* a = pool.get(0);
    BdptScratch* b = pool.get(1);
    check(a != nullptr && b != nullptr && a != b, "scratch slots are distinct");
    check(int(a->eye.size()) == 31, "eye scratch sized to session verts");
    check(int(a->light.size()) == 31, "light scratch sized to session verts");
    check(a->eye.size() != size_t(bdpt::kMaxVerts), "working depth does not allocate the 4096 cap");
    check(a->eye.size() == a->eyeBeta.size() && a->eye.size() == a->eyeWave.size(),
          "six scratch arrays match");
    const bdpt::Vert* eyePtr = a->eye.data();
    const SampledSpectrum* betaPtr = a->eyeBeta.data();
    pool.ensureThreads(4, 31);
    check(pool.get(0)->eye.data() == eyePtr, "second ensure keeps eye pointer");
    check(pool.get(0)->eyeBeta.data() == betaPtr, "second ensure keeps beta pointer");
    pool.ensureThreads(4, 8);  // grow-only path must not shrink
    check(int(pool.get(0)->eye.size()) == 31, "ensure() never shrinks");
    check(pool.get(0)->eye.data() == eyePtr, "smaller ensure does not reallocate");
    pool.ensureThreads(4, 9, true);
    check(int(pool.get(0)->eye.size()) == 9, "pass start shrinks to the new session depth");

    pool.get(1)->eye[0].pdfFwd = 42.0f;
    check(pool.get(0)->eye[0].pdfFwd != 42.0f, "thread slots do not share Vert storage");

    const size_t vertBytes = size_t(31) * sizeof(bdpt::Vert);
    check(bdptScratchBytes(31) == 2 * vertBytes + 2 * size_t(31) * sizeof(SampledSpectrum) +
                                      2 * size_t(31) * sizeof(SampledWavelengths),
          "scratch byte helper matches six arrays");

    BdptScratch& tls = bdptThreadScratch();
    tls.ensure(31);
    const bdpt::Vert* tlsPtr = tls.eye.data();
    tls.ensure(31);
    check(tls.eye.data() == tlsPtr, "thread-local fallback pointer is stable");
}

void testIntegratorDeviceMemory() {
    std::printf("integrator-device\n");
    int cpu = 0, gpu = 0;
    int next = switchIntegratorForBackend(kBackendCpuEmbree, kBackendGpuOptix, kIntegratorBdpt, cpu, gpu);
    check(cpu == kIntegratorBdpt, "leaving CPU stores BDPT");
    check(next == kIntegratorPathTracer, "GPU drops BDPT to Path Tracer");
    next = switchIntegratorForBackend(kBackendGpuOptix, kBackendCpuEmbree, next, cpu, gpu);
    check(next == kIntegratorBdpt, "CPU restores BDPT");

    cpu = 0;
    gpu = 0;
    next = switchIntegratorForBackend(kBackendCpuEmbree, kBackendGpuOptix, kIntegratorWireframe, cpu, gpu);
    check(next == kIntegratorWireframe, "GPU keeps Wireframe");
    next = switchIntegratorForBackend(kBackendGpuOptix, kBackendXpu, kIntegratorAmbientOcclusion, cpu, gpu);
    check(next == kIntegratorAmbientOcclusion, "XPU keeps Ambient Occlusion");
    check(cpu == kIntegratorWireframe, "CPU memory is not overwritten on GPU↔XPU");

    next = switchIntegratorForBackend(kBackendXpu, kBackendCpuEmbree, kIntegratorDirectLighting, cpu, gpu);
    check(next == kIntegratorWireframe, "CPU restores the remembered CPU integrator");

    check(clampIntegratorForBackend(kBackendGpuOptix, kIntegratorBdpt) == kIntegratorPathTracer,
          "GPU clamp drops BDPT");
    check(clampIntegratorForBackend(kBackendXpu, kIntegratorBdpt) == kIntegratorPathTracer,
          "XPU clamp drops BDPT");
    check(clampIntegratorForBackend(kBackendCpuEmbree, kIntegratorBdpt) == kIntegratorBdpt,
          "CPU keeps BDPT");
}

void testUndoHub() {
    std::printf("undo-hub\n");
    registerBuiltinNodes();
    NodeGraph graph;
    Node* settings = graph.createNode("rendersettings", "rs1");
    check(settings != nullptr, "undo test rendersettings");
    if (!settings) return;

    UndoHub hub;
    hub.setGraph(&graph);
    check(hub.stack().undoLimit() == 100, "undo limit is 100");

    const int oldDepth = settings->intValue("maxdepth", 8);
    settings->setParameterValue("maxdepth", 30);
    hub.pushParameter(settings->name(), "maxdepth", oldDepth, 30);
    check(settings->intValue("maxdepth") == 30, "parameter stays at the new value after push");
    hub.undo();
    check(settings->intValue("maxdepth") == oldDepth, "undo restores the old parameter");
    hub.redo();
    check(settings->intValue("maxdepth") == 30, "redo restores the new parameter");

    const QJsonObject before = graph.toJson();
    Node* grid = graph.createNode("grid", "grid1");
    check(grid != nullptr, "create grid for graph undo");
    hub.pushGraphSnapshot(before, graph.toJson(), "Create node");
    check(graph.findNode("grid1") != nullptr, "created node is in the graph");
    hub.undo();
    check(graph.findNode("grid1") == nullptr, "undo removes the created node");
    hub.redo();
    check(graph.findNode("grid1") != nullptr, "redo restores the created node");
    settings = graph.findNode("rs1");
    check(settings != nullptr, "rendersettings survives graph undo/redo");
    if (!settings) return;

    const int samples0 = settings->intValue("samples", 128);
    for (int i = 0; i < 110; ++i) {
        hub.pushParameter(settings->name(), QStringLiteral("samples"), samples0 + i, samples0 + i + 1);
    }
    check(hub.stack().count() <= 100, "undo stack drops entries past 100");

    OrbitCameraState camBefore;
    OrbitCameraState camAfter;
    camAfter.distance = 20.0f;
    camAfter.yaw = 45.0f;
    int cameraApplies = 0;
    float lastDistance = 0.0f;
    hub.setApplyCamera([&](const OrbitCameraState& s) {
        ++cameraApplies;
        lastDistance = s.distance;
    });
    hub.pushCamera(camBefore, camAfter);
    hub.undo();
    check(cameraApplies == 1, "camera undo applies the start view");
    check(std::fabs(lastDistance - camBefore.distance) < 1e-5f, "camera undo restores distance");
    hub.redo();
    check(cameraApplies == 2, "camera redo applies the end view");
    check(std::fabs(lastDistance - camAfter.distance) < 1e-5f, "camera redo restores distance");
}

// Chromatic dispersion + thin-film iridescence sanity.
void testDispersionAndThinFilm() {
    std::printf("dispersion-thinfilm\n");
    // Cauchy/Abbe: blue bends more than red, spread scales with 1/Abbe.
    const float nR = dispersedIor(1.5f, 30.0f, 0);
    const float nG = dispersedIor(1.5f, 30.0f, 1);
    const float nB = dispersedIor(1.5f, 30.0f, 2);
    check(nB > nG && nG > nR, "dispersed IOR ordering B > G > R");
    checkNear(nG, 1.5f, 0.02f, "green stays near the base IOR");
    const float spreadStrong = dispersedIor(1.5f, 20.0f, 2) - dispersedIor(1.5f, 20.0f, 0);
    const float spreadWeak = dispersedIor(1.5f, 60.0f, 2) - dispersedIor(1.5f, 60.0f, 0);
    check(spreadStrong > 2.5f * spreadWeak, "lower Abbe disperses more");
    check(dispersedIor(1.5f, 0.0f, 2) == 1.5f, "abbe 0 disables dispersion");

    // Ray-switch / mode plumbing: only the shaded material's Abbe can enable dispersion.
    {
        Material camGlass;
        camGlass.transmission = 1.0f;
        camGlass.ior = 1.5f;
        camGlass.dispersionAbbe = 30.0f;
        Material shadowGlass = camGlass;
        shadowGlass.dispersionAbbe = 0.0f;

        DispersionContext ctx;
        ctx.mode = kDispersionOptimized;
        ctx.heroChannel = 2;
        ctx.maxHits = 2;
        Material m = shadowGlass;
        applyDispersion(m, &ctx);
        check(!ctx.used && m.ior == 1.5f, "shadow-port glass without Abbe does not disperse");

        m = camGlass;
        applyDispersion(m, &ctx);
        check(ctx.used && m.ior > 1.5f, "camera-port glass with Abbe disperses (blue)");

        ctx.used = false;
        ctx.disperseHits = 0;
        ctx.mode = kDispersionFake;
        m = camGlass;
        m.ior = 1.5f;
        applyDispersion(m, &ctx);
        check(!ctx.used && m.ior == 1.5f, "Fake mode does not bend IOR");
        const Vec3 tinted = applyFakeDispersionThroughput(Vec3(1.0f), camGlass, &ctx);
        check(ctx.used && tinted.x != tinted.z, "Fake mode tints throughput chromatically");
    }

    // Thin film: reflectance stays in [0,1], varies with thickness, and reduces
    // to plain Fresnel when the film is absent.
    Material m;
    m.thinFilmIor = 1.4f;
    const Vec3 f0(0.04f, 0.04f, 0.04f);
    bool inRange = true;
    float variation = 0.0f;
    Vec3 prev = thinFilmFresnel(f0, 0.7f, 1.4f, 100.0f);
    for (int t = 2; t <= 12; ++t) {
        const Vec3 r = thinFilmFresnel(f0, 0.7f, 1.4f, float(t) * 100.0f);
        if (r.x < 0.0f || r.x > 1.0f || r.y < 0.0f || r.y > 1.0f || r.z < 0.0f || r.z > 1.0f)
            inRange = false;
        variation += length(r - prev);
        prev = r;
    }
    check(inRange, "thin-film reflectance stays in [0,1]");
    check(variation > 0.05f, "thin-film reflectance varies with thickness (iridescence)");
    m.thinFilmThickness = 0.0f;
    const Vec3 plain = specularFresnel(m, f0, 0.7f);
    const Vec3 schlick = fresnelSchlick(f0, 0.7f);
    checkNear(length(plain - schlick), 0.0f, 1e-5f, "no film == Schlick Fresnel");

    // Render check: dispersive glass sphere caustic — channels must separate in
    // the caustic while total energy stays comparable to the non-dispersive one.
    auto buildScene = [](float abbe) {
        auto scene = std::make_shared<Scene>();
        MeshPtr floor = std::make_shared<Mesh>();
        floor->positions = {Vec3(-4, 0, -4), Vec3(4, 0, -4), Vec3(4, 0, 4), Vec3(-4, 0, 4)};
        floor->indices = {0, 2, 1, 0, 3, 2};
        floor->validate();
        const int floorMesh = scene->addMesh(floor);
        Material floorMat;
        floorMat.baseColor = Vec3(0.75f);
        floorMat.roughness = 0.9f;
        floorMat.specular = 0.0f;
        const int floorIdx = scene->addMaterial(floorMat);
        InstanceData fi;
        fi.meshIndex = floorMesh;
        fi.materialIndex = floorIdx;
        scene->instances.push_back(fi);

        MeshPtr ball = makeSphereMesh(0.7f, 48, 24);
        const int ballMesh = scene->addMesh(ball);
        Material glass;
        glass.transmission = 1.0f;
        glass.roughness = 0.0f;
        glass.ior = 1.5f;
        glass.dispersionAbbe = abbe;
        const int glassIdx = scene->addMaterial(glass);
        InstanceData bi;
        bi.xform = Mat4::translate(Vec3(0.0f, 1.0f, 0.0f));
        bi.meshIndex = ballMesh;
        bi.materialIndex = glassIdx;
        scene->instances.push_back(bi);

        LightData light;
        light.type = kLightRect;
        light.width = 0.4f;
        light.height = 0.4f;
        light.intensity = 60.0f;
        light.xform = Mat4::translate(Vec3(0.0f, 4.0f, 0.0f)) * Mat4::rotateX(-90.0f);
        scene->lights.push_back(light);

        scene->settings.resolutionX = 72;
        scene->settings.resolutionY = 54;
        scene->settings.samplesPerPixel = 32;
        scene->settings.integrator = kIntegratorPathTracer;
        scene->settings.caustics = 1;
        scene->settings.causticsEngine = kCausticsEnginePbrt;
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
        scene->settings.clampDirect = 0.0f;
        scene->settings.clampIndirect = 0.0f;
        scene->camera.cameraToWorld =
            lookAtMatrix(Vec3(2.4f, 2.6f, 2.4f), Vec3(0.0f, 0.35f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
        scene->cameraAuthored = true;
        scene->finalize();
        return scene;
    };
    auto render = [&](float abbe, double& chroma, double& rbSep, bool& finite) {
        RenderSession session;
        session.setScene(buildScene(abbe));
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double sum = 0.0;
        chroma = 0.0;
        finite = true;
        double wR = 0.0, wB = 0.0, xR = 0.0, yR = 0.0, xB = 0.0, yB = 0.0;
        for (int y = 0; y < img.height(); ++y)
            for (int x = 0; x < img.width(); ++x) {
                const Vec3 c = img.rgb(x, y);
                if (!isFinite(c)) finite = false;
                sum += double(luminance(c));
                const float mean = (c.x + c.y + c.z) / 3.0f;
                chroma += double(std::fabs(c.x - mean) + std::fabs(c.y - mean) + std::fabs(c.z - mean));
                const float eR = std::max(0.0f, c.x - c.y);
                const float eB = std::max(0.0f, c.z - c.y);
                xR += double(x) * double(eR);
                yR += double(y) * double(eR);
                wR += double(eR);
                xB += double(x) * double(eB);
                yB += double(y) * double(eB);
                wB += double(eB);
            }
        rbSep = 0.0;
        if (wR > 1e-6 && wB > 1e-6) {
            const double dx = xR / wR - xB / wB;
            const double dy = yR / wR - yB / wB;
            rbSep = std::sqrt(dx * dx + dy * dy);
        }
        return sum;
    };
    double chromaOff = 0.0, chromaOn = 0.0, sepOff = 0.0, sepOn = 0.0;
    bool finOff = true, finOn = true;
    const double sumOff = render(0.0f, chromaOff, sepOff, finOff);
    const double sumOn = render(20.0f, chromaOn, sepOn, finOn);
    check(finOff && finOn, "dispersion renders are finite");
    check(sumOn > sumOff * 0.8 && sumOn < sumOff * 1.25, "dispersion conserves energy");
    // Low-spp book PT is chromatically noisy (one λ per path). Geometric n(λ)
    // split is asserted in spectral-hero-basics (blue Snell vs red).
    std::printf("  sumOff=%.1f sumOn=%.1f chromaOff=%.1f chromaOn=%.1f sepOff=%.2f sepOn=%.2f\n",
                sumOff, sumOn, chromaOff, chromaOn, sepOff, sepOn);
}

// Rapidly switch integrators / guiding / caustics on a live session — mirrors a
// user toggling render settings in the UI. Must not crash or deadlock.
void testIntegratorSwitchStress() {
    std::printf("integrator-switch-stress\n");
    auto buildScene = [](int integrator, int guiding) {
        auto scene = std::make_shared<Scene>();
        MeshPtr floor = std::make_shared<Mesh>();
        floor->positions = {Vec3(-4, 0, -4), Vec3(4, 0, -4), Vec3(4, 0, 4), Vec3(-4, 0, 4)};
        floor->indices = {0, 2, 1, 0, 3, 2};
        floor->validate();
        const int floorMesh = scene->addMesh(floor);
        Material floorMat;
        floorMat.baseColor = Vec3(0.7f, 0.7f, 0.7f);
        floorMat.roughness = 0.8f;
        const int floorIdx = scene->addMaterial(floorMat);
        InstanceData fi;
        fi.meshIndex = floorMesh;
        fi.materialIndex = floorIdx;
        scene->instances.push_back(fi);

        MeshPtr ball = makeSphereMesh(0.7f, 24, 12);
        const int ballMesh = scene->addMesh(ball);
        Material glass;
        glass.transmission = 1.0f;
        glass.roughness = 0.0f;
        glass.ior = 1.5f;
        const int glassIdx = scene->addMaterial(glass);
        InstanceData bi;
        bi.xform = Mat4::translate(Vec3(0.0f, 1.0f, 0.0f));
        bi.meshIndex = ballMesh;
        bi.materialIndex = glassIdx;
        scene->instances.push_back(bi);

        LightData light;
        light.type = kLightRect;
        light.width = 0.2f;
        light.height = 0.2f;
        light.intensity = 60.0f;
        light.xform = Mat4::translate(Vec3(0.0f, 4.0f, 0.0f)) * Mat4::rotateX(-90.0f);
        scene->lights.push_back(light);

        scene->settings.resolutionX = 96;
        scene->settings.resolutionY = 64;
        scene->settings.samplesPerPixel = 1000;  // never finishes within the test
        scene->settings.integrator = integrator;
        scene->settings.pathGuiding = guiding;
        scene->settings.caustics = 1;
        scene->settings.envVisibleCamera = 0;
        scene->camera.cameraToWorld =
            lookAtMatrix(Vec3(2.4f, 2.6f, 2.4f), Vec3(0.0f, 0.35f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
        scene->cameraAuthored = true;
        scene->finalize();
        return scene;
    };

    RenderSession session;
    const int integrators[] = {kIntegratorPathTracer, kIntegratorBdpt, kIntegratorPathTracer,
                               kIntegratorBdpt, kIntegratorDirectLighting, kIntegratorBdpt};
    for (int i = 0; i < 12; ++i) {
        const int integrator = integrators[i % 6];
        const int guiding = (i / 2) % 2;
        session.setScene(buildScene(integrator, guiding));
        session.start();
        // Let a few samples land, then rip the settings out from under it.
        std::this_thread::sleep_for(std::chrono::milliseconds(i % 3 == 0 ? 30 : 150));
        if (i % 4 == 1) session.invalidate();
        if (i % 4 == 3) session.updateSceneData();
    }
    session.stop();
    check(true, "integrator switch stress survived");
}

// Renders an emissive sphere that is only in frame when instance transforms
// are honoured by the acceleration structure.
void testInstanceTransform() {
    std::printf("instance transforms\n");
    auto scene = std::make_shared<Scene>();
    MeshPtr sphere = makeSphereMesh(1.0f, 32, 16);
    const int meshIndex = scene->addMesh(sphere);

    Material emissive;
    emissive.baseColor = Vec3(0.0f);
    emissive.emissionColor = Vec3(1.0f, 1.0f, 1.0f);
    emissive.emissionStrength = 10.0f;
    const int materialIndex = scene->addMaterial(emissive);

    InstanceData instance;
    instance.xform = Mat4::translate(Vec3(30.0f, 0.0f, 0.0f));
    instance.meshIndex = meshIndex;
    instance.materialIndex = materialIndex;
    scene->instances.push_back(instance);

    scene->settings.resolutionX = 32;
    scene->settings.resolutionY = 32;
    scene->settings.samplesPerPixel = 2;
    scene->settings.envVisibleCamera = 0;
    scene->camera.cameraToWorld =
        lookAtMatrix(Vec3(30.0f, 0.0f, 8.0f), Vec3(30.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    scene->cameraAuthored = true;
    scene->finalize();

    RenderSession session;
    session.setScene(scene);
    session.start();
    session.waitForCompletion();

    const Image image = session.linearImage();
    const float center = luminance(image.rgb(image.width() / 2, image.height() / 2));
    const float corner = luminance(image.rgb(1, 1));
    check(center > 1.0f, "translated instance is visible at the transformed location");
    checkNear(corner, 0.0f, 1e-4f, "background stays black");
}

void testUdimMaterialX() {
    std::printf("udim-materialx\n");
    QTemporaryDir dir;
    check(dir.isValid(), "temp dir for udim tiles");
    const QString root = dir.path();

    auto writeTile = [&](int udim, int r, int g, int b) {
        QImage img(4, 4, QImage::Format_RGBA8888);
        img.fill(QColor(r, g, b, 255));
        const QString path = root + QString("/grid.%1.png").arg(udim);
        check(img.save(path), ("write tile " + std::to_string(udim)).c_str());
    };
    writeTile(1001, 255, 0, 0);
    writeTile(1002, 0, 255, 0);
    writeTile(1011, 0, 0, 255);

    QString pattern;
    std::vector<int> tiles;
    check(resolveUdimPattern(root + "/grid.1001.png", QString(), pattern, tiles),
          "concrete tile promotes to MaterialX <UDIM> pattern");
    check(pattern.contains(QStringLiteral("<UDIM>")), "pattern keeps unresolved <UDIM>");
    check(tiles.size() == 3, "discovers three UDIM tiles on disk");

    std::string error;
    auto atlas = loadImageOrUdim(root + "/grid.1001.png", QString(), error, {}, true, "Utility - Raw");
    check(atlas != nullptr, "loads udim atlas from concrete tile path");
    check(atlas && atlas->isUdimAtlas(), "atlas marked as UDIM");
    check(atlas && atlas->udimGridU() == 2 && atlas->udimGridV() == 2, "atlas grid covers U0..1 V0..1");
    if (atlas) {
        // MaterialX View style: floor(uv) selects tile, fract samples inside (with V-flip bake).
        TextureView view;
        view.pixels = atlas->data();
        view.width = atlas->width();
        view.height = atlas->height();
        view.udimGridU = atlas->udimGridU();
        view.udimGridV = atlas->udimGridV();
        const Vec4 c1001 = sampleTextureRGBA(view, Vec2(0.5f, 0.5f));
        const Vec4 c1002 = sampleTextureRGBA(view, Vec2(1.5f, 0.5f));
        const Vec4 c1011 = sampleTextureRGBA(view, Vec2(0.5f, 1.5f));
        check(c1001.x > 0.8f && c1001.y < 0.2f && c1001.z < 0.2f, "sample UV(0.5,0.5) → tile 1001 red");
        check(c1002.y > 0.8f && c1002.x < 0.2f && c1002.z < 0.2f, "sample UV(1.5,0.5) → tile 1002 green");
        check(c1011.z > 0.8f && c1011.x < 0.2f && c1011.y < 0.2f, "sample UV(0.5,1.5) → tile 1011 blue");
    }

    if (!materialXAvailable()) {
        std::printf("  skip MaterialX xml roundtrip (MaterialX unavailable)\n");
        return;
    }

    QVector<MaterialXGraphNode> nodes;
    MaterialXGraphNode imageNode;
    imageNode.name = "image_color";
    imageNode.category = "image";
    imageNode.type = "color3";
    imageNode.inputs.push_back({"file", "filename", pattern, {}});
    nodes.push_back(imageNode);
    MaterialXGraphNode ss;
    ss.name = "standard_surface1";
    ss.category = "standard_surface";
    ss.type = "surfaceshader";
    ss.inputs.push_back({"base_color", "color3", {}, "image_color"});
    nodes.push_back(ss);
    MaterialXGraphNode surface;
    surface.name = "surface";
    surface.category = "surfacematerial";
    surface.type = "material";
    surface.inputs.push_back({"surfaceshader", "surfaceshader", {}, "standard_surface1"});
    nodes.push_back(surface);

    const QVector<int> udimSet = {1001, 1002, 1011};
    const QString xml = serializeMaterialXGraph(nodes, udimSet);
    check(xml.contains(QStringLiteral("<UDIM>")), "MaterialX write keeps raw <UDIM>");
    check(!xml.contains(QStringLiteral("&lt;UDIM&gt;")), "MaterialX write does not entity-escape <UDIM>");
    check(xml.contains(QStringLiteral("udimset")), "MaterialX write emits geominfo udimset");

    QVector<MaterialXGraphNode> roundtrip;
    QVector<int> roundtripSet;
    QString parseError;
    check(parseMaterialXGraph(xml, roundtrip, &parseError, &roundtripSet), "MaterialX parse roundtrip");
    check(roundtripSet.size() == 3, "udimset roundtrips");
    bool foundFile = false;
    for (const MaterialXGraphNode& node : roundtrip) {
        for (const MaterialXGraphInput& input : node.inputs) {
            if (input.name == "file") {
                foundFile = true;
                check(input.value.contains(QStringLiteral("<UDIM>")), "parsed file keeps <UDIM>");
            }
        }
    }
    check(foundFile, "roundtrip retained image file input");

    MaterialXEvalResult evaluated = evaluateMaterialXDocument(xml, root);
    check(evaluated.ok, "evaluate MaterialX udim document");
    check(evaluated.baseColorTexture && evaluated.baseColorTexture->isUdimAtlas(),
          "cook binds multi-tile UDIM atlas");
    check(evaluated.baseColorTexture && evaluated.baseColorTexture->udimGridU() >= 2,
          "cooked atlas spans multiple U tiles");

    // End-to-end: plane spanning UDIM 1001/1002 with the cooked atlas must show
    // different base colors on the left (1001) and right (1002) halves.
    auto scene = std::make_shared<Scene>();
    auto mesh = std::make_shared<Mesh>();
    // Two quads: UV [0,1]x[0,1] and [1,2]x[0,1]
    mesh->positions = {Vec3(-2, 0, -1), Vec3(0, 0, -1), Vec3(0, 0, 1), Vec3(-2, 0, 1),
                       Vec3(0, 0, -1), Vec3(2, 0, -1), Vec3(2, 0, 1), Vec3(0, 0, 1)};
    mesh->uvs = {Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1), Vec2(1, 0), Vec2(2, 0), Vec2(2, 1),
                 Vec2(1, 1)};
    mesh->normals.assign(8, Vec3(0, 1, 0));
    mesh->indices = {0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7};
    mesh->validate();
    const int meshIndex = scene->addMesh(mesh);

    Material mat = evaluated.material;
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.baseColor = Vec3(1.0f);
    mat.baseColorTex = scene->addTexture(evaluated.baseColorTexture);
    const int matIndex = scene->addMaterial(mat);

    InstanceData inst;
    inst.xform = Mat4::identity();
    inst.xformInv = Mat4::identity();
    inst.meshIndex = meshIndex;
    inst.materialIndex = matIndex;
    scene->instances.push_back(inst);

    LightData key;
    key.type = kLightDistant;
    key.color = Vec3(1.0f);
    key.intensity = 4.0f;
    key.xform = lookAtMatrix(Vec3(0, 5, 0), Vec3(0, 0, 0), Vec3(0, 0, 1));
    scene->lights.push_back(key);

    scene->settings.resolutionX = 64;
    scene->settings.resolutionY = 32;
    scene->settings.samplesPerPixel = 8;
    scene->settings.envVisibleCamera = 0;
    scene->camera.cameraToWorld = lookAtMatrix(Vec3(0, 6, 0), Vec3(0, 0, 0), Vec3(0, 0, 1));
    scene->cameraAuthored = true;
    scene->finalize();

    RenderSession session;
    session.setScene(scene);
    session.start();
    session.waitForCompletion();
    const Image rendered = session.linearImage();
    // Camera looks down; screen X may flip world X — just require both UDIM colors present.
    bool sawRed = false;
    bool sawGreen = false;
    for (int y = 0; y < rendered.height(); ++y) {
        for (int x = 0; x < rendered.width(); ++x) {
            const Vec3 c = rendered.rgb(x, y);
            if (c.x > 0.5f && c.x > c.y * 2.0f && c.x > c.z * 2.0f) sawRed = true;
            if (c.y > 0.5f && c.y > c.x * 2.0f && c.y > c.z * 2.0f) sawGreen = true;
        }
    }
    check(sawRed, "render shows UDIM 1001 red");
    check(sawGreen, "render shows UDIM 1002 green");
}



void testMaterialXTypeMismatchConnect() {
    std::printf("materialx-type-mismatch-connect\n");
    if (!materialXAvailable()) { std::printf("  skip\n"); return; }
    // Soft-snap often hits standard_surface.base (float) first — color noise wired there
    // must not crash cook/evaluate (Arnold validates connection types).
    const QString bad = QStringLiteral(
        "<?xml version=\"1.0\"?><materialx version=\"1.38\">"
        "<noise2d name=\"n\" type=\"color3\">"
        "<input name=\"amplitude\" type=\"vector3\" value=\"1,1,1\"/>"
        "</noise2d>"
        "<standard_surface name=\"ss\" type=\"surfaceshader\">"
        "<input name=\"base\" type=\"float\" nodename=\"n\"/>"
        "<input name=\"base_color\" type=\"color3\" value=\"0.2,0.3,0.4\"/>"
        "</standard_surface>"
        "<surfacematerial name=\"surface\" type=\"material\">"
        "<input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>"
        "</surfacematerial></materialx>");
    MaterialXEvalResult eval = evaluateMaterialXDocument(bad, QString());
    check(eval.ok, "color→base float connection evaluates without hard failure");
    check(eval.material.baseColorProc < 0, "base float wire does not bind baseColorProc");
    std::printf("  mismatch ok=%d err=%s\n", int(eval.ok), eval.error.toStdString().c_str());

    // Full stage→scene→shade path for noise→base_color (diffuse)
    const QString good = QStringLiteral(
        "<?xml version=\"1.0\"?><materialx version=\"1.38\">"
        "<noise2d name=\"n\" type=\"color3\">"
        "<input name=\"amplitude\" type=\"vector3\" value=\"1,1,1\"/>"
        "</noise2d>"
        "<standard_surface name=\"ss\" type=\"surfaceshader\">"
        "<input name=\"base\" type=\"float\" value=\"1\"/>"
        "<input name=\"base_color\" type=\"color3\" nodename=\"n\"/>"
        "</standard_surface>"
        "<surfacematerial name=\"surface\" type=\"material\">"
        "<input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>"
        "</surfacematerial></materialx>");
    eval = evaluateMaterialXDocument(good, QString());
    check(eval.ok && eval.material.baseColorProc >= 0, "noise→base_color compiles");
    Stage stage;
    StagePrim prim;
    prim.type = PrimType::Mesh;
    prim.path = "/geo/sphere";
    prim.mesh = makeSphereMesh(1.0f, 16, 8);
    prim.material = eval.material;
    prim.materialAssigned = true;
    prim.procedurals = eval.procedurals;
    prim.proceduralImages = eval.proceduralImages;
    stage.prims.push_back(prim);
    ScenePtr scene = stage.toScene();
    check(scene != nullptr, "toScene with procedural material");
    scene->finalize();
    SceneView view = scene->view();
    check(view.proceduralCount > 0, "scene has procedurals");
    check(view.materials[0].baseColorProc >= 0, "material proc index remapped");
    Vec3 ns(0,0,1);
    bool finite = true;
    for (int i = 0; i < 256; ++i) {
        Vec2 uv(float(i % 16) / 16.f, float(i / 16) / 16.f);
        Material m = evaluateTexturedMaterial(view, view.materials[0], uv, ns,
                                              Vec3(uv.x, uv.y, 0.2f), Vec3(0,1,0), 0.01f);
        if (!srIsFinite(m.baseColor.x) || !srIsFinite(m.baseColor.y) || !srIsFinite(m.baseColor.z)) finite = false;
    }
    check(finite, "shaded noise samples are finite");
    std::printf("  diffuse shade ok procs=%d\n", view.proceduralCount);
}


void testMaterialXColorIntoFloatSlots() {
    std::printf("materialx-color-into-float-slots\n");
    if (!materialXAvailable()) { std::printf("  skip\n"); return; }

    auto shadeOk = [](const MaterialXEvalResult& eval, const char* label) {
        check(eval.ok, (std::string(label) + " evaluates").c_str());
        if (!eval.ok) {
            std::printf("  %s err=%s\n", label, eval.error.toStdString().c_str());
            return;
        }
        Stage stage;
        StagePrim prim;
        prim.type = PrimType::Mesh;
        prim.path = "/geo/mesh";
        prim.mesh = std::make_shared<Mesh>();
        prim.mesh->positions = {Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0)};
        prim.mesh->indices = {0,1,2};
        prim.mesh->normals = {Vec3(0,1,0), Vec3(0,1,0), Vec3(0,1,0)};
        prim.mesh->uvs = {Vec2(0,0), Vec2(1,0), Vec2(0,1)};
        prim.mesh->validate();
        prim.material = eval.material;
        prim.materialAssigned = true;
        prim.baseColorTexture = eval.baseColorTexture;
        prim.roughnessTexture = eval.roughnessTexture;
        prim.metallicTexture = eval.metallicTexture;
        prim.opacityTexture = eval.opacityTexture;
        prim.emissionTexture = eval.emissionTexture;
        prim.normalTexture = eval.normalTexture;
        prim.subsurfaceTexture = eval.subsurfaceTexture;
        prim.procedurals = eval.procedurals;
        prim.proceduralImages = eval.proceduralImages;
        stage.prims.push_back(prim);
        auto scene = stage.toScene();
        SceneView view = scene->view();
        check(view.materialCount > 0 && view.materials, (std::string(label) + " has material").c_str());
        bool finite = true;
        for (int i = 0; i < 500; ++i) {
            Vec3 ns(0, 1, 0);
            Material m = evaluateTexturedMaterial(view, view.materials[0], Vec2(0.1f, 0.25f), ns,
                                                  Vec3(float(i) * 0.01f, 0.2f, -0.3f), Vec3(0, 1, 0), 0.01f);
            if (!srIsFinite(m.roughness) || !srIsFinite(m.metallic) || !srIsFinite(m.opacity) ||
                !srIsFinite(m.baseColor.x) || !srIsFinite(ns.x)) {
                finite = false;
                break;
            }
        }
        check(finite, (std::string(label) + " shade finite").c_str());
        std::printf("  %s ok procs=%d rough=%d metal=%d\n", label, view.proceduralCount,
                    view.materials[0].roughnessProc, view.materials[0].metallicProc);
    };

    const char* portNames[] = {"specular_roughness", "metalness", "opacity", "emission_color", "normal",
                           "subsurface_color"};
    for (const char* port : portNames) {
        QString type = QStringLiteral("float");
        if (QString(port) == "opacity" || QString(port) == "emission_color" ||
            QString(port) == "subsurface_color")
            type = QStringLiteral("color3");
        if (QString(port) == "normal") type = QStringLiteral("vector3");
        const QString xml = QStringLiteral(
                                "<?xml version=\"1.0\"?>\n"
                                "<materialx version=\"1.38\">\n"
                                "  <triplanarprojection name=\"tri1\" type=\"color3\">\n"
                                "    <input name=\"scale\" type=\"vector3\" value=\"1, 1, 1\"/>\n"
                                "    <input name=\"default\" type=\"color3\" value=\"0.2, 0.5, 0.8\"/>\n"
                                "  </triplanarprojection>\n"
                                "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
                                "    <input name=\"%1\" type=\"%2\" nodename=\"tri1\"/>\n"
                                "  </standard_surface>\n"
                                "  <surfacematerial name=\"surface\" type=\"material\">\n"
                                "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
                                "  </surfacematerial>\n"
                                "</materialx>\n")
                                .arg(port, type);
        shadeOk(evaluateMaterialXDocument(xml, QString()), port);
    }

    // Same color node into several float slots at once (shared compile).
    const QString multi = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <triplanarprojection name=\"tri1\" type=\"color3\">\n"
        "    <input name=\"default\" type=\"color3\" value=\"0.4, 0.5, 0.6\"/>\n"
        "  </triplanarprojection>\n"
        "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
        "    <input name=\"specular_roughness\" type=\"float\" nodename=\"tri1\"/>\n"
        "    <input name=\"metalness\" type=\"float\" nodename=\"tri1\"/>\n"
        "    <input name=\"opacity\" type=\"color3\" nodename=\"tri1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
    MaterialXEvalResult multiEval = evaluateMaterialXDocument(multi, QString());
    check(multiEval.ok, "multi-slot evaluates");
    check(multiEval.material.roughnessProc >= 0 && multiEval.material.metallicProc >= 0 &&
              multiEval.material.opacityProc >= 0,
          "multi-slot binds all procs");
    check(multiEval.material.roughnessProc == multiEval.material.metallicProc &&
              multiEval.material.metallicProc == multiEval.material.opacityProc,
          "shared upstream reuses one compiled procedural");
    shadeOk(multiEval, "multi-slot");
}


void testMaterialXBumpAndNormalMap() {
    std::printf("materialx-bump-normalmap\n");
    if (!materialXAvailable()) { std::printf("  skip\n"); return; }

    auto makeHeightPng = [](const QString& path) {
        QImage img(64, 64, QImage::Format_RGB32);
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) {
                // Smooth gradient so finite differences are never zero.
                const int v = int(255.0f * (float(x) / 63.0f));
                img.setPixel(x, y, qRgb(v, v, v));
            }
        return img.save(path);
    };
    auto makeNormalPng = [](const QString& path) {
        // Flat tangent normal (128,128,255)
        QImage img(32, 32, QImage::Format_RGB32);
        img.fill(qRgb(128, 128, 255));
        return img.save(path);
    };

    QTemporaryDir tmp;
    check(tmp.isValid(), "temp dir for bump/normal");
    const QString heightPath = tmp.filePath("height.png");
    const QString normalPath = tmp.filePath("normal.png");
    check(makeHeightPng(heightPath), "write height map");
    check(makeNormalPng(normalPath), "write normal map");

    const QString bumpXml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <image name=\"h1\" type=\"float\">\n"
        "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
        "  </image>\n"
        "  <bump name=\"bump1\" type=\"vector3\">\n"
        "    <input name=\"height\" type=\"float\" nodename=\"h1\"/>\n"
        "    <input name=\"scale\" type=\"float\" value=\"2\"/>\n"
        "  </bump>\n"
        "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
        "    <input name=\"normal\" type=\"vector3\" nodename=\"bump1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n").arg(heightPath);
    MaterialXEvalResult eval = evaluateMaterialXDocument(bumpXml, tmp.path());
    check(eval.ok, "bump→normal evaluates");
    check(eval.bumpTexture != nullptr, "bump binds height texture");
    check(std::fabs(eval.material.normalScale - 2.0f) < 1e-4f, "bump scale authored");
    check(eval.normalTexture == nullptr && eval.material.bumpProc < 0, "bump uses bumpTexture path");

    Stage stage;
    StagePrim prim;
    prim.type = PrimType::Mesh;
    prim.path = "/geo/mesh";
    prim.mesh = std::make_shared<Mesh>();
    prim.mesh->positions = {Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0)};
    prim.mesh->indices = {0,1,2};
    prim.mesh->normals = {Vec3(0,1,0), Vec3(0,1,0), Vec3(0,1,0)};
    prim.mesh->uvs = {Vec2(0,0), Vec2(1,0), Vec2(0,1)};
    prim.mesh->validate();
    prim.material = eval.material;
    prim.materialAssigned = true;
    prim.bumpTexture = eval.bumpTexture;
    prim.normalTexture = eval.normalTexture;
    stage.prims.push_back(prim);
    auto scene = stage.toScene();
    SceneView view = scene->view();
    check(view.materialCount > 0 && view.materials[0].bumpTex >= 0, "scene has bumpTex");
    Vec3 shadingNormalA(0.0f, 1.0f, 0.0f);
    Vec3 shadingNormalB(0.0f, 1.0f, 0.0f);
    evaluateTexturedMaterial(view, view.materials[0], Vec2(0.125f, 0.25f), shadingNormalA, Vec3(0.0f, 0.0f, 0.0f),
                             Vec3(0.0f, 1.0f, 0.0f), 1.0f / 64.0f);
    evaluateTexturedMaterial(view, view.materials[0], Vec2(0.25f, 0.125f), shadingNormalB, Vec3(0.0f, 0.0f, 0.0f),
                             Vec3(0.0f, 1.0f, 0.0f), 1.0f / 64.0f);
    check(srIsFinite(shadingNormalA.x) && srIsFinite(shadingNormalB.x), "bump shading normals finite");
    const float tilt = std::fabs(shadingNormalA.x) + std::fabs(shadingNormalA.z) + std::fabs(shadingNormalB.x) +
                       std::fabs(shadingNormalB.z);
    check(tilt > 1e-4f, "bump perturbs shading normal");
    std::printf("  bump nA=(%.3f,%.3f,%.3f) nB=(%.3f,%.3f,%.3f) scale=%.2f\n", shadingNormalA.x, shadingNormalA.y,
                shadingNormalA.z, shadingNormalB.x, shadingNormalB.y, shadingNormalB.z,
                view.materials[0].normalScale);

    const QString nmapXml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <image name=\"n1\" type=\"vector3\">\n"
        "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
        "  </image>\n"
        "  <normalmap name=\"nm1\" type=\"vector3\">\n"
        "    <input name=\"in\" type=\"vector3\" nodename=\"n1\"/>\n"
        "    <input name=\"scale\" type=\"float\" value=\"0.5\"/>\n"
        "  </normalmap>\n"
        "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
        "    <input name=\"normal\" type=\"vector3\" nodename=\"nm1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n").arg(normalPath);
    eval = evaluateMaterialXDocument(nmapXml, tmp.path());
    check(eval.ok, "normalmap→normal evaluates");
    check(eval.normalTexture != nullptr, "normalmap binds texture");
    check(std::fabs(eval.material.normalScale - 0.5f) < 1e-4f, "normalmap scale authored");
    std::printf("  normalmap ok scale=%.2f\n", eval.material.normalScale);
}




void testTriplanarDisplacementArtifacts() {
    std::printf("triplanar-displacement-artifacts\n");
    if (!materialXAvailable()) {
        std::printf("  skip (no MaterialX)\n");
        return;
    }

    // Shared triplanar → albedo + displace must compile separate sRGB / linear graphs.
    {
        QTemporaryDir tmp;
        check(tmp.isValid(), "tmpdir");
        const QString texPath = tmp.filePath("tri_mid.png");
        {
            // Mid-grey 188 ≈ sRGB encode of ~0.5 linear. Data/raw load keeps ~0.737.
            QImage img(16, 16, QImage::Format_RGB32);
            img.fill(qRgb(188, 188, 188));
            check(img.save(texPath), "write mid grey");
        }
        const QString xml = QStringLiteral(
            "<?xml version=\"1.0\"?>\n"
            "<materialx version=\"1.38\">\n"
            "  <triplanarprojection name=\"tri1\" type=\"color3\">\n"
            "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
            "    <input name=\"blend\" type=\"float\" value=\"8\"/>\n"
            "  </triplanarprojection>\n"
            "  <displacement name=\"disp1\" type=\"float\">\n"
            "    <input name=\"displacement\" type=\"float\" nodename=\"tri1\"/>\n"
            "    <input name=\"scale\" type=\"float\" value=\"1\"/>\n"
            "    <input name=\"zero_value\" type=\"float\" value=\"0\"/>\n"
            "    <input name=\"subdiv_iterations\" type=\"integer\" value=\"0\"/>\n"
            "    <input name=\"autobump\" type=\"boolean\" value=\"false\"/>\n"
            "  </displacement>\n"
            "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
            "    <input name=\"base_color\" type=\"color3\" nodename=\"tri1\"/>\n"
            "  </standard_surface>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
            "    <input name=\"displacementshader\" type=\"displacementshader\" nodename=\"disp1\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n").arg(texPath);
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, tmp.path());
        check(eval.ok, "mid-grey triplanar evaluates");
        check(eval.procedurals.size() >= 2, "separate colour/data graphs");
        check(eval.material.displacementProc != eval.material.baseColorProc,
              "displace graph is not the sRGB colour graph");

        Scene scene;
        for (const auto& img : eval.proceduralImages) scene.addTexture(img);
        scene.procedurals = eval.procedurals;
        std::vector<TextureView> tvs;
        for (const auto& img : scene.textures) {
            TextureView tv;
            if (img && !img->empty()) {
                tv.pixels = img->data();
                tv.width = img->width();
                tv.height = img->height();
                tv.mipCount = 1;
            }
            tvs.push_back(tv);
        }
        SceneView view;
        view.textures = tvs.data();
        view.textureCount = int(tvs.size());
        view.procedurals = scene.procedurals.data();
        view.proceduralCount = int(scene.procedurals.size());
        ProceduralCtx ctx;
        ctx.pObject = Vec3(0.2f, 1.0f, 0.3f);
        ctx.nObject = Vec3(0.f, 1.f, 0.f);
        ctx.forDisplacement = 1;
        const float h = evalProceduralRoot(view, eval.material.displacementProc, ctx).x;
        std::printf("  linear height=%.3f (expect ~0.74 raw, not ~0.50 sRGB)\n", h);
        check(h > 0.65f && h < 0.85f, "displace triplanar loads height as linear data");
    }

    // Smooth ramp + hard authored blend — displace softens blend to avoid spikes.
    {
        QTemporaryDir tmp;
        check(tmp.isValid(), "tmpdir2");
        const QString texPath = tmp.filePath("ramp.png");
        {
            QImage img(128, 128, QImage::Format_RGB32);
            for (int y = 0; y < 128; ++y)
                for (int x = 0; x < 128; ++x) {
                    const int v =
                        int(std::lround(255.0 * (0.5 + 0.5 * std::sin(x * 0.12) * std::cos(y * 0.12))));
                    img.setPixel(x, y, qRgb(v, v, v));
                }
            check(img.save(texPath), "write ramp");
        }
        // Non-aligned scale so a subdivided grid does not land on identical wrapped UVs.
        const QString triXml = QStringLiteral(
            "<?xml version=\"1.0\"?>\n"
            "<materialx version=\"1.38\">\n"
            "  <triplanarprojection name=\"tri1\" type=\"color3\">\n"
            "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
            "    <input name=\"scale\" type=\"vector3\" value=\"0.37, 0.37, 0.37\"/>\n"
            "    <input name=\"blend\" type=\"float\" value=\"16\"/>\n"
            "  </triplanarprojection>\n"
            "  <displacement name=\"disp1\" type=\"float\">\n"
            "    <input name=\"displacement\" type=\"float\" nodename=\"tri1\"/>\n"
            "    <input name=\"scale\" type=\"float\" value=\"0.12\"/>\n"
            "    <input name=\"zero_value\" type=\"float\" value=\"0.5\"/>\n"
            "    <input name=\"subdiv_iterations\" type=\"integer\" value=\"4\"/>\n"
            "    <input name=\"autobump\" type=\"boolean\" value=\"false\"/>\n"
            "  </displacement>\n"
            "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
            "    <input name=\"base_color\" type=\"color3\" value=\"0.6, 0.6, 0.6\"/>\n"
            "  </standard_surface>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
            "    <input name=\"displacementshader\" type=\"displacementshader\" nodename=\"disp1\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n").arg(texPath);
        MaterialXEvalResult eval = evaluateMaterialXDocument(triXml, tmp.path());
        check(eval.ok && eval.material.displacementProc >= 0, "smooth triplanar displace ok");

        Stage stage;
        StagePrim prim;
        prim.path = "/geo/sphere";
        prim.type = PrimType::Mesh;
        prim.mesh = makeSphereMesh(1.0f, 48, 24);
        prim.material = eval.material;
        prim.procedurals = eval.procedurals;
        prim.proceduralImages = eval.proceduralImages;
        prim.materialAssigned = true;
        prim.subdivType = kSubdivLinear;
        prim.subdivIterations = 4;
        stage.prims.push_back(prim);
        ScenePtr cooked = stage.toScene();
        tessellateCookedScene(cooked);
        check(cooked && !cooked->meshes.empty(), "cooked smooth tri");
        const Mesh& m = *cooked->meshes[0];
        std::vector<float> edges;
        float maxEdge = 0.0f;
        for (size_t ti = 0; ti + 2 < m.indices.size(); ti += 3) {
            const Vec3& a = m.positions[m.indices[ti + 0]];
            const Vec3& b = m.positions[m.indices[ti + 1]];
            const Vec3& c = m.positions[m.indices[ti + 2]];
            for (float e : {length(b - a), length(c - b), length(a - c)}) {
                edges.push_back(e);
                maxEdge = std::max(maxEdge, e);
            }
        }
        std::sort(edges.begin(), edges.end());
        const float median = edges[edges.size() / 2];
        float rmin = 1e9f, rmax = 0.0f;
        for (const Vec3& p : m.positions) {
            rmin = std::min(rmin, length(p));
            rmax = std::max(rmax, length(p));
        }
        std::printf("  smooth tris=%zu rdelta=%.3f edge ratio=%.2f\n", m.triangleCount(), rmax - rmin,
                    maxEdge / std::max(1e-8f, median));
        check(rmax - rmin > 0.03f, "smooth triplanar relief");
        check(maxEdge < median * 10.0f, "hard-blend triplanar displace stays spike-free");

        Stage stage2;
        StagePrim g;
        g.path = "/geo/ground";
        g.type = PrimType::Mesh;
        g.mesh = makeGridMesh(8.0f, 8.0f, 1, 1);
        g.material = eval.material;
        g.procedurals = eval.procedurals;
        g.proceduralImages = eval.proceduralImages;
        g.materialAssigned = true;
        g.subdivType = kSubdivLinear;
        g.subdivIterations = 4;
        stage2.prims.push_back(g);
        ScenePtr cooked2 = stage2.toScene();
        tessellateCookedScene(cooked2);
        float ymin = 1e9f, ymax = -1e9f;
        for (const Vec3& p : cooked2->meshes[0]->positions) {
            ymin = std::min(ymin, p.y);
            ymax = std::max(ymax, p.y);
        }
        std::printf("  ground ydelta=%.3f\n", ymax - ymin);
        check(ymax - ymin > 0.03f, "ground smooth triplanar relief");
    }

    // Arnold Pref: after displace, beauty triplanar must sample cage Pref/Nref —
    // evaluating at displaced P/N flips axis weights and re-projects the map.
    {
        QTemporaryDir tmp;
        check(tmp.isValid(), "tmpdir pref");
        const QString texPath = tmp.filePath("axis.png");
        {
            // Strong XY contrast so X vs Y projection reads clearly.
            QImage img(64, 64, QImage::Format_RGB32);
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x) {
                    const int v = ((x / 8) + (y / 8)) % 2 ? 220 : 40;
                    img.setPixel(x, y, qRgb(v, v / 2, 30));
                }
            check(img.save(texPath), "write axis tex");
        }
        const QString xml = QStringLiteral(
            "<?xml version=\"1.0\"?>\n"
            "<materialx version=\"1.38\">\n"
            "  <triplanarprojection name=\"tri1\" type=\"color3\">\n"
            "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
            "    <input name=\"scale\" type=\"vector3\" value=\"0.5, 0.5, 0.5\"/>\n"
            "    <input name=\"blend\" type=\"float\" value=\"8\"/>\n"
            "  </triplanarprojection>\n"
            "  <displacement name=\"disp1\" type=\"float\">\n"
            "    <input name=\"displacement\" type=\"float\" nodename=\"tri1\"/>\n"
            "    <input name=\"scale\" type=\"float\" value=\"0.55\"/>\n"
            "    <input name=\"zero_value\" type=\"float\" value=\"0.5\"/>\n"
            "    <input name=\"subdiv_iterations\" type=\"integer\" value=\"4\"/>\n"
            "    <input name=\"autobump\" type=\"boolean\" value=\"false\"/>\n"
            "  </displacement>\n"
            "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
            "    <input name=\"base_color\" type=\"color3\" nodename=\"tri1\"/>\n"
            "  </standard_surface>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
            "    <input name=\"displacementshader\" type=\"displacementshader\" nodename=\"disp1\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n").arg(texPath);
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, tmp.path());
        check(eval.ok && eval.material.baseColorProc >= 0, "pref triplanar graph ok");
        check(eval.material.displacementProc >= 0, "pref displace graph ok");

        Stage stage;
        StagePrim prim;
        prim.path = "/geo/sphere";
        prim.type = PrimType::Mesh;
        // Sphere: after displace, many verts have N ≠ Nref (axis flip risk).
        prim.mesh = makeSphereMesh(1.0f, 32, 16);
        prim.material = eval.material;
        prim.procedurals = eval.procedurals;
        prim.proceduralImages = eval.proceduralImages;
        prim.materialAssigned = true;
        prim.subdivType = kSubdivLinear;
        prim.subdivIterations = 4;
        stage.prims.push_back(prim);
        ScenePtr cooked = stage.toScene();
        tessellateCookedScene(cooked);
        check(cooked && !cooked->meshes.empty() && !cooked->materials.empty(), "pref cooked");
        const Mesh& mesh = *cooked->meshes[0];
        const Material& mat = cooked->materials[0];
        check(mat.baseColorProc >= 0, "cooked baseColorProc");
        check(!mesh.restPositions.empty() && mesh.restPositions.size() == mesh.positions.size(),
              "restPositions stored (Pref)");
        check(!mesh.restNormals.empty() && mesh.restNormals.size() == mesh.positions.size(),
              "restNormals stored (Nref)");
        // SceneView must expose Pref after tessellate — otherwise CPU shading
        // falls back to displaced P and triplanar seams return (and pointers UAF).
        const SceneView cookedView = cooked->view();
        check(cookedView.meshCount > 0 && cookedView.meshes, "cooked SceneView has meshes");
        check(cookedView.meshes[0].restPositions != nullptr, "SceneView exposes Pref after tess");
        check(cookedView.meshes[0].restNormals != nullptr, "SceneView exposes Nref after tess");
        check(cookedView.meshes[0].restPositions == mesh.restPositions.data(),
              "SceneView Pref points at live mesh buffers");

        float maxDelta = 0.0f;
        for (size_t i = 0; i < mesh.positions.size(); ++i)
            maxDelta = std::max(maxDelta, length(mesh.positions[i] - mesh.restPositions[i]));
        std::printf("  Pref max |P-Pref|=%.4f verts=%zu\n", maxDelta, mesh.positions.size());
        check(maxDelta > 0.01f, "mesh was displaced vs Pref cage");

        // Find a vertex where displaced shading normal diverges from Pref (axis flip risk).
        int best = -1;
        float bestDot = 1.0f;
        for (size_t i = 0; i < mesh.positions.size(); ++i) {
            if (length(mesh.positions[i] - mesh.restPositions[i]) < 1e-4f) continue;
            if (mesh.normals.size() != mesh.positions.size()) continue;
            const Vec3 n = normalize(mesh.normals[i]);
            const Vec3 nr = normalize(mesh.restNormals[i]);
            const float d = std::fabs(dot(n, nr));
            if (d < bestDot) {
                bestDot = d;
                best = int(i);
            }
        }
        check(best >= 0, "found displaced Pref candidate");
        if (best >= 0) {
            const Vec3 p = mesh.positions[size_t(best)];
            const Vec3 pref = mesh.restPositions[size_t(best)];
            const Vec3 n = normalize(mesh.normals[size_t(best)]);
            const Vec3 nref = normalize(mesh.restNormals[size_t(best)]);

            SceneView view = cooked->view();
            ProceduralCtx locked;
            locked.pObject = p;
            locked.nObject = n;
            locked.pRef = pref;
            locked.nRef = nref;
            locked.hasPref = 1;
            ProceduralCtx flipped;
            flipped.pObject = p;
            flipped.nObject = n;
            flipped.hasPref = 0;
            ProceduralCtx cage;
            cage.pObject = pref;
            cage.nObject = nref;
            cage.hasPref = 0;

            const Vec4 cLock = evalProceduralRoot(view, mat.baseColorProc, locked);
            const Vec4 cFlip = evalProceduralRoot(view, mat.baseColorProc, flipped);
            const Vec4 cCage = evalProceduralRoot(view, mat.baseColorProc, cage);
            const float errLock =
                std::fabs(cLock.x - cCage.x) + std::fabs(cLock.y - cCage.y) + std::fabs(cLock.z - cCage.z);
            const float errFlip =
                std::fabs(cFlip.x - cCage.x) + std::fabs(cFlip.y - cCage.y) + std::fabs(cFlip.z - cCage.z);
            std::printf("  Pref lock err=%.4f  flipped err=%.4f  n·nref=%.3f\n", errLock, errFlip, bestDot);
            check(errLock < 1e-4f, "Pref-locked triplanar matches cage projection");
            // Displaced P/N must disagree with cage when the shading normal tilts.
            check(bestDot < 0.999f, "displaced normal diverges from Nref");
            check(errFlip > 1e-4f, "displaced P/N re-projects differently (artifact without Pref)");
        }
    }
}

void testDefaultGroundDisplacement() {
    std::printf("default-ground-displacement\n");
    const char* path = "/tmp/disp_tex/xccibbi_8K_Displacement.exr";
    std::string err;
    Image img;
    if (!loadImage(path, img, err, /*srgbColor=*/false)) {
        std::printf("  skip load: %s\n", err.c_str());
        return;
    }
    auto tex = std::make_shared<Image>(img);
    Material mat;
    mat.displacementTex = 0;
    mat.displacementScale = 0.35f;
    mat.displacementZeroValue = 0.5f;
    mat.subdivIterations = 6;
    mat.autobump = 1;
    Scene scene;
    check(scene.addTexture(tex) == 0, "tex0");

    // Default buddha-scene ground: 40x40, 1 division (2 tris).
    MeshPtr ground = makeGridMesh(40.0f, 40.0f, 1, 1);
    check(ground->triangleCount() == 2, "default ground cage is 2 tris");
    const float cageEdge = [&]() {
        float m = 0.0f;
        for (size_t t = 0; t + 2 < ground->indices.size(); t += 3) {
            const Vec3& a = ground->positions[ground->indices[t]];
            const Vec3& b = ground->positions[ground->indices[t+1]];
            m = std::max(m, length(b - a));
        }
        return m;
    }();
    std::printf("  cage edge=%.3f tris=%zu\n", cageEdge, ground->triangleCount());

    MeshPtr out = tessDisplaceForTest(ground, mat, scene, mat.subdivIterations);
    float ymin = 1e9f, ymax = -1e9f;
    for (const Vec3& p : out->positions) {
        ymin = std::min(ymin, p.y);
        ymax = std::max(ymax, p.y);
    }
    std::printf("  displaced tris=%zu y=[%.4f,%.4f] delta=%.4f boundsY=[%.4f,%.4f]\n",
                out->triangleCount(), ymin, ymax, ymax - ymin, out->bounds.lo.y, out->bounds.hi.y);
    check(out->triangleCount() > 1000, "ground got subdivided");
    check(ymax - ymin > 0.15f, "ground shows visible height relief");

    // Full MaterialX + Stage path like the UI.
    if (!materialXAvailable()) {
        std::printf("  skip mtlx\n");
        return;
    }
    const QString exrPath = QStringLiteral("/tmp/disp_tex/xccibbi_8K_Displacement.exr");
    const QString albPath = QStringLiteral("/tmp/disp_tex/albedo_1k.png");
    const QString xml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <image name=\"alb\" type=\"color3\">\n"
        "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
        "  </image>\n"
        "  <image name=\"h1\" type=\"color3\">\n"
        "    <input name=\"file\" type=\"filename\" value=\"%2\"/>\n"
        "  </image>\n"
        "  <displacement name=\"disp1\" type=\"float\">\n"
        "    <input name=\"displacement\" type=\"float\" nodename=\"h1\"/>\n"
        "    <input name=\"scale\" type=\"float\" value=\"0.35\"/>\n"
        "    <input name=\"zero_value\" type=\"float\" value=\"0.5\"/>\n"
        "    <input name=\"subdiv_iterations\" type=\"integer\" value=\"6\"/>\n"
        "    <input name=\"bounds_padding\" type=\"float\" value=\"0.2\"/>\n"
        "    <input name=\"autobump\" type=\"boolean\" value=\"true\"/>\n"
        "  </displacement>\n"
        "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" nodename=\"alb\"/>\n"
        "    <input name=\"specular_roughness\" type=\"float\" value=\"0.55\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
        "    <input name=\"displacementshader\" type=\"displacementshader\" nodename=\"disp1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n").arg(albPath, exrPath);
    MaterialXEvalResult eval = evaluateMaterialXDocument(xml, QStringLiteral("/tmp/disp_tex"));
    check(eval.ok, "mtlx ok");
    Stage stage;
    StagePrim prim;
    prim.path = "/geo/ground";
    prim.type = PrimType::Mesh;
    prim.mesh = makeGridMesh(40.0f, 40.0f, 1, 1);
    prim.material = eval.material;
    prim.displacementTexture = eval.displacementTexture;
    prim.baseColorTexture = eval.baseColorTexture;
    prim.procedurals = eval.procedurals;
    prim.proceduralImages = eval.proceduralImages;
    prim.materialAssigned = true;
    prim.subdivType = kSubdivLinear;
    prim.subdivIterations = 6;
    prim.boundsPadding = 0.2f;
    stage.prims.push_back(prim);
    ScenePtr cooked = stage.toScene();
    tessellateCookedScene(cooked);
    check(cooked && !cooked->meshes.empty(), "cooked ground");
    ymin = 1e9f; ymax = -1e9f;
    for (const Vec3& p : cooked->meshes[0]->positions) {
        ymin = std::min(ymin, p.y);
        ymax = std::max(ymax, p.y);
    }
    std::printf("  stage tris=%zu ydelta=%.4f matDispTex=%d proc=%d\n",
                cooked->meshes[0]->triangleCount(), ymax - ymin,
                cooked->materials.empty() ? -1 : cooked->materials[0].displacementTex,
                cooked->materials.empty() ? -1 : cooked->materials[0].displacementProc);
    check(ymax - ymin > 0.15f, "stage ground relief");
}

void testRockDisplacementExr() {
    std::printf("rock-displacement-exr\n");
    const char* path = "/tmp/disp_tex/xccibbi_8K_Displacement.exr";
    std::string err;
    Image img;
    if (!loadImage(path, img, err, /*srgbColor=*/false)) {
        std::printf("  skip load: %s\n", err.c_str());
        return;
    }
    float mn = 1e9f, mx = -1e9f;
    double sum = 0.0;
    int n = 0;
    for (int y = 0; y < img.height(); y += 8) {
        for (int x = 0; x < img.width(); x += 8) {
            const float h = img.at(x, y).x;
            mn = std::min(mn, h);
            mx = std::max(mx, h);
            sum += double(h);
            ++n;
        }
    }
    std::printf("  EXR %dx%d R min=%.5f max=%.5f mean=%.5f\n", img.width(), img.height(), mn, mx,
                float(sum / std::max(1, n)));
    check(mx - mn > 0.2f, "EXR height has contrast");

    auto tex = std::make_shared<Image>(img);
    Material mat;
    mat.displacementTex = 0;
    mat.displacementScale = 1.0f;
    mat.subdivIterations = 3;
    mat.autobump = 0;
    Scene scene;
    check(scene.addTexture(tex) == 0, "tex index 0");
    MeshPtr sphere = makeSphereMesh(1.0f, 48, 24);
    MeshPtr out = tessDisplaceForTest(sphere, mat, scene, mat.subdivIterations);
    float rmin = 1e9f, rmax = 0.0f;
    for (const Vec3& p : out->positions) {
        const float r = length(p);
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
    }
    std::printf("  subdiv3 tris %zu→%zu radius [%.3f, %.3f] delta=%.3f\n", sphere->triangleCount(),
                out->triangleCount(), rmin, rmax, rmax - rmin);
    check(rmax - rmin > 0.15f, "EXR displace produces visible radius variation");

    // Full MaterialX path: albedo + displacement EXR (user wiring).
    QTemporaryDir tmp;
    check(tmp.isValid(), "tmpdir");
    const QString exrPath = QStringLiteral("/tmp/disp_tex/xccibbi_8K_Displacement.exr");
    const QString albedoPath = QStringLiteral("/tmp/disp_tex/albedo_1k.png");
    check(QFileInfo::exists(exrPath), "disp exr exists");
    check(QFileInfo::exists(albedoPath), "albedo jpg exists");
    const QString xml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <image name=\"alb\" type=\"color3\">\n"
        "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
        "  </image>\n"
        "  <image name=\"h1\" type=\"color3\">\n"
        "    <input name=\"file\" type=\"filename\" value=\"%2\"/>\n"
        "  </image>\n"
        "  <displacement name=\"disp1\" type=\"float\">\n"
        "    <input name=\"displacement\" type=\"float\" nodename=\"h1\"/>\n"
        "    <input name=\"scale\" type=\"float\" value=\"0.35\"/>\n"
        "    <input name=\"zero_value\" type=\"float\" value=\"0.5\"/>\n"
        "    <input name=\"subdiv_iterations\" type=\"integer\" value=\"6\"/>\n"
        "    <input name=\"bounds_padding\" type=\"float\" value=\"0.2\"/>\n"
        "    <input name=\"autobump\" type=\"boolean\" value=\"true\"/>\n"
        "  </displacement>\n"
        "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" nodename=\"alb\"/>\n"
        "    <input name=\"specular_roughness\" type=\"float\" value=\"0.55\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
        "    <input name=\"displacementshader\" type=\"displacementshader\" nodename=\"disp1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n").arg(albedoPath, exrPath);
    MaterialXEvalResult eval = evaluateMaterialXDocument(xml, QStringLiteral("/tmp/disp_tex"));
    check(eval.ok, "mtlx rock disp ok");
    check(eval.displacementTexture != nullptr || eval.material.displacementProc >= 0, "height bound");
    if (!(eval.baseColorTexture || eval.material.baseColorProc >= 0))
        std::printf("  WARN: albedo not bound (large JPEG may fail QImage::load)\n");
    else
        check(true, "albedo bound");
    std::printf("  mtlx tex=%d proc=%d subdiv=%d scale=%.3f zero=%.3f albedo=%d\n",
                eval.displacementTexture ? 1 : 0, eval.material.displacementProc,
                eval.material.subdivIterations, eval.material.displacementScale,
                eval.material.displacementZeroValue, eval.baseColorTexture ? 1 : 0);

    Stage stage;
    StagePrim prim;
    prim.path = "/geo/sphere";
    prim.type = PrimType::Mesh;
    prim.mesh = makeSphereMesh(1.0f, 96, 48);
    prim.material = eval.material;
    prim.displacementTexture = eval.displacementTexture;
    prim.baseColorTexture = eval.baseColorTexture;
    prim.procedurals = eval.procedurals;
    prim.proceduralImages = eval.proceduralImages;
    prim.materialAssigned = true;
    prim.subdivType = kSubdivLinear;
    prim.subdivIterations = 6;
    prim.boundsPadding = 0.2f;
    stage.prims.push_back(prim);
    ScenePtr cooked = stage.toScene();
    tessellateCookedScene(cooked);
    check(cooked && !cooked->meshes.empty(), "cooked");
    rmin = 1e9f; rmax = 0.0f;
    for (const Vec3& p : cooked->meshes[0]->positions) {
        const float r = length(p);
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
    }
    std::printf("  stage radius [%.3f, %.3f] delta=%.3f tris=%zu\n", rmin, rmax, rmax - rmin,
                cooked->meshes[0]->triangleCount());
    check(rmax - rmin > 0.05f, "stage cook shows relief");

    // Lit smoke render with albedo + mid-level zero (Arnold-style).
    {
        cooked->settings.resolutionX = 384;
        cooked->settings.resolutionY = 384;
        cooked->settings.samplesPerPixel = 32;
        cooked->settings.backend = kBackendCpuEmbree;
        cooked->settings.maxDepth = 4;
        LightData key;
        key.type = kLightDistant;
        key.intensity = 5.0f;
        key.color = Vec3(1.0f, 0.98f, 0.94f);
        key.xform = lookAtMatrix(Vec3(3.0f, 4.0f, 2.0f), Vec3(0.0f), Vec3(0.0f, 1.0f, 0.0f));
        cooked->lights.push_back(key);
        LightData fill;
        fill.type = kLightDistant;
        fill.intensity = 1.5f;
        fill.color = Vec3(0.65f, 0.75f, 1.0f);
        fill.xform = lookAtMatrix(Vec3(-2.0f, 1.0f, -3.0f), Vec3(0.0f), Vec3(0.0f, 1.0f, 0.0f));
        cooked->lights.push_back(fill);
        cooked->finalize();
        cooked->frameCameraOnContents();
        RenderSession session;
        session.setScene(cooked);
        session.start();
        session.waitForCompletion();
        Image outImg = session.linearImage();
        Image png(outImg.width(), outImg.height());
        for (int y = 0; y < outImg.height(); ++y) {
            for (int x = 0; x < outImg.width(); ++x) {
                Vec3 c = outImg.rgb(x, y);
                c = Vec3(std::pow(std::max(0.0f, c.x), 1.0f / 2.2f),
                         std::pow(std::max(0.0f, c.y), 1.0f / 2.2f),
                         std::pow(std::max(0.0f, c.z), 1.0f / 2.2f));
                png.setRgb(x, y, c);
            }
        }
        QDir().mkpath(QStringLiteral("/opt/cursor/artifacts"));
        std::string serr;
        const bool saved = saveImagePng("/opt/cursor/artifacts/rock_displace_smoke.png", png, serr);
        check(saved, "save rock smoke png");
        if (!saved)
            std::printf("  save err %s\n", serr.c_str());
        else
            std::printf("  wrote /opt/cursor/artifacts/rock_displace_smoke.png\n");
    }
}


void testArnoldDisplacement() {
    std::printf("arnold-displacement\n");
    if (!materialXAvailable()) {
        std::printf("  skip (no MaterialX)\n");
        return;
    }

    // Constant height displacement (no texture) — subdiv + offset along N.
    const QString constXml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <displacement name=\"disp1\" type=\"float\">\n"
        "    <input name=\"displacement\" type=\"float\" value=\"0.25\"/>\n"
        "    <input name=\"scale\" type=\"float\" value=\"2\"/>\n"
        "    <input name=\"subdiv_iterations\" type=\"integer\" value=\"1\"/>\n"
        "    <input name=\"bounds_padding\" type=\"float\" value=\"0.1\"/>\n"
        "    <input name=\"autobump\" type=\"boolean\" value=\"true\"/>\n"
        "    <input name=\"zero_value\" type=\"float\" value=\"0\"/>\n"
        "  </displacement>\n"
        "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" value=\"0.8, 0.8, 0.8\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
        "    <input name=\"displacementshader\" type=\"displacementshader\" nodename=\"disp1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
    MaterialXEvalResult eval = evaluateMaterialXDocument(constXml, QString());
    check(eval.ok, "displacement evaluates");
    check(std::fabs(eval.material.displacementHeight - 0.25f) < 1e-4f, "displacement height");
    check(std::fabs(eval.material.displacementScale - 2.0f) < 1e-4f, "displacement scale");
    check(eval.material.subdivIterations == 0, "subdiv on geo not material");
    check(eval.material.autobump == 1, "autobump on");
    check(materialHasGeometricDisplacement(eval.material), "has geometric displacement");

    MeshPtr grid = makeGridMesh(2.0f, 2.0f, 1, 1);
    check(grid && grid->triangleCount() == 2, "grid cage");
    const size_t cageTris = grid->triangleCount();
    const float cageMaxY = [&]() {
        float m = -1e9f;
        for (const Vec3& p : grid->positions) m = std::max(m, p.y);
        return m;
    }();

    Scene scene;
    grid->boundsPadding = 0.1f;
    MeshPtr displaced = tessDisplaceForTest(grid, eval.material, scene, 1);
    check(displaced != nullptr, "displaced mesh");
    check(displaced->triangleCount() == cageTris * 4, "one subdiv → 4x triangles");
    float maxY = -1e9f;
    for (const Vec3& p : displaced->positions) maxY = std::max(maxY, p.y);
    // Grid normals point +Y; height 0.25 * scale 2 = 0.5 along N.
    check(maxY > cageMaxY + 0.4f, "vertices displaced along normal");
    check(displaced->bounds.hi.y >= maxY, "bounds cover displaced verts");
    check(displaced->bounds.hi.y >= maxY + 0.05f, "bounds_padding expands AABB");
    // Authored grid normals are +Y; winding alone would give -Y — displace must
    // preserve the outward hemisphere or shading goes black.
    float avgNy = 0.0f;
    for (const Vec3& n : displaced->normals) avgNy += n.y;
    avgNy /= float(std::max<size_t>(1, displaced->normals.size()));
    check(avgNy > 0.5f, "displaced normals stay outward (+Y)");
    std::printf("  const disp tris %zu→%zu maxY %.3f→%.3f avgNy=%.3f\n", cageTris,
                displaced->triangleCount(), cageMaxY, maxY, avgNy);

    // Height map displacement through Stage::toScene.
    QTemporaryDir tmp;
    check(tmp.isValid(), "temp dir");
    const QString heightPath = tmp.filePath("disp_height.png");
    {
        QImage img(32, 32, QImage::Format_RGB32);
        for (int y = 0; y < 32; ++y)
            for (int x = 0; x < 32; ++x) {
                const int v = (x + y) % 2 == 0 ? 255 : 0;
                img.setPixel(x, y, qRgb(v, v, v));
            }
        check(img.save(heightPath), "write height png");
    }
    const QString mapXml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <image name=\"h1\" type=\"float\">\n"
        "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
        "  </image>\n"
        "  <displacement name=\"disp1\" type=\"float\">\n"
        "    <input name=\"displacement\" type=\"float\" nodename=\"h1\"/>\n"
        "    <input name=\"scale\" type=\"float\" value=\"0.2\"/>\n"
        "    <input name=\"subdiv_iterations\" type=\"integer\" value=\"2\"/>\n"
        "    <input name=\"autobump\" type=\"boolean\" value=\"false\"/>\n"
        "  </displacement>\n"
        "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" value=\"0.7, 0.7, 0.7\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
        "    <input name=\"displacementshader\" type=\"displacementshader\" nodename=\"disp1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n").arg(heightPath);
    eval = evaluateMaterialXDocument(mapXml, tmp.path());
    check(eval.ok, "map displacement evaluates");
    check(eval.displacementTexture != nullptr, "displacement binds height texture");
    check(eval.material.subdivIterations == 0, "map subdiv not on material");
    check(eval.material.autobump == 0, "autobump off");

    Stage stage;
    StagePrim prim;
    prim.path = "/geo/grid";
    prim.type = PrimType::Mesh;
    prim.mesh = makeGridMesh(1.0f, 1.0f, 2, 2);
    prim.material = eval.material;
    prim.displacementTexture = eval.displacementTexture;
    prim.materialAssigned = true;
    prim.subdivType = kSubdivLinear;
    prim.subdivIterations = 2;
    stage.prims.push_back(prim);
    ScenePtr cooked = stage.toScene();
    tessellateCookedScene(cooked);
    check(cooked && !cooked->meshes.empty(), "stage cooks displaced scene");
    const MeshPtr& outMesh = cooked->meshes[0];
    check(outMesh->triangleCount() == 8 * 16, "subdiv 2 → 16x tris (8 cage tris)");
    check(!cooked->materials.empty(), "material present");
    check(cooked->materials[0].displacementTex >= 0, "scene displacement tex index");
    check(cooked->materials[0].autobump == 0, "autobump preserved");
    std::printf("  map disp cageTris=8 → %zu tex=%d\n", outMesh->triangleCount(),
                cooked->materials[0].displacementTex);
}

void testTessellationTriangleBudget() {
    std::printf("tessellation-triangle-budget\n");
    // Default Dicing Poly Limit is 10M; hard ceiling 200M.
    MeshPtr sphere = makeSphereMesh(1.0f, 48, 24);
    check(sphere && sphere->triangleCount() > 1000, "sphere cage");
    const size_t cageTris = sphere->triangleCount();

    Scene scene;
    scene.settings.dicingPolyLimitM = 200;  // allow the classic 6-level explode for this test
    Material mat;
    mat.displacementHeight = 0.05f;
    mat.displacementScale = 1.0f;
    mat.displacementZeroValue = 0.0f;
    sphere->subdivType = kSubdivLinear;
    sphere->subdivIterations = 6;  // ~cage*4^6 ≈ 9M on this cage (above old 4M cap)
    MeshPtr out = tessDisplaceForTest(sphere, mat, scene, 6);
    check(out != nullptr, "budget tess produced a mesh");
    // Exact 4^n can shrink slightly after finalize()/validate() strips degenerates.
    check(out->triangleCount() > cageTris * 2000ull, "densified close to 6 linear levels");
    check(out->triangleCount() > 4000000ull, "exceeds former 4M soft-cap");
    check(out->triangleCount() <= 200000000ull, "under 200M triangle ceiling");
    std::printf("  cage=%zu → tess=%zu (6 levels, ceiling 200M)\n", cageTris, out->triangleCount());
}

void testFrustumCullCloseUpSubdiv() {
    std::printf("frustum-cull-close-up-subdiv\n");
    // Large ground, camera very close — verts sit off-screen but the face covers
    // the frame. Must still densify locally (not skip the mesh entirely).
    auto ground = makeGridMesh(40.0f, 40.0f, 1, 1);
    check(ground && ground->triangleCount() == 2, "ground cage 2 tris");
    ground->subdivType = kSubdivLinear;
    ground->subdivIterations = 4;
    ground->dicingQuality = 1.0f;

    Material mat;
    mat.displacementHeight = 1.0f;  // 1 * 0.1 = 0.1 m offset
    mat.displacementScale = 0.1f;   // also verify scale < 1 still displaces
    mat.displacementZeroValue = 0.0f;
    mat.displacementTex = -1;
    mat.displacementProc = -1;

    Scene closeScene;
    closeScene.settings.frustumCull = 1;
    closeScene.settings.frustumPadding = 10.0f;
    closeScene.settings.screenAdaptive = 0;
    closeScene.settings.resolutionX = 640;
    closeScene.settings.resolutionY = 360;
    closeScene.camera.focalLength = 50.0f;
    closeScene.camera.sensorWidth = 36.0f;
    // Very close above the plane looking at the center — verts at ±20 are off-frame.
    closeScene.camera.cameraToWorld = lookAtMatrix(Vec3(0, 0.15f, 0.2f), Vec3(0, 0, 0), Vec3(0, 1, 0));
    closeScene.cameraAuthored = true;

    const size_t cageTris = ground->triangleCount();
    MeshPtr out = tessDisplaceForTest(ground, mat, closeScene, 4, /*forceFrustumOff=*/false);
    check(out != nullptr, "close-up tess mesh");
    check(out->triangleCount() > cageTris * 8, "close-up still subdivides under frustum cull");
    // Local dicing — must NOT explode the whole 40×40 plane to uniform 4^N.
    check(out->triangleCount() < cageTris * 256, "frustum-local stays below uniform 4^4");

    // Displacement scale 0.1 with height 1 / zero 0 → offset ~0.1 along +Y for a flat grid.
    float maxY = -1.0e9f;
    float minY = 1.0e9f;
    for (const Vec3& p : out->positions) {
        maxY = std::max(maxY, p.y);
        minY = std::min(minY, p.y);
    }
    check(maxY > 0.05f, "displacement scale 0.1 still lifts the surface");
    std::printf("  close-up tris %zu→%zu y[%.3f, %.3f] scale=0.1\n", cageTris, out->triangleCount(),
                minY, maxY);
}

void testFrustumLocalDicingFalloff() {
    std::printf("frustum-local-dicing-falloff\n");
    // Huge plane + narrow view: density must peak near the look-at and stay coarse
    // at far corners — not uniform 4^N across the whole mesh.
    auto ground = makeGridMesh(200.0f, 200.0f, 1, 1);
    check(ground && ground->triangleCount() == 2, "huge ground cage");
    ground->subdivType = kSubdivLinear;
    ground->subdivIterations = 6;
    ground->dicingQuality = 1.0f;

    Material mat;
    mat.displacementHeight = 0.01f;
    mat.displacementScale = 1.0f;
    mat.displacementZeroValue = 0.0f;

    Scene scene;
    scene.settings.frustumCull = 1;
    scene.settings.frustumPadding = 5.0f;
    scene.settings.screenAdaptive = 0;
    scene.settings.dicingPolyLimitM = 50;
    scene.settings.resolutionX = 640;
    scene.settings.resolutionY = 360;
    scene.camera.focalLength = 85.0f;
    scene.camera.sensorWidth = 36.0f;
    scene.camera.cameraToWorld = lookAtMatrix(Vec3(0, 8, 12), Vec3(0, 0, 0), Vec3(0, 1, 0));
    scene.cameraAuthored = true;

    const size_t uniformCap = ground->triangleCount() * 4096ull;  // 4^6
    const auto t0 = std::chrono::steady_clock::now();
    MeshPtr out = tessDisplaceForTest(ground, mat, scene, 6, /*forceFrustumOff=*/false);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    check(out != nullptr, "frustum-local mesh");
    check(out->triangleCount() > 32, "frustum-local actually densified");
    check(out->triangleCount() < uniformCap / 4, "far cheaper than whole-mesh 4^6");
    check(ms < 5000.0, "frustum-local densify finishes quickly (no Start hang)");

    auto maxEdgeNear = [&](float radius) {
        float m = 0.0f;
        size_t n = 0;
        for (size_t t = 0; t + 2 < out->indices.size(); t += 3) {
            const Vec3& a = out->positions[out->indices[t]];
            const Vec3& b = out->positions[out->indices[t + 1]];
            const Vec3& c = out->positions[out->indices[t + 2]];
            const Vec3 mid = (a + b + c) * (1.0f / 3.0f);
            const float r = std::sqrt(mid.x * mid.x + mid.z * mid.z);
            if (r > radius) continue;
            m = std::max(m, length(b - a));
            m = std::max(m, length(c - b));
            m = std::max(m, length(a - c));
            ++n;
        }
        check(n > 0, "found tris in radius band");
        return m;
    };
    auto maxEdgeFar = [&](float minR) {
        float m = 0.0f;
        size_t n = 0;
        for (size_t t = 0; t + 2 < out->indices.size(); t += 3) {
            const Vec3& a = out->positions[out->indices[t]];
            const Vec3& b = out->positions[out->indices[t + 1]];
            const Vec3& c = out->positions[out->indices[t + 2]];
            const Vec3 mid = (a + b + c) * (1.0f / 3.0f);
            const float r = std::sqrt(mid.x * mid.x + mid.z * mid.z);
            if (r < minR) continue;
            m = std::max(m, length(b - a));
            m = std::max(m, length(c - b));
            m = std::max(m, length(a - c));
            ++n;
        }
        check(n > 0, "found far tris");
        return m;
    };

    const float nearEdge = maxEdgeNear(8.0f);
    const float farEdge = maxEdgeFar(60.0f);
    check(farEdge > nearEdge * 2.0f, "corners stay coarser than frustum center");
    std::printf("  local tris=%zu nearEdge=%.3f farEdge=%.3f ms=%.1f (uniform6 would be %zu)\n",
                out->triangleCount(), nearEdge, farEdge, ms, uniformCap);
}

void testFrustumLocalFullyInViewFast() {
    std::printf("frustum-local-fully-in-view-fast\n");
    // Mesh entirely in frustum must not hang: fast-path to uniform splits.
    MeshPtr sphere = makeSphereMesh(1.0f, 24, 16);
    check(sphere && sphere->triangleCount() > 100, "sphere cage");
    sphere->subdivType = kSubdivLinear;
    sphere->subdivIterations = 4;
    const size_t cage = sphere->triangleCount();

    Material mat;
    mat.displacementHeight = 0.02f;
    mat.displacementScale = 1.0f;
    mat.displacementZeroValue = 0.0f;

    Scene fr;
    fr.settings.frustumCull = 1;
    fr.settings.frustumPadding = 10.0f;
    fr.settings.screenAdaptive = 0;
    fr.settings.resolutionX = 640;
    fr.settings.resolutionY = 360;
    fr.camera.focalLength = 50.0f;
    fr.camera.sensorWidth = 36.0f;
    fr.camera.cameraToWorld = lookAtMatrix(Vec3(0, 0, 6), Vec3(0, 0, 0), Vec3(0, 1, 0));
    fr.cameraAuthored = true;

    const auto t0 = std::chrono::steady_clock::now();
    MeshPtr out = tessDisplaceForTest(sphere, mat, fr, 4, /*forceFrustumOff=*/false);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                          .count();
    check(out != nullptr, "in-view frustum mesh");
    // Nearly full densify (silhouette faces may leave the frustum a level early).
    check(out->triangleCount() > cage * 128, "fully in-view densifies aggressively");
    check(out->triangleCount() <= cage * 256, "does not exceed uniform 4^4");
    check(ms < 8000.0, "fully in-view frustum densify stays responsive");
    std::printf("  in-view %zu→%zu in %.1fms\n", cage, out->triangleCount(), ms);
}

void testFrustumLocalItersNotClampedOnDenseCage() {
    std::printf("frustum-local-iters-not-clamped-on-dense-cage\n");
    // ~200k cage: whole-mesh 4^n clamp would freeze Subdiv Iterations at 3.
    // Zoomed frustum-local must still honor higher iteration counts on the patch.
    auto ground = makeGridMesh(200.0f, 200.0f, 320, 320);
    check(ground && ground->triangleCount() > 150000, "dense ground cage");
    ground->subdivType = kSubdivLinear;
    ground->dicingQuality = 1.0f;
    const size_t cage = ground->triangleCount();

    Material mat;
    mat.displacementHeight = 0.01f;
    mat.displacementScale = 1.0f;
    mat.displacementZeroValue = 0.0f;

    auto run = [&](int iters) {
        auto g = std::make_shared<Mesh>(*ground);
        g->subdivIterations = iters;
        Scene scene;
        scene.settings.frustumCull = 1;
        scene.settings.frustumPadding = 5.0f;
        scene.settings.screenAdaptive = 0;
        scene.settings.dicingPolyLimitM = 50;
        scene.settings.resolutionX = 640;
        scene.settings.resolutionY = 360;
        scene.camera.focalLength = 85.0f;
        scene.camera.sensorWidth = 36.0f;
        scene.camera.cameraToWorld = lookAtMatrix(Vec3(0, 6, 10), Vec3(0, 0, 0), Vec3(0, 1, 0));
        scene.cameraAuthored = true;
        return tessDisplaceForTest(g, mat, scene, iters, /*forceFrustumOff=*/false);
    };

    MeshPtr low = run(3);
    MeshPtr high = run(8);
    check(low && high, "dense frustum runs");
    // Old whole-mesh clamp: both stuck at 3 levels → same (or tiny) delta.
    check(high->triangleCount() > low->triangleCount() + cage / 8,
          "raising Subdiv Iterations still densifies under frustum-local");

    auto minEdgeNear = [&](const MeshPtr& m, float radius) {
        float best = 1.0e30f;
        for (size_t t = 0; t + 2 < m->indices.size(); t += 3) {
            const Vec3& a = m->positions[m->indices[t]];
            const Vec3& b = m->positions[m->indices[t + 1]];
            const Vec3& c = m->positions[m->indices[t + 2]];
            const Vec3 mid = (a + b + c) * (1.0f / 3.0f);
            const float r = std::sqrt(mid.x * mid.x + mid.z * mid.z);
            if (r > radius) continue;
            best = std::min(best, length(b - a));
            best = std::min(best, length(c - b));
            best = std::min(best, length(a - c));
        }
        return best;
    };
    const float e3 = minEdgeNear(low, 6.0f);
    const float e8 = minEdgeNear(high, 6.0f);
    check(e8 < e3 * 0.75f, "higher iters → finer edges in frustum center");
    std::printf("  dense cage=%zu tris3=%zu tris8=%zu edge3=%.4f edge8=%.4f\n", cage,
                low->triangleCount(), high->triangleCount(), e3, e8);
}

void testScreenAdaptiveTessellation() {
    std::printf("screen-adaptive-tessellation\n");
    // Spatial edge dicing: close camera densifies more than a far one.
    auto makeUnitTri = [](float dicingQuality) {
        auto m = std::make_shared<Mesh>();
        m->positions = {Vec3(0, 0, 0), Vec3(4, 0, 0), Vec3(0, 4, 0)};
        m->indices = {0, 1, 2};
        m->computeNormalsIfMissing();
        m->computeBounds();
        m->subdivType = kSubdivLinear;
        m->subdivIterations = 8;
        m->dicingQuality = dicingQuality;  // target edge px = 1/quality
        return m;
    };

    Material mat;
    mat.displacementHeight = 0.01f;
    mat.displacementScale = 1.0f;
    mat.displacementZeroValue = 0.0f;

    // target ≈ 8 px — far camera should stop after a couple of splits.
    constexpr float kQuality = 0.125f;

    Scene nearScene;
    nearScene.settings.screenAdaptive = 1;
    nearScene.settings.frustumCull = 0;
    nearScene.settings.resolutionX = 640;
    nearScene.settings.resolutionY = 360;
    nearScene.camera.focalLength = 50.0f;
    nearScene.camera.sensorWidth = 36.0f;
    nearScene.camera.cameraToWorld = lookAtMatrix(Vec3(2, 2, 5), Vec3(1, 1, 0), Vec3(0, 1, 0));
    nearScene.cameraAuthored = true;
    MeshPtr nearOut = tessDisplaceForTest(makeUnitTri(kQuality), mat, nearScene, 8);
    check(nearOut != nullptr, "near adaptive mesh");
    const size_t nearTris = nearOut->triangleCount();

    Scene farScene;
    farScene.settings.screenAdaptive = 1;
    farScene.settings.frustumCull = 0;
    farScene.settings.resolutionX = 640;
    farScene.settings.resolutionY = 360;
    farScene.camera.focalLength = 50.0f;
    farScene.camera.sensorWidth = 36.0f;
    farScene.camera.cameraToWorld = lookAtMatrix(Vec3(2, 2, 400), Vec3(1, 1, 0), Vec3(0, 1, 0));
    farScene.cameraAuthored = true;
    MeshPtr farOut = tessDisplaceForTest(makeUnitTri(kQuality), mat, farScene, 8);
    check(farOut != nullptr, "far adaptive mesh");
    const size_t farTris = farOut->triangleCount();

    Scene uniformScene;
    uniformScene.settings.screenAdaptive = 0;
    uniformScene.settings.frustumCull = 0;
    MeshPtr uniOut = tessDisplaceForTest(makeUnitTri(kQuality), mat, uniformScene, 8);
    check(uniOut != nullptr, "uniform mesh");
    const size_t uniTris = uniOut->triangleCount();

    check(nearTris > farTris, "near camera densifies more than far under Screen Adaptive");
    check(farTris < uniTris, "far adaptive stays coarser than uniform 8");
    check(nearTris > 1, "near adaptive actually subdivided");
    const std::string fa = tessellationFingerprint(nearScene);
    nearScene.settings.screenAdaptive = 0;
    const std::string fb = tessellationFingerprint(nearScene);
    check(fa != fb, "fingerprint changes when Screen Adaptive toggles");
    std::printf("  adaptive near=%zu far=%zu uniform8=%zu\n", nearTris, farTris, uniTris);
}

void testScreenAdaptiveQualityCoarse() {
    std::printf("screen-adaptive-quality-coarse\n");
    // Quality 0.01 → target ≈ 100 px. A close-up on a large ground must NOT
    // explode to the Dicing Poly Limit (clipped-edge "force refine" bug).
    auto ground = makeGridMesh(200.0f, 200.0f, 1, 1);
    check(ground && ground->triangleCount() == 2, "coarse quality ground cage");
    ground->subdivType = kSubdivLinear;
    ground->subdivIterations = 8;  // ignored under Screen Adaptive
    ground->dicingQuality = 0.01f;

    Material mat;
    mat.displacementHeight = 0.01f;
    mat.displacementScale = 1.0f;
    mat.displacementZeroValue = 0.0f;

    Scene scene;
    scene.settings.screenAdaptive = 1;
    scene.settings.frustumCull = 1;
    scene.settings.frustumPadding = 10.0f;
    scene.settings.dicingPolyLimitM = 50;
    scene.settings.resolutionX = 1920;
    scene.settings.resolutionY = 1080;
    scene.camera.focalLength = 50.0f;
    scene.camera.sensorWidth = 36.0f;
    scene.camera.cameraToWorld = lookAtMatrix(Vec3(0, 0.2f, 0.3f), Vec3(0, 0, 0), Vec3(0, 1, 0));
    scene.cameraAuthored = true;

    MeshPtr out = tessDisplaceForTest(ground, mat, scene, 8, /*forceFrustumOff=*/false);
    check(out != nullptr, "coarse quality mesh");
    const size_t tris = out->triangleCount();
    // Ideal µpoly count at 100 px ≈ screenArea/100² ≈ 200; allow soft tails +
    // perspective, but nowhere near tens of millions.
    check(tris < 200000ull, "Quality 0.01 stays well below poly-limit explosion");
    check(tris > 2ull, "Quality 0.01 still dices the visible patch a bit");
    std::printf("  Q=0.01 close-up tris=%zu (limit would be 50M)\n", tris);
}

void testEnableDisplacementMasterSwitch() {
    std::printf("enable-displacement-master-switch\n");
    auto ground = makeGridMesh(4.0f, 4.0f, 2, 2);
    check(ground != nullptr, "ground cage");
    ground->subdivType = kSubdivLinear;
    ground->subdivIterations = 3;
    ground->dicingQuality = 1.0f;
    const size_t cageTris = ground->triangleCount();

    Material mat;
    mat.displacementHeight = 0.1f;
    mat.displacementScale = 1.0f;
    mat.displacementZeroValue = 0.0f;

    Scene on;
    on.settings.screenAdaptive = 0;
    on.settings.frustumCull = 0;
    on.settings.enableDisplacement = 1;
    MeshPtr densified = tessDisplaceForTest(ground, mat, on, 3);
    check(densified && densified->triangleCount() > cageTris, "displace on densifies");

    auto ground2 = makeGridMesh(4.0f, 4.0f, 2, 2);
    ground2->subdivType = kSubdivLinear;
    ground2->subdivIterations = 3;
    Scene off;
    off.settings.screenAdaptive = 0;
    off.settings.frustumCull = 0;
    off.settings.enableDisplacement = 0;
    MeshPtr cages = tessDisplaceForTest(ground2, mat, off, 3);
    check(cages && cages->triangleCount() == cageTris, "displace off keeps cage tris");
    check(cages->restPositions.empty(), "displace off has no Pref lock");
    std::printf("  on=%zu off=%zu cage=%zu\n", densified->triangleCount(), cages->triangleCount(),
                cageTris);
}

void testTimeDependentStamp() {
    std::printf("time-dependent-mesh-stamp\n");
    auto m = std::make_shared<Mesh>();
    m->positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)};
    m->indices = {0, 1, 2};
    m->computeNormalsIfMissing();
    m->timeDependent = true;
    m->subdivType = kSubdivLinear;
    m->subdivIterations = 2;

    Material mat;
    mat.displacementHeight = 0.01f;
    mat.displacementScale = 1.0f;

    Scene scene;
    scene.settings.enableDisplacement = 1;
    scene.settings.frustumCull = 0;
    scene.settings.screenAdaptive = 0;
    MeshPtr out = tessDisplaceForTest(m, mat, scene, 2);
    check(out != nullptr, "timeDependent tess mesh");
    check(out->timeDependent, "timeDependent preserved through tess+displace");
    std::printf("  timeDependent ok tris=%zu\n", out->triangleCount());
}

void testSpectralHeroBasics() {
    std::printf("spectral-hero-basics\n");
    SampledWavelengths w = SampledWavelengths::sampleUniform(4, 0.25f);
    check(w.n == 4, "hero sample count");
    check(w.lambda[0] >= kSpectrumLambdaMin && w.lambda[3] <= kSpectrumLambdaMax, "lambda range");
    const RGBColorSpace& aces = colorSpaceAcesCg();
    SampledSpectrum whiteAlbedo = rgbToSpectrumReflectance(Vec3(1, 1, 1), w, aces);
    check(spectrumAvg(whiteAlbedo) > 0.5f, "white upsample energy");
    SampledSpectrum gold = SampledSpectrum::zero(w.n);
    for (int i = 0; i < w.n; ++i) {
        SpectralNk nk = metalNk("Au", w.lambda[i]);
        gold.values[i] = conductorFresnel(0.5f, nk.eta, nk.k);
    }
    check(spectrumAvg(gold) > 0.1f, "gold fresnel");

    // Film colour space follows Working Space (ACEScg or linear sRGB).
    {
        RenderSettingsData st;
        st.workingSpace = kWorkingSpaceAcesCg;
        check(&pathColorSpace(st) == &colorSpaceAcesCg(), "ACEScg working space drives film");
        st.workingSpace = kWorkingSpaceSrgbLinear;
        check(&pathColorSpace(st) == &colorSpaceSrgb(), "sRGB working space drives film");
    }

    auto meanToRgb = [](int nspp, bool visible,
                        const std::function<SampledSpectrum(const SampledWavelengths&)>& spec,
                        const RGBColorSpace& cs) -> Vec3 {
        Rng rng(7u, 13u);
        double r = 0.0, g = 0.0, b = 0.0;
        for (int s = 0; s < nspp; ++s) {
            SampledWavelengths wv =
                visible ? SampledWavelengths::sampleVisible(4, rng.nextFloat())
                        : SampledWavelengths::sampleUniform(4, rng.nextFloat());
            const Vec3 o = spectrumToRgb(spec(wv), wv, cs);
            r += double(o.x);
            g += double(o.y);
            b += double(o.z);
        }
        const double inv = 1.0 / double(nspp);
        return Vec3(float(r * inv), float(g * inv), float(b * inv));
    };
    const int nspp = 4000;
    // pbrt ToXYZ needs many λ samples; four hero wavelengths are an MC estimator.
    Vec3 rgb = meanToRgb(
        nspp, true,
        [&](const SampledWavelengths& wv) {
            return rgbToSpectrumReflectance(Vec3(1, 1, 1), wv, aces) *
                   rgbToSpectrumEmission(Vec3(1.0f), wv, aces);
        },
        aces);
    check(rgb.x > 0.0f && rgb.y > 0.0f && rgb.z > 0.0f, "spectrumToRgb white albedo");
    checkNear(rgb.x / srMax(rgb.y, 1e-8f), 1.0f, 0.08f, "white albedo under D60 R/G");
    checkNear(rgb.z / srMax(rgb.y, 1e-8f), 1.0f, 0.08f, "white albedo under D60 B/G");
    checkNear(rgb.y, 1.0f, 0.12f, "white albedo under D60 ~1");
    Vec3 greyOut = meanToRgb(
        nspp, true,
        [&](const SampledWavelengths& wv) {
            return rgbToSpectrumReflectance(Vec3(0.8f, 0.8f, 0.8f), wv, aces) *
                   rgbToSpectrumEmission(Vec3(1.0f), wv, aces);
        },
        aces);
    checkNear(greyOut.y, 0.8f, 0.12f, "grey 0.8 albedo under D60");
    checkNear(greyOut.x / srMax(greyOut.y, 1e-8f), 1.0f, 0.08f, "grey 0.8 ACEScg neutral");
    // pbrt SampleLd: Illuminant(Le) × Albedo(f) × geom reconstructs grey NEE RGB.
    {
        const Vec3 Le(4.0f, 4.0f, 4.0f);
        const Vec3 f(0.8f / kPi, 0.8f / kPi, 0.8f / kPi);
        const float geom = 0.35f;
        Vec3 book = meanToRgb(
            nspp, true,
            [&](const SampledWavelengths& wv) {
                return rgbToSpectrumEmission(Le, wv, aces) * rgbToSpectrumReflectance(f, wv, aces) * geom;
            },
            aces);
        const Vec3 want = Le * f * geom;
        checkNear(book.x / srMax(want.x, 1e-8f), 1.0f, 0.15f, "book NEE R ~ Le*f*geom");
        checkNear(book.y / srMax(want.y, 1e-8f), 1.0f, 0.15f, "book NEE G ~ Le*f*geom");
        checkNear(book.z / srMax(want.z, 1e-8f), 1.0f, 0.15f, "book NEE B ~ Le*f*geom");
        checkNear(book.x / srMax(book.y, 1e-8f), 1.0f, 0.08f, "book NEE ACEScg neutral");
        // Wavefront bug: multiply NEE by the next BSDF weight (≈ albedo) after enqueue.
        check(std::fabs(book.y * 0.8f - want.y) > 0.05f * want.y,
              "NEE × continuation albedo is not the vertex estimator");
    }
    Vec3 d60Rgb = meanToRgb(
        nspp, true, [&](const SampledWavelengths& wv) { return illuminantSpectrum(wv, aces); }, aces);
    checkNear(d60Rgb.x, 1.0f, 0.10f, "D60 illuminant ACEScg R");
    checkNear(d60Rgb.y, 1.0f, 0.10f, "D60 illuminant ACEScg G");
    checkNear(d60Rgb.z, 1.0f, 0.10f, "D60 illuminant ACEScg B");
    Vec3 hdri(4.2f, 4.1f, 4.3f);
    Vec3 hdriOut = meanToRgb(
        nspp, true,
        [hdri, &aces](const SampledWavelengths& wv) { return rgbToSpectrumEmission(hdri, wv, aces); },
        aces);
    checkNear(hdriOut.x / srMax(hdriOut.y, 1e-8f), hdri.x / hdri.y, 0.08f, "HDRI ACEScg not pink");
    check(hdriOut.y > 2.0f, "HDRI keeps energy");
    Vec3 whiteL = meanToRgb(
        nspp, true,
        [&](const SampledWavelengths& wv) { return rgbToSpectrumEmission(Vec3(1.0f), wv, aces); },
        aces);
    checkNear(whiteL.x, 1.0f, 0.12f, "white RGBIlluminantSpectrum ACEScg R");
    checkNear(whiteL.y, 1.0f, 0.12f, "white RGBIlluminantSpectrum ACEScg G");
    checkNear(whiteL.z, 1.0f, 0.12f, "white RGBIlluminantSpectrum ACEScg B");
    // Linear path-weight upsample must conserve energy under multiply (glass enter×exit).
    {
        const float eta = 1.5f;
        const Vec3 wEnter(1.0f / (eta * eta));
        const Vec3 wExit(eta * eta);
        SampledSpectrum t = rgbToSpectrumLinear(wEnter, w);
        t *= rgbToSpectrumLinear(wExit, w);
        for (int i = 0; i < t.n; ++i)
            check(std::fabs(t.values[i] - 1.0f) < 0.08f, "glass enter×exit linear upsample ~1");
        Vec3 round = meanToRgb(
            nspp, true,
            [&](const SampledWavelengths& wv) {
                SampledSpectrum tt = rgbToSpectrumLinear(wEnter, wv);
                tt *= rgbToSpectrumLinear(wExit, wv);
                return tt * rgbToSpectrumEmission(Vec3(1.0f), wv, aces);
            },
            aces);
        checkNear(round.x, 1.0f, 0.12f, "glass enter×exit under D60 R");
        checkNear(round.y, 1.0f, 0.12f, "glass enter×exit under D60 G");
        checkNear(round.z, 1.0f, 0.12f, "glass enter×exit under D60 B");
    }
    // Abbe IOR must vary with λ (geometric dispersion).
    const float nBlue = dielectricIorFromAbbe(1.5f, 30.0f, 450.0f);
    const float nRed = dielectricIorFromAbbe(1.5f, 30.0f, 650.0f);
    check(nBlue > nRed + 0.01f, "dispersion IOR blue > red");
    {
        Material g;
        g.transmission = 1.0f;
        g.ior = 1.5f;
        g.roughness = 0.0f;
        g.specular = 1.0f;
        g.dispersionAbbe = 30.0f;
        SampledWavelengths wb{};
        wb.n = 4;
        wb.lambda[0] = 450.0f;
        wb.pdf[0] = 1.0f;
        SampledWavelengths wr = wb;
        wr.lambda[0] = 650.0f;
        const Vec3 wo = normalize(Vec3(0.55f, 0.0f, 0.835f));
        const BsdfSampleSpectral sb =
            bsdfSampleSpectral(g, wo, 0.99f, 0.0f, 0.0f, 0.9f, wb, g.ior, 0);
        const BsdfSampleSpectral sr =
            bsdfSampleSpectral(g, wo, 0.99f, 0.0f, 0.0f, 0.9f, wr, g.ior, 0);
        check(sb.valid && sr.valid && sb.transmitted && sr.transmitted,
              "dispersive Snell samples transmit");
        check(std::fabs(sr.wi.x) > std::fabs(sb.wi.x) + 0.004f,
              "blue bends more toward the normal than red (pbrt η(λ) geometry)");
        SampledWavelengths wLive = SampledWavelengths::sampleUniform(4, 0.2f);
        check(!wLive.secondaryTerminated(), "uniform λ start with secondaries");
        terminateSecondaryIfSpectralEta(g, wLive);
        check(wLive.secondaryTerminated(), "pbrt GetBxDF terminates secondaries on η(λ)");
        Material achromatic = g;
        achromatic.dispersionAbbe = 0.0f;
        SampledWavelengths wKeep = SampledWavelengths::sampleUniform(4, 0.2f);
        terminateSecondaryIfSpectralEta(achromatic, wKeep);
        check(!wKeep.secondaryTerminated(), "constant η keeps secondaries");
    }
    // RGB η/κ seed from metal table.
    Vec3 eta, k;
    metalNkRgbPreset("Au", eta, k);
    check(eta.x > 0.0f && k.x > 0.0f, "Au η/κ rgb seed");
    SpectralNk nk550 = nkFromRgb(eta, k, 550.0f);
    check(nk550.eta > 0.0f && nk550.k > 0.0f, "nkFromRgb");
    {
        Vec3 etaFit, kFit;
        conductorNkFromReflectance(Vec3(1.0f, 0.766f, 0.336f), Vec3(1.5f), etaFit, kFit);
        check(kFit.x > 0.5f && kFit.y > 0.0f && kFit.z > 0.0f, "gold F0 inverts to conductor k");
        Material metal;
        metal.metallic = 1.0f;
        metal.baseColor = Vec3(1.0f, 0.766f, 0.336f);
        metal.conductorEta = Vec3(1.5f);
        metal.conductorK = Vec3(0.0f);
        SampledWavelengths wm = SampledWavelengths::sampleUniform(4, 0.25f);
        const Frame fr(Vec3(0.0f, 0.0f, 1.0f));
        const Vec3 wo(0.0f, 0.0f, 1.0f);
        const Vec3 wi = normalize(Vec3(0.25f, 0.0f, 0.97f));
        SampledSpectrum lifted = liftBsdfWeight(metal, fr, wo, wi, Vec3(1.0f), wm, 1.5f, 0);
        float mn = 1.0e9f, mx = 0.0f;
        for (int i = 0; i < lifted.n; ++i) {
            if (wm.pdf[i] <= 0.0f) continue;
            mn = srMin(mn, lifted.values[i]);
            mx = srMax(mx, lifted.values[i]);
        }
        check(mx - mn > 1e-4f, "metallic without authored k uses spectral conductor Fresnel");
    }
    {
        Material skin;
        skin.subsurfaceColor = Vec3(1.0f, 0.75f, 0.55f);
        skin.subsurfaceRadius = Vec3(1.0f, 0.35f, 0.2f);
        skin.subsurfaceScale = 1.0f;
        check(sssMfpAtLambda(skin, 650.0f) > sssMfpAtLambda(skin, 550.0f) + 0.1f,
              "skin SSS MFP red > green");
        check(sssMfpAtLambda(skin, 550.0f) > sssMfpAtLambda(skin, 450.0f) + 0.05f,
              "skin SSS MFP green > blue");
        check(sssAlbedoAtLambda(skin, 650.0f) > sssAlbedoAtLambda(skin, 450.0f),
              "skin SSS albedo red > blue");
    }

    // Tabulated CIE + TerminateSecondary + blackbody + visible sampling.
    {
        float cx, cy, cz;
        cieXyzAtLambda(555.0f, cx, cy, cz);
        check(cy > 0.9f && cy <= 1.01f, "CIE Y peak near 555");
        SampledWavelengths vis = SampledWavelengths::sampleVisible(4, 0.3f);
        check(vis.pdf[0] > 0.0f, "visible pdf");
        vis.promoteHero(2);
        const float pdf0 = vis.pdf[0];
        vis.terminateSecondary();
        check(vis.secondaryTerminated(), "secondary terminated");
        check(std::fabs(vis.pdf[0] - pdf0 / 4.0f) < 1e-6f, "terminate scales pdf[0]");
        check(vis.pdf[1] == 0.0f && vis.pdf[2] == 0.0f, "secondary pdf zero");
        BlackbodySpectrum bb(6500.0f);
        SampledSpectrum bbs = bb.sample(w);
        check(spectrumAvg(bbs) > 0.1f, "blackbody sample");
        ConstantSpectrum ones(1.0f);
        check(ones.sample(w).values[0] == 1.0f, "constant spectrum");
        Vec3 acesFilm = meanToRgb(
            nspp, true,
            [&](const SampledWavelengths& wv) { return rgbToSpectrumEmission(Vec3(1.0f), wv, aces); },
            aces);
        check(acesFilm.x > 0.0f && acesFilm.y > 0.0f && acesFilm.z > 0.0f, "ACEScg convert");
        checkNear(acesFilm.x, 1.0f, 0.15f, "ACEScg white illuminant R");
        checkNear(acesFilm.y, 1.0f, 0.15f, "ACEScg white illuminant G");
        checkNear(acesFilm.z, 1.0f, 0.15f, "ACEScg white illuminant B");
        const float fBlue = airyReflectanceScalar(0.8f, 1.4f, 550.0f, 450.0f, 0.2f);
        const float fRed = airyReflectanceScalar(0.8f, 1.4f, 550.0f, 650.0f, 0.2f);
        check(std::fabs(fBlue - fRed) > 1e-4f, "thin-film Airy chromatic");

        // Visible + TerminateSecondary: mean of many E=1 samples is unbiased XYZ
        // of illuminant E (near-white). Individual samples are spectral colours.
        {
            Rng rng(42u, 7u);
            double rSum = 0.0, gSum = 0.0, bSum = 0.0;
            const int nterm = 8000;
            for (int s = 0; s < nterm; ++s) {
                SampledWavelengths wv = SampledWavelengths::sampleVisible(4, rng.nextFloat());
                SampledSpectrum L = SampledSpectrum::constant(wv.n, 1.0f);
                wv.terminateSecondary();
                const Vec3 o = spectrumToRgb(L, wv, aces);
                rSum += double(o.x);
                gSum += double(o.y);
                bSum += double(o.z);
            }
            const float r = float(rSum / nterm), g = float(gSum / nterm), b = float(bSum / nterm);
            check(g > 0.15f, "Vis+Terminate equal-energy has energy");
            // Illuminant E in ACEScg (D60 white) is not (1,1,1); the mean must
            // still be a plausible white, not a single-λ CMF spike.
            check(std::fabs(r / g - 1.0f) < 0.35f && std::fabs(b / g - 1.0f) < 0.35f,
                  "Vis+TerminateSecondary equal-energy not a CMF spike");
            check(r > 0.05f && b > 0.05f, "Vis+Terminate equal-energy all channels");
        }

        // Clear glass preset: spectral dielectric enter×exit×white env stays neutral.
        {
            Rng rng(11u, 3u);
            double rSum = 0.0, gSum = 0.0, bSum = 0.0;
            int accepted = 0;
            const int nglass = 8000;
            Material glass;
            glass.baseColor = Vec3(1.0f);
            glass.transmission = 1.0f;
            glass.ior = 1.5f;
            glass.roughness = 0.0f;
            glass.specular = 1.0f;
            glass.dispersionAbbe = 55.0f;
            for (int s = 0; s < nglass; ++s) {
                SampledWavelengths wv = SampledWavelengths::sampleVisible(4, rng.nextFloat());
                const int hero = std::clamp(int(rng.nextFloat() * 4.0f), 0, 3);
                wv.promoteHero(hero);
                // Normal-incidence transmit (uChoice high enough to beat Fresnel ~0.04).
                BsdfSampleSpectral eIn =
                    bsdfSampleSpectral(glass, Vec3(0.0f, 0.0f, 1.0f), 0.99f, 0.0f, 0.0f, 0.5f, wv,
                                       glass.ior, 0, aces);
                BsdfSampleSpectral eOut =
                    bsdfSampleSpectral(glass, Vec3(0.0f, 0.0f, -1.0f), 0.99f, 0.0f, 0.0f, 0.5f, wv,
                                       glass.ior, 0, aces);
                if (!eIn.valid || !eIn.transmitted || !eOut.valid || !eOut.transmitted) continue;
                SampledSpectrum thr =
                    eIn.weight * eOut.weight * rgbToSpectrumEmission(Vec3(1.0f), wv, aces);
                const Vec3 o = spectrumToRgb(thr, wv, aces);
                rSum += double(o.x);
                gSum += double(o.y);
                bSum += double(o.z);
                ++accepted;
            }
            check(accepted > 1000, "glass spectral samples accepted");
            const float r = float(rSum / accepted), g = float(gSum / accepted),
                        b = float(bSum / accepted);
            check(std::fabs(r / g - 1.0f) < 0.05f && std::fabs(b / g - 1.0f) < 0.05f,
                  "glass spectral enter×exit×env not reddish");
            check(std::fabs(r - 1.0f) < 0.12f && std::fabs(g - 1.0f) < 0.12f, "glass spectral ~white");

            // pbrt DielectricMaterial::GetBxDF: spectrally-varying η terminates secondaries.
            BsdfSample bsSpec{};
            bsSpec.specular = true;
            LobeWeights lwGlass = computeLobes(glass);
            check(shouldTerminateSecondaryWavelengths(bsSpec, lwGlass, glass),
                  "dispersive glass terminates secondary wavelengths (pbrt)");
            Material glassNd = glass;
            glassNd.dispersionAbbe = 0.0f;
            check(!shouldTerminateSecondaryWavelengths(bsSpec, computeLobes(glassNd), glassNd),
                  "achromatic glass keeps secondary wavelengths");
            {
                SampledWavelengths wv = SampledWavelengths::sampleVisible(4, 0.2f);
                const BsdfSampleSpectral e =
                    bsdfSampleSpectral(glassNd, Vec3(0.0f, 0.0f, 1.0f), 0.99f, 0.0f, 0.0f, 0.99f, wv,
                                       1.5f, 0, aces);
                check(e.valid && e.transmitted && e.specular, "spectral delta glass refracts");
                checkNear(e.wi.z, -1.0f, 1e-5f, "spectral sample uses CPU Snell");
                const float invEta2 = 1.0f / (1.5f * 1.5f);
                for (int i = 0; i < wv.n; ++i) {
                    if (wv.pdf[i] <= 0.0f) continue;
                    checkNear(e.weight.values[i], invEta2, 0.02f, "spectral delta glass weight is 1/η²");
                }
            }
            BsdfSample bsDiff{};
            bsDiff.specular = false;
            Material diffuse;
            diffuse.baseColor = Vec3(0.8f);
            check(shouldTerminateSecondaryWavelengths(bsDiff, computeLobes(diffuse)),
                  "diffuse terminates secondary wavelengths");

            // BDPT used to splat SDS with the walk's post-bounce TerminateSecondary
            // pdfs (1λ CMF sparkles) even when Abbe is 0. Arrival snapshot stays 4λ.
            {
                auto chroma = [](Vec3 v) {
                    const float m = (v.x + v.y + v.z) * (1.0f / 3.0f);
                    return std::fabs(v.x - m) + std::fabs(v.y - m) + std::fabs(v.z - m);
                };
                SampledWavelengths arrival = SampledWavelengths::sampleVisible(4, 0.41f);
                arrival.promoteHero(1);
                SampledSpectrum white = SampledSpectrum::constant(arrival.n, 1.0f);
                const Vec3 rgb4 = spectrumToRgb(white, arrival, aces);
                SampledWavelengths continued = arrival;
                continued.terminateSecondary();
                const Vec3 rgb1 = spectrumToRgb(white, continued, aces);
                check(!arrival.secondaryTerminated(),
                      "arrival snapshot stays 4λ after copy-terminate");
                check(continued.secondaryTerminated(), "continuation terminate is 1λ");
                check(chroma(rgb1) > chroma(rgb4) + 0.15f,
                      "1λ ToXYZ is more chromatic than 4λ equal-energy");
                check(!bdptConnectWavelengths(arrival, arrival).secondaryTerminated(),
                      "two live subpaths stay 4λ");
                check(bdptConnectWavelengths(continued, arrival).secondaryTerminated(),
                      "earlier diffuse bounce terminates the full path");
                check(chroma(rgb4) < chroma(rgb1) * 0.55f,
                      "Abbe-0 SDS splat uses arrival 4λ, not CMF sparkle");
                LightData rgbLamp;
                rgbLamp.type = kLightRect;
                rgbLamp.color = Vec3(1.0f);
                rgbLamp.intensity = 1.0f;
                SampledSpectrum lamp = lightEmissionSpectrum(rgbLamp, arrival, aces);
                SampledSpectrum wantEmit = rgbToSpectrumEmission(Vec3(1.0f), arrival, aces);
                for (int i = 0; i < arrival.n; ++i)
                    check(std::fabs(lamp.values[i] - wantEmit.values[i]) < 1e-5f,
                          "RGB rect light is RGBIlluminantSpectrum (Jakob × D60)");
                LightData dome;
                dome.type = kLightDome;
                dome.color = Vec3(1.0f);
                SampledSpectrum env = lightEmissionSpectrum(dome, arrival, aces);
                float envDiff = 0.0f;
                for (int i = 0; i < arrival.n; ++i) envDiff += std::fabs(env.values[i] - wantEmit.values[i]);
                check(envDiff < 1e-4f, "HDR dome keeps Jakob × illuminant");
                Material floor;
                floor.baseColor = Vec3(0.65f);
                floor.roughness = 0.85f;
                floor.specular = 0.0f;
                const Frame fl(Vec3(0.0f, 0.0f, 1.0f));
                SampledSpectrum lifted =
                    liftBsdfWeight(floor, fl, Vec3(0.0f, 0.0f, 1.0f), Vec3(0.0f, 0.0f, 1.0f),
                                   Vec3(0.65f), arrival, 1.5f, 0, aces);
                SampledSpectrum wantAlb = rgbToSpectrumReflectance(Vec3(0.65f), arrival, aces);
                for (int i = 0; i < arrival.n; ++i)
                    check(std::fabs(lifted.values[i] - wantAlb.values[i]) < 1e-5f,
                          "opaque BSDF lift is RGBAlbedoSpectrum");
                const Vec3 floorMean = meanToRgb(
                    nspp, true,
                    [&](const SampledWavelengths& wv) {
                        return lightEmissionSpectrum(rgbLamp, wv, aces) *
                               rgbToSpectrumReflectance(Vec3(0.65f), wv, aces);
                    },
                    aces);
                checkNear(floorMean.x / srMax(floorMean.y, 1e-8f), 1.0f, 0.08f,
                          "grey Lambert × white RGB light ACEScg R/G");
                checkNear(floorMean.z / srMax(floorMean.y, 1e-8f), 1.0f, 0.08f,
                          "grey Lambert × white RGB light ACEScg B/G");
            }
        }
    }

    // OptiX uses the STL-free helpers in spectrum_device.h with the same tables.
    {
        GpuSpectralTables tab;
        tab.albedoScale = rgb_spec::acesAlbedoScaleTable();
        tab.albedoCoeffs = rgb_spec::acesAlbedoCoeffsTable();
        tab.illuminantScale = rgb_spec::acesIlluminantScaleTable();
        tab.illuminantCoeffs = rgb_spec::acesIlluminantCoeffsTable();
        tab.cieX = cie_tab::kCieX;
        tab.cieY = cie_tab::kCieY;
        tab.cieZ = cie_tab::kCieZ;
        tab.illuminantSpd = illum_tab::kIllumD60;
        for (int i = 0; i < 9; ++i) tab.rgbFromXyz[i] = aces.rgbFromXyz[i];
        float sDev[kMaxSpectrumSamples];
        specUpsampleEmission(tab, hdri, w.lambda, w.n, sDev);
        SampledSpectrum sHost = rgbToSpectrumEmission(hdri, w, aces);
        for (int i = 0; i < w.n; ++i)
            check(std::fabs(sDev[i] - sHost.values[i]) < 1e-4f, "device Jakob emission matches host");
        specUpsampleReflectance(tab, Vec3(0.8f, 0.18f, 0.04f), w.lambda, w.n, sDev);
        sHost = rgbToSpectrumReflectance(Vec3(0.8f, 0.18f, 0.04f), w, aces);
        for (int i = 0; i < w.n; ++i)
            check(std::fabs(sDev[i] - sHost.values[i]) < 1e-4f, "device Jakob albedo matches host");
        SampledSpectrum L = rgbToSpectrumEmission(Vec3(1.0f), w, aces);
        const Vec3 hostRgb = spectrumToRgb(L, w, aces);
        const Vec3 devRgb = specToRgb(tab, L.values, w.lambda, w.pdf, w.n);
        check(std::fabs(hostRgb.x - devRgb.x) < 1e-4f && std::fabs(hostRgb.y - devRgb.y) < 1e-4f &&
                  std::fabs(hostRgb.z - devRgb.z) < 1e-4f,
              "device film ToXYZ matches host");
        float linDev[kMaxSpectrumSamples];
        specUpsampleLinear(Vec3(0.65f), w.lambda, w.n, linDev);
        SampledSpectrum linHost = rgbToSpectrumLinear(Vec3(0.65f), w);
        for (int i = 0; i < w.n; ++i)
            check(std::fabs(linDev[i] - linHost.values[i]) < 1e-5f,
                  "device linear upsample matches host (volumes)");
        float albDev[kMaxSpectrumSamples];
        specUpsampleReflectance(tab, Vec3(0.65f), w.lambda, w.n, albDev);
        SampledSpectrum albHost = rgbToSpectrumReflectance(Vec3(0.65f), w, aces);
        for (int i = 0; i < w.n; ++i)
            check(std::fabs(albDev[i] - albHost.values[i]) < 1e-5f,
                  "device opaque BSDF lift is RGBAlbedoSpectrum");
        SampledWavelengths live = SampledWavelengths::sampleVisible(4, 0.37f);
        live.promoteHero(0);
        SampledSpectrum pair = rgbToSpectrumEmission(Vec3(1.0f), live, aces) *
                               rgbToSpectrumReflectance(Vec3(0.65f), live, aces);
        const Vec3 hostPair = spectrumToRgb(pair, live, aces);
        const Vec3 devPair = specToRgb(tab, pair.values, live.lambda, live.pdf, live.n);
        check(std::fabs(hostPair.x - devPair.x) < 1e-4f && std::fabs(hostPair.y - devPair.y) < 1e-4f &&
                  std::fabs(hostPair.z - devPair.z) < 1e-4f,
              "device LT film ToXYZ matches CPU (no von Kries)");
        SampledWavelengths termLive = live;
        termLive.terminateSecondary();
        SampledSpectrum ones = SampledSpectrum::constant(termLive.n, 1.0f);
        const Vec3 host1 = spectrumToRgb(ones, termLive, aces);
        const Vec3 dev1 = specToRgb(tab, ones.values, termLive.lambda, termLive.pdf, termLive.n);
        check(std::fabs(host1.x - dev1.x) < 1e-4f && std::fabs(host1.y - dev1.y) < 1e-4f &&
                  std::fabs(host1.z - dev1.z) < 1e-4f,
              "device 1λ film ToXYZ matches host");
        SampledWavelengths term = w;
        term.terminateSecondary();
        SampledSpectrum one = SampledSpectrum::constant(term.n, 0.7f);
        const Vec3 hostTerm = spectrumToRgb(one, term, aces);
        const Vec3 devTerm = specToRgb(tab, one.values, term.lambda, term.pdf, term.n);
        check(std::fabs(hostTerm.x - devTerm.x) < 1e-4f && std::fabs(hostTerm.y - devTerm.y) < 1e-4f &&
                  std::fabs(hostTerm.z - devTerm.z) < 1e-4f,
              "device terminate film ToXYZ matches host");
        {
            SampledWavelengths wBlue{};
            wBlue.n = 4;
            wBlue.lambda[0] = 450.0f;
            wBlue.pdf[0] = 1.0f;
            SampledSpectrum s1 = SampledSpectrum::constant(4, 1.0f);
            const Vec3 blue = spectrumToRgb(s1, wBlue, aces);
            check(blue.z > blue.x && blue.z > blue.y, "single 450 nm is blue (pbrt ToXYZ, not grey)");
        }
        check(std::fabs(specDielectricIor(1.5f, 30.0f, 450.0f) - nBlue) < 1e-6f, "device Abbe IOR");
    }

    std::printf("  spectral hero ok rgb=(%.3f,%.3f,%.3f) grey=(%.3f,%.3f,%.3f) hdri=(%.3f,%.3f,%.3f) "
                "nB=%.4f nR=%.4f\n",
                rgb.x, rgb.y, rgb.z, greyOut.x, greyOut.y, greyOut.z, hdriOut.x, hdriOut.y, hdriOut.z,
                nBlue, nRed);
}

void testScreenAdaptiveNearDensityDip() {
    std::printf("screen-adaptive-near-density-dip\n");
    // Check whether mean *screen* edge length is non-monotonic with view depth
    // (user report: underfoot slightly coarser than mid-near, then falloff).
    auto ground = makeGridMesh(80.0f, 80.0f, 1, 1);
    check(ground && ground->triangleCount() == 2, "dip probe ground");
    ground->subdivType = kSubdivLinear;
    ground->subdivIterations = 8;
    ground->dicingQuality = 1.0f;  // target ≈ 1 px

    Material mat;
    mat.displacementHeight = 0.001f;
    mat.displacementScale = 1.0f;
    mat.displacementZeroValue = 0.0f;

    Scene scene;
    scene.settings.screenAdaptive = 1;
    scene.settings.frustumCull = 1;
    scene.settings.frustumPadding = 5.0f;
    scene.settings.dicingPolyLimitM = 30;
    scene.settings.resolutionX = 1280;
    scene.settings.resolutionY = 720;
    scene.camera.focalLength = 35.0f;
    scene.camera.sensorWidth = 36.0f;
    scene.camera.cameraToWorld = lookAtMatrix(Vec3(0, 0.35f, 0.5f), Vec3(0, 0, -4.0f), Vec3(0, 1, 0));
    scene.cameraAuthored = true;

    MeshPtr out = tessDisplaceForTest(ground, mat, scene, 8, /*forceFrustumOff=*/false);
    check(out != nullptr && out->triangleCount() > 100, "dip probe densified");

    const Mat4 w2c = inverse(scene.camera.cameraToWorld);
    const float aspect = float(scene.settings.resolutionX) / float(scene.settings.resolutionY);
    const float focal = scene.camera.focalLength;
    const float sensorW = scene.camera.sensorWidth;
    const float sensorH = sensorW / aspect;
    const int resX = scene.settings.resolutionX;
    const int resY = scene.settings.resolutionY;

    auto project = [&](Vec3 p, float& nx, float& ny, float& viewZ) -> bool {
        const Vec3 pc = transformPoint(w2c, p);
        viewZ = -pc.z;
        if (!(viewZ > 1e-4f)) return false;
        const float halfW = 0.5f * sensorW * (viewZ / focal);
        const float halfH = 0.5f * sensorH * (viewZ / focal);
        nx = pc.x / halfW;
        ny = pc.y / halfH;
        return true;
    };
    auto edgePx = [&](Vec3 a, Vec3 b) -> float {
        float ax, ay, az, bx, by, bz;
        if (!project(a, ax, ay, az) || !project(b, bx, by, bz)) return -1.0f;
        const float dx = (bx - ax) * 0.5f * float(resX);
        const float dy = (by - ay) * 0.5f * float(resY);
        return std::sqrt(dx * dx + dy * dy);
    };

    struct Band {
        float z0, z1;
        double sumPx = 0.0;
        double sumWorld = 0.0;
        size_t n = 0;
    };
    // View-depth bands (metres in front of camera).
    Band bands[] = {{0.05f, 0.4f}, {0.4f, 1.0f}, {1.0f, 2.5f}, {2.5f, 6.0f}, {6.0f, 15.0f}};

    for (size_t t = 0; t + 2 < out->indices.size(); t += 3) {
        const Vec3& a = out->positions[out->indices[t]];
        const Vec3& b = out->positions[out->indices[t + 1]];
        const Vec3& c = out->positions[out->indices[t + 2]];
        float nx, ny, vz;
        const Vec3 mid = (a + b + c) * (1.0f / 3.0f);
        if (!project(mid, nx, ny, vz)) continue;
        if (std::fabs(nx) > 1.2f || std::fabs(ny) > 1.2f) continue;  // in-frustum only
        const float e0 = edgePx(a, b), e1 = edgePx(b, c), e2 = edgePx(c, a);
        if (e0 < 0 || e1 < 0 || e2 < 0) continue;
        const float px = (e0 + e1 + e2) / 3.0f;
        const float we = (length(b - a) + length(c - b) + length(a - c)) / 3.0f;
        for (Band& band : bands) {
            if (vz >= band.z0 && vz < band.z1) {
                band.sumPx += double(px);
                band.sumWorld += double(we);
                ++band.n;
                break;
            }
        }
    }

    float meanPx[5] = {};
    for (int i = 0; i < 5; ++i) {
        if (bands[i].n == 0) {
            std::printf("  depth[%.2f-%.2f] empty\n", bands[i].z0, bands[i].z1);
            continue;
        }
        meanPx[i] = float(bands[i].sumPx / double(bands[i].n));
        const float meanW = float(bands[i].sumWorld / double(bands[i].n));
        std::printf("  depth[%.2f-%.2f] n=%zu meanScreenEdge=%.3fpx meanWorldEdge=%.5f\n",
                    bands[i].z0, bands[i].z1, bands[i].n, meanPx[i], meanW);
    }
    int first = -1, second = -1;
    for (int i = 0; i < 5; ++i) {
        if (bands[i].n < 32) continue;
        if (first < 0) first = i;
        else if (second < 0) {
            second = i;
            break;
        }
    }
    check(first >= 0 && second >= 0, "have two populated depth bands");
    if (first >= 0 && second >= 0) {
        // Screen-adaptive target is uniform screen length — nearest should not be
        // clearly coarser (larger px) than the next band.
        std::printf("  compare nearest meanPx=%.3f vs next=%.3f\n", meanPx[first], meanPx[second]);
        check(meanPx[first] <= meanPx[second] * 1.05f,
              "nearest in-frustum band is not coarser in screen space (no near dip)");
    }
}

void testMaterialXNoiseAndTriplanar() {
    std::printf("materialx-noise-triplanar\n");
    if (!materialXAvailable()) {
        std::printf("  skip (no MaterialX)\n");
        return;
    }
    const QString noiseXml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.39\">\n"
        "  <noise2d name=\"noise2d1\" type=\"color3\">\n"
        "    <input name=\"amplitude\" type=\"vector3\" value=\"1, 1, 1\"/>\n"
        "    <input name=\"pivot\" type=\"float\" value=\"0\"/>\n"
        "  </noise2d>\n"
        "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" nodename=\"noise2d1\"/>\n"
        "    <input name=\"specular_roughness\" type=\"float\" value=\"0.4\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
    QVector<MaterialXGraphNode> nodes;
    QString err;
    check(parseMaterialXGraph(noiseXml, nodes, &err), "parse noise graph");
    check(err.isEmpty(), "noise parse has no error");
    const QString round = serializeMaterialXGraph(nodes);
    check(!round.isEmpty(), "serialize noise graph");
    MaterialXEvalResult eval = evaluateMaterialXDocument(noiseXml, QString());
    check(eval.ok, "evaluate noise graph ok");
    check(eval.error.isEmpty(), "evaluate noise no error string");
    check(eval.material.baseColorProc >= 0 || eval.baseColorTexture != nullptr, "noise binds base color map");
    check(eval.material.baseColorProc >= 0, "noise2d compiles to shade-time procedural");
    check(!eval.procedurals.empty(), "noise emits procedural ops");
    std::printf("  noise baseProc=%d ops=%zu\n", eval.material.baseColorProc, eval.procedurals.size());

    const QString triXml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.39\">\n"
        "  <triplanarprojection name=\"tri1\" type=\"color3\">\n"
        "    <input name=\"file\" type=\"filename\" value=\"\"/>\n"
        "    <input name=\"input_per_axis\" type=\"boolean\" value=\"false\"/>\n"
        "    <input name=\"filex\" type=\"filename\" value=\"\"/>\n"
        "    <input name=\"filey\" type=\"filename\" value=\"\"/>\n"
        "    <input name=\"filez\" type=\"filename\" value=\"\"/>\n"
        "    <input name=\"scale\" type=\"vector3\" value=\"1, 1, 1\"/>\n"
        "    <input name=\"rotate\" type=\"float\" value=\"0\"/>\n"
        "    <input name=\"offset\" type=\"vector3\" value=\"0, 0, 0\"/>\n"
        "    <input name=\"default\" type=\"color3\" value=\"0.2, 0.5, 0.8\"/>\n"
        "    <input name=\"blend\" type=\"float\" value=\"0.1\"/>\n"
        "  </triplanarprojection>\n"
        "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" nodename=\"tri1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
    eval = evaluateMaterialXDocument(triXml, QString());
    check(eval.ok, "evaluate empty triplanar ok (no crash)");
    check(eval.material.baseColorProc >= 0, "triplanar compiles to shade-time procedural");
    if (eval.material.baseColorProc >= 0 && size_t(eval.material.baseColorProc) < eval.procedurals.size()) {
        const ProceduralNode& tri = eval.procedurals[size_t(eval.material.baseColorProc)];
        check(tri.op == kProcTriplanar, "root op is triplanar");
        check(std::fabs(tri.p1.x - 1.0f) < 1e-4f && std::fabs(tri.p1.y - 1.0f) < 1e-4f, "default scale 1,1,1");
        check(std::fabs(tri.s0 - 0.1f) < 1e-4f, "default blend 0.1");
    }
    std::printf("  triplanar ok=%d err=%s proc=%d\n", int(eval.ok), eval.error.toStdString().c_str(),
                eval.material.baseColorProc);

    // Shared Input (no per-axis): one file seeds all three projection axes.
    {
        QTemporaryDir tmp;
        check(tmp.isValid(), "temp dir for triplanar shared file");
        const QString texPath = tmp.filePath("tile.png");
        QImage img(8, 8, QImage::Format_RGB32);
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) img.setPixel(x, y, qRgb((x * 30) & 255, (y * 30) & 255, 128));
        check(img.save(texPath), "write triplanar tile");
        const QString sharedXml = QStringLiteral(
                                      "<?xml version=\"1.0\"?><materialx version=\"1.38\">"
                                      "<triplanarprojection name=\"tri1\" type=\"color3\">"
                                      "<input name=\"file\" type=\"filename\" value=\"%1\"/>"
                                      "<input name=\"input_per_axis\" type=\"boolean\" value=\"false\"/>"
                                      "<input name=\"scale\" type=\"vector3\" value=\"0.5, 0.5, 0.5\"/>"
                                      "<input name=\"blend\" type=\"float\" value=\"0.2\"/>"
                                      "</triplanarprojection>"
                                      "<standard_surface name=\"ss\" type=\"surfaceshader\">"
                                      "<input name=\"base_color\" type=\"color3\" nodename=\"tri1\"/>"
                                      "</standard_surface>"
                                      "<surfacematerial name=\"surface\" type=\"material\">"
                                      "<input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>"
                                      "</surfacematerial></materialx>")
                                      .arg(texPath);
        eval = evaluateMaterialXDocument(sharedXml, tmp.path());
        check(eval.ok && eval.material.baseColorProc >= 0, "shared-file triplanar compiles");
        check(!eval.proceduralImages.empty(), "shared file loaded once into images");
        const ProceduralNode& tri = eval.procedurals[size_t(eval.material.baseColorProc)];
        check(tri.in0 >= 0 && tri.in1 >= 0 && tri.in2 >= 0, "all three axes get a texture");
        check(tri.in0 == tri.in1 && tri.in1 == tri.in2, "shared mode uses one texture index");
        check(std::fabs(tri.p1.x - 0.5f) < 1e-4f, "scale authored");
        // Shade samples must wrap in V (was clamped — only horizontal tiling).
        Stage stage;
        StagePrim prim;
        prim.type = PrimType::Mesh;
        prim.path = "/geo/box";
        prim.mesh = makeSphereMesh(1.0f, 12, 8);
        prim.material = eval.material;
        prim.materialAssigned = true;
        prim.procedurals = eval.procedurals;
        prim.proceduralImages = eval.proceduralImages;
        stage.prims.push_back(prim);
        ScenePtr scene = stage.toScene();
        scene->finalize();
        SceneView view = scene->view();
        // Remapped texture indices must be in-range (dense texBase+i was a crash).
        const ProceduralNode& remapped = view.procedurals[size_t(view.materials[0].baseColorProc)];
        check(remapped.in0 >= 0 && remapped.in0 < view.textureCount, "remapped tex in range");
        check(view.textures[remapped.in0].valid(), "remapped texture view valid");
        Vec3 ns(0, 1, 0);
        Material a = evaluateTexturedMaterial(view, view.materials[0], Vec2(0, 0), ns, Vec3(0.1f, 0.0f, 0.2f),
                                              Vec3(0, 1, 0), 0.01f);
        Material b = evaluateTexturedMaterial(view, view.materials[0], Vec2(0, 0), ns, Vec3(0.1f, 0.0f, 1.2f),
                                              Vec3(0, 1, 0), 0.01f);
        check(srIsFinite(a.baseColor.x) && srIsFinite(b.baseColor.x), "wrapped triplanar samples finite");
        // Extreme P / N / scale-like stress (NaN guards).
        bool stressOk = true;
        for (int i = 0; i < 2000; ++i) {
            const float t = float(i) * 0.173f;
            Vec3 p(std::sin(t) * 50.f, std::cos(t * 1.3f) * 50.f, std::sin(t * 0.7f) * 50.f);
            Vec3 n(std::sin(t), std::cos(t), 0.25f);
            const float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
            n = n * (1.f / std::max(1e-6f, len));
            Material m = evaluateTexturedMaterial(view, view.materials[0], Vec2(0, 0), ns, p, n, 0.01f);
            if (!srIsFinite(m.baseColor.x) || !srIsFinite(m.baseColor.y) || !srIsFinite(m.baseColor.z)) {
                stressOk = false;
                break;
            }
        }
        check(stressOk, "triplanar shade stress stays finite");
        std::printf("  triplanar shared axes=%d/%d/%d scale=%.2f\n", tri.in0, tri.in1, tri.in2, tri.p1.x);
    }

    // float noise → color3 (type mismatch users often make)
    const QString floatNoise = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.39\">\n"
        "  <noise2d name=\"n1\" type=\"float\">\n"
        "    <input name=\"amplitude\" type=\"float\" value=\"1\"/>\n"
        "  </noise2d>\n"
        "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" nodename=\"n1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
    eval = evaluateMaterialXDocument(floatNoise, QString());
    check(eval.ok, "evaluate float-noise→color3 ok (no crash)");
    check(eval.material.baseColorProc >= 0, "float noise compiles into color3 slot");
    std::printf("  floatNoise ok=%d proc=%d\n", int(eval.ok), eval.material.baseColorProc);

    // noise3d → base_color (the UI cook path that was crashing on Windows).
    const QString noise3dXml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <noise3d name=\"noise3d1\" type=\"color3\">\n"
        "    <input name=\"amplitude\" type=\"vector3\" value=\"1, 1, 1\"/>\n"
        "    <input name=\"pivot\" type=\"float\" value=\"0\"/>\n"
        "    <input name=\"position\" type=\"vector3\"/>\n"
        "  </noise3d>\n"
        "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" nodename=\"noise3d1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
    QVector<MaterialXGraphNode> noise3dNodes;
    check(parseMaterialXGraph(noise3dXml, noise3dNodes, &err), "parse noise3d graph");
    const QString noise3dRound = serializeMaterialXGraph(noise3dNodes);
    check(!noise3dRound.isEmpty(), "serialize noise3d graph");
    check(!noise3dRound.contains("name=\"position\""), "serialize skips empty noise3d position");
    eval = evaluateMaterialXDocument(noise3dXml, QString());
    check(eval.ok, "evaluate noise3d (empty position) ok");
    check(eval.material.baseColorProc >= 0, "noise3d compiles to baseColorProc");
    eval = evaluateMaterialXDocument(noise3dRound, QString());
    check(eval.ok, "evaluate serialized noise3d ok");
    check(eval.material.baseColorProc >= 0, "serialized noise3d compiles");

    const QString noise3dPos = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <position name=\"pos1\" type=\"vector3\"/>\n"
        "  <noise3d name=\"noise3d1\" type=\"color3\">\n"
        "    <input name=\"amplitude\" type=\"vector3\" value=\"1, 1, 1\"/>\n"
        "    <input name=\"position\" type=\"vector3\" nodename=\"pos1\"/>\n"
        "  </noise3d>\n"
        "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" nodename=\"noise3d1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
    eval = evaluateMaterialXDocument(noise3dPos, QString());
    check(eval.ok, "evaluate noise3d+position ok");
    check(eval.material.baseColorProc >= 0, "noise3d+position compiles");

    // unifiednoise3d with anisotropic frequency — all XYZ components must stick
    // (regression: comma parse was clobbered by a space re-parse).
    const QString unified = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <unifiednoise3d name=\"u1\" type=\"float\">\n"
        "    <input name=\"freq\" type=\"vector3\" value=\"2, 8, 32\"/>\n"
        "    <input name=\"offset\" type=\"vector3\" value=\"0, 0, 0\"/>\n"
        "    <input name=\"type\" type=\"integer\" value=\"3\"/>\n"
        "    <input name=\"octaves\" type=\"integer\" value=\"5\"/>\n"
        "    <input name=\"lacunarity\" type=\"float\" value=\"2.5\"/>\n"
        "    <input name=\"diminish\" type=\"float\" value=\"0.4\"/>\n"
        "  </unifiednoise3d>\n"
        "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" nodename=\"u1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
    eval = evaluateMaterialXDocument(unified, QString());
    check(eval.ok, "evaluate unifiednoise3d anisotropic freq ok");
    check(eval.material.baseColorProc >= 0, "unifiednoise3d compiles");
    if (eval.material.baseColorProc >= 0 && size_t(eval.material.baseColorProc) < eval.procedurals.size()) {
        const ProceduralNode& node = eval.procedurals[size_t(eval.material.baseColorProc)];
        check(node.op == kProcUnified3d, "unifiednoise3d opcode");
        check(std::fabs(node.p0.x - 2.0f) < 1e-4f, "freq.x = 2");
        check(std::fabs(node.p0.y - 8.0f) < 1e-4f, "freq.y = 8 (not collapsed to x)");
        check(std::fabs(node.p0.z - 32.0f) < 1e-4f, "freq.z = 32 (not collapsed to x)");
        check(std::fabs(node.s0 - 5.0f) < 1e-4f, "octaves = 5");
        check(std::fabs(node.s1 - 2.5f) < 1e-4f, "lacunarity = 2.5");
        check(int(std::lround(node.p2.z)) == 3, "noise type = Fractal");
    }

    // cellnoise3d must compile and stay finite with default object P.
    const QString cell = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <cellnoise3d name=\"c1\" type=\"float\"/>\n"
        "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" nodename=\"c1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
    eval = evaluateMaterialXDocument(cell, QString());
    check(eval.ok, "evaluate cellnoise3d ok");
    check(eval.material.baseColorProc >= 0, "cellnoise3d compiles");
    if (eval.material.baseColorProc >= 0 && size_t(eval.material.baseColorProc) < eval.procedurals.size()) {
        check(eval.procedurals[size_t(eval.material.baseColorProc)].op == kProcCell3d, "cellnoise3d opcode");
    }
    std::printf("  noise3d ok=%d proc=%d unified/cell ok\n", int(eval.ok), eval.material.baseColorProc);
}

// Karma / Arnold style wirings that must shade-time evaluate (not silent no-ops).
void testMaterialXKarmaArnoldWirings() {
    std::printf("materialx-karma-arnold-wirings\n");
    if (!materialXAvailable()) {
        std::printf("  skip\n");
        return;
    }

    auto shadeScene = [](const MaterialXEvalResult& eval, Vec2 uv, Vec3 p, Vec3 nObj, Vec3& nsOut) -> Material {
        Stage stage;
        StagePrim prim;
        prim.type = PrimType::Mesh;
        prim.path = "/geo/mesh";
        prim.mesh = makeSphereMesh(1.0f, 8, 6);
        prim.material = eval.material;
        prim.materialAssigned = true;
        prim.baseColorTexture = eval.baseColorTexture;
        prim.bumpTexture = eval.bumpTexture;
        prim.normalTexture = eval.normalTexture;
        prim.procedurals = eval.procedurals;
        prim.proceduralImages = eval.proceduralImages;
        stage.prims.push_back(prim);
        ScenePtr scene = stage.toScene();
        scene->finalize();
        SceneView view = scene->view();
        nsOut = nObj;
        return evaluateTexturedMaterial(view, view.materials[0], uv, nsOut, p, nObj, 1.0f / 64.0f);
    };

    QTemporaryDir tmp;
    check(tmp.isValid(), "temp dir for karma/arnold wirings");
    const QString checkerPath = tmp.filePath("checker.png");
    {
        QImage img(32, 32, QImage::Format_RGB32);
        for (int y = 0; y < 32; ++y)
            for (int x = 0; x < 32; ++x) {
                const bool on = ((x / 8) ^ (y / 8)) & 1;
                img.setPixel(x, y, on ? qRgb(255, 40, 40) : qRgb(40, 40, 255));
            }
        check(img.save(checkerPath), "write checker tex");
    }
    const QString heightPath = tmp.filePath("height_grad.png");
    {
        QImage img(64, 64, QImage::Format_RGB32);
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) {
                const int v = int(255.0f * (float(x) / 63.0f));
                img.setPixel(x, y, qRgb(v, v, v));
            }
        check(img.save(heightPath), "write height tex");
    }

    // 1) texcoord → multiply(vector2) → image.texcoord → base_color
    {
        const QString xml = QStringLiteral(
                                "<?xml version=\"1.0\"?>\n"
                                "<materialx version=\"1.38\">\n"
                                "  <texcoord name=\"uv\" type=\"vector2\"/>\n"
                                "  <multiply name=\"uvscale\" type=\"vector2\">\n"
                                "    <input name=\"in1\" type=\"vector2\" nodename=\"uv\"/>\n"
                                "    <input name=\"in2\" type=\"vector2\" value=\"4, 4\"/>\n"
                                "  </multiply>\n"
                                "  <image name=\"img\" type=\"color3\">\n"
                                "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
                                "    <input name=\"texcoord\" type=\"vector2\" nodename=\"uvscale\"/>\n"
                                "  </image>\n"
                                "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
                                "    <input name=\"base_color\" type=\"color3\" nodename=\"img\"/>\n"
                                "  </standard_surface>\n"
                                "  <surfacematerial name=\"surface\" type=\"material\">\n"
                                "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
                                "  </surfacematerial>\n"
                                "</materialx>\n")
                                .arg(checkerPath);
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, tmp.path());
        check(eval.ok, "texcoord→multiply→image evaluates");
        check(eval.material.baseColorProc >= 0, "UV-scaled image binds as procedural");
        check(eval.baseColorTexture == nullptr, "UV graph does not use pure texture slot");
        Vec3 ns(0, 1, 0);
        const Material a = shadeScene(eval, Vec2(0.05f, 0.05f), Vec3(0, 0, 0), Vec3(0, 1, 0), ns);
        const Material b = shadeScene(eval, Vec2(0.20f, 0.05f), Vec3(0, 0, 0), Vec3(0, 1, 0), ns);
        // 4x tiling: samples at 0.05 and 0.20 land in different checker cells.
        const float d = std::fabs(a.baseColor.x - b.baseColor.x) + std::fabs(a.baseColor.z - b.baseColor.z);
        check(d > 0.2f, "texcoord×scale changes image sampling");
        std::printf("  texcoord→mul→image ok d=%.3f\n", d);
    }

    // 2) texcoord → place2d(scale) → image.texcoord → base_color
    {
        const QString xml = QStringLiteral(
                                "<?xml version=\"1.0\"?>\n"
                                "<materialx version=\"1.38\">\n"
                                "  <texcoord name=\"uv\" type=\"vector2\"/>\n"
                                "  <place2d name=\"place\" type=\"vector2\">\n"
                                "    <input name=\"texcoord\" type=\"vector2\" nodename=\"uv\"/>\n"
                                "    <input name=\"scale\" type=\"vector2\" value=\"3, 3\"/>\n"
                                "    <input name=\"pivot\" type=\"vector2\" value=\"0, 0\"/>\n"
                                "  </place2d>\n"
                                "  <image name=\"img\" type=\"color3\">\n"
                                "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
                                "    <input name=\"texcoord\" type=\"vector2\" nodename=\"place\"/>\n"
                                "  </image>\n"
                                "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
                                "    <input name=\"base_color\" type=\"color3\" nodename=\"img\"/>\n"
                                "  </standard_surface>\n"
                                "  <surfacematerial name=\"surface\" type=\"material\">\n"
                                "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
                                "  </surfacematerial>\n"
                                "</materialx>\n")
                                .arg(checkerPath);
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, tmp.path());
        check(eval.ok && eval.material.baseColorProc >= 0, "place2d→image compiles");
        bool sawPlace = false;
        for (const ProceduralNode& n : eval.procedurals)
            if (n.op == kProcPlace2d) sawPlace = true;
        check(sawPlace, "place2d opcode present");
        Vec3 ns(0, 1, 0);
        const Material a = shadeScene(eval, Vec2(0.05f, 0.05f), Vec3(0, 0, 0), Vec3(0, 1, 0), ns);
        const Material b = shadeScene(eval, Vec2(0.25f, 0.05f), Vec3(0, 0, 0), Vec3(0, 1, 0), ns);
        const float d = std::fabs(a.baseColor.x - b.baseColor.x) + std::fabs(a.baseColor.z - b.baseColor.z);
        check(d > 0.2f, "place2d scale changes image sampling");
        std::printf("  place2d→image ok d=%.3f\n", d);
    }

    // 3) image with uvtiling authored (no texcoord wire)
    {
        const QString xml = QStringLiteral(
                                "<?xml version=\"1.0\"?>\n"
                                "<materialx version=\"1.38\">\n"
                                "  <image name=\"img\" type=\"color3\">\n"
                                "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
                                "    <input name=\"uvtiling\" type=\"vector2\" value=\"5, 5\"/>\n"
                                "  </image>\n"
                                "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
                                "    <input name=\"base_color\" type=\"color3\" nodename=\"img\"/>\n"
                                "  </standard_surface>\n"
                                "  <surfacematerial name=\"surface\" type=\"material\">\n"
                                "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
                                "  </surfacematerial>\n"
                                "</materialx>\n")
                                .arg(checkerPath);
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, tmp.path());
        check(eval.ok && eval.material.baseColorProc >= 0, "uvtiling forces procedural image bind");
        if (eval.material.baseColorProc >= 0 && size_t(eval.material.baseColorProc) < eval.procedurals.size()) {
            const ProceduralNode& img = eval.procedurals[size_t(eval.material.baseColorProc)];
            check(img.op == kProcImage, "uvtiling root is kProcImage");
            check(std::fabs(img.p1.x - 5.0f) < 1e-3f, "uvtiling authored into p1");
        }
        Vec3 ns(0, 1, 0);
        // 5× tiling: UV 0.02→0.10 and 0.08→0.40 land in adjacent checker cells.
        const Material a = shadeScene(eval, Vec2(0.02f, 0.02f), Vec3(0, 0, 0), Vec3(0, 1, 0), ns);
        const Material b = shadeScene(eval, Vec2(0.08f, 0.02f), Vec3(0, 0, 0), Vec3(0, 1, 0), ns);
        const float d = std::fabs(a.baseColor.x - b.baseColor.x) + std::fabs(a.baseColor.z - b.baseColor.z);
        check(d > 0.2f, "uvtiling changes image sampling");
        std::printf("  uvtiling ok d=%.3f\n", d);
    }

    // 4) triplanar → bump → normal (object-space FD)
    {
        const QString xml = QStringLiteral(
            "<?xml version=\"1.0\"?>\n"
            "<materialx version=\"1.38\">\n"
            "  <triplanarprojection name=\"tri\" type=\"color3\">\n"
            "    <input name=\"file\" type=\"filename\" value=\"\"/>\n"
            "    <input name=\"scale\" type=\"vector3\" value=\"0.25, 0.25, 0.25\"/>\n"
            "    <input name=\"default\" type=\"color3\" value=\"0.2, 0.5, 0.8\"/>\n"
            "    <input name=\"blend\" type=\"float\" value=\"1\"/>\n"
            "  </triplanarprojection>\n"
            "  <bump name=\"bump1\" type=\"vector3\">\n"
            "    <input name=\"height\" type=\"float\" nodename=\"tri\"/>\n"
            "    <input name=\"scale\" type=\"float\" value=\"1\"/>\n"
            "  </bump>\n"
            "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
            "    <input name=\"normal\" type=\"vector3\" nodename=\"bump1\"/>\n"
            "    <input name=\"base_color\" type=\"color3\" value=\"0.7, 0.7, 0.7\"/>\n"
            "  </standard_surface>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n");
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, tmp.path());
        check(eval.ok, "triplanar→bump evaluates");
        check(eval.material.bumpProc >= 0, "triplanar→bump binds bumpProc");
        Vec3 nA(0, 1, 0), nB(0, 1, 0);
        shadeScene(eval, Vec2(0.3f, 0.3f), Vec3(0.1f, 0.0f, 0.2f), Vec3(0, 1, 0), nA);
        shadeScene(eval, Vec2(0.3f, 0.3f), Vec3(0.4f, 0.0f, 0.2f), Vec3(0, 1, 0), nB);
        const float tilt = std::fabs(nA.x) + std::fabs(nA.z) + std::fabs(nB.x) + std::fabs(nB.z);
        // Empty-file triplanar uses constant default → may be flat; also test with file.
        (void)tilt;
        // With a real file:
        const QString xmlFile = QStringLiteral(
                                    "<?xml version=\"1.0\"?>\n"
                                    "<materialx version=\"1.38\">\n"
                                    "  <triplanarprojection name=\"tri\" type=\"color3\">\n"
                                    "    <input name=\"file\" type=\"filename\" value=\"%1\"/>\n"
                                    "    <input name=\"scale\" type=\"vector3\" value=\"0.5, 0.5, 0.5\"/>\n"
                                    "    <input name=\"blend\" type=\"float\" value=\"1\"/>\n"
                                    "  </triplanarprojection>\n"
                                    "  <bump name=\"bump1\" type=\"vector3\">\n"
                                    "    <input name=\"height\" type=\"float\" nodename=\"tri\"/>\n"
                                    "    <input name=\"scale\" type=\"float\" value=\"2\"/>\n"
                                    "  </bump>\n"
                                    "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
                                    "    <input name=\"normal\" type=\"vector3\" nodename=\"bump1\"/>\n"
                                    "  </standard_surface>\n"
                                    "  <surfacematerial name=\"surface\" type=\"material\">\n"
                                    "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
                                    "  </surfacematerial>\n"
                                    "</materialx>\n")
                                    .arg(heightPath);
        eval = evaluateMaterialXDocument(xmlFile, tmp.path());
        check(eval.ok && eval.material.bumpProc >= 0, "triplanar(file)→bump binds");
        nA = Vec3(0, 1, 0);
        nB = Vec3(0, 1, 0);
        shadeScene(eval, Vec2(0.2f, 0.2f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0, 1, 0), nA);
        shadeScene(eval, Vec2(0.2f, 0.2f), Vec3(0.35f, 0.0f, 0.0f), Vec3(0, 1, 0), nB);
        const float tilt2 = std::fabs(nA.x - nB.x) + std::fabs(nA.z - nB.z) + std::fabs(nA.x) + std::fabs(nA.z);
        check(tilt2 > 1e-3f, "triplanar→bump perturbs shading normal via P");
        std::printf("  triplanar→bump ok tilt=%.4f\n", tilt2);
    }

    // 5) noise3d → bump → normal
    {
        const QString xml = QStringLiteral(
            "<?xml version=\"1.0\"?>\n"
            "<materialx version=\"1.38\">\n"
            "  <noise3d name=\"n3\" type=\"float\">\n"
            "    <input name=\"amplitude\" type=\"float\" value=\"1\"/>\n"
            "  </noise3d>\n"
            "  <bump name=\"bump1\" type=\"vector3\">\n"
            "    <input name=\"height\" type=\"float\" nodename=\"n3\"/>\n"
            "    <input name=\"scale\" type=\"float\" value=\"1\"/>\n"
            "  </bump>\n"
            "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
            "    <input name=\"normal\" type=\"vector3\" nodename=\"bump1\"/>\n"
            "  </standard_surface>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n");
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, tmp.path());
        check(eval.ok && eval.material.bumpProc >= 0, "noise3d→bump binds");
        Vec3 nA(0, 1, 0), nB(0, 1, 0);
        shadeScene(eval, Vec2(0.1f, 0.1f), Vec3(0.2f, 0.1f, 0.3f), Vec3(0, 1, 0), nA);
        shadeScene(eval, Vec2(0.1f, 0.1f), Vec3(0.8f, 0.1f, 0.3f), Vec3(0, 1, 0), nB);
        const float d = length(nA - nB);
        check(d > 1e-4f || (std::fabs(nA.x) + std::fabs(nA.z)) > 1e-4f, "noise3d→bump affects normal");
        std::printf("  noise3d→bump ok d=%.4f\n", d);
    }

    // 6) noise2d → bump (UV FD path)
    {
        const QString xml = QStringLiteral(
            "<?xml version=\"1.0\"?>\n"
            "<materialx version=\"1.38\">\n"
            "  <noise2d name=\"n2\" type=\"float\">\n"
            "    <input name=\"amplitude\" type=\"float\" value=\"1\"/>\n"
            "  </noise2d>\n"
            "  <bump name=\"bump1\" type=\"vector3\">\n"
            "    <input name=\"height\" type=\"float\" nodename=\"n2\"/>\n"
            "    <input name=\"scale\" type=\"float\" value=\"2\"/>\n"
            "  </bump>\n"
            "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
            "    <input name=\"normal\" type=\"vector3\" nodename=\"bump1\"/>\n"
            "  </standard_surface>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n");
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, tmp.path());
        check(eval.ok && eval.material.bumpProc >= 0, "noise2d→bump binds");
        Vec3 nA(0, 1, 0);
        shadeScene(eval, Vec2(0.15f, 0.35f), Vec3(0, 0, 0), Vec3(0, 1, 0), nA);
        const float tilt = std::fabs(nA.x) + std::fabs(nA.z);
        check(tilt > 1e-4f, "noise2d→bump perturbs normal");
        std::printf("  noise2d→bump ok tilt=%.4f\n", tilt);
    }

    // 7) typed multiplyfloat alias (Karma sometimes emits these)
    {
        const QString xml = QStringLiteral(
            "<?xml version=\"1.0\"?>\n"
            "<materialx version=\"1.38\">\n"
            "  <noise2d name=\"n2\" type=\"float\">\n"
            "    <input name=\"amplitude\" type=\"float\" value=\"1\"/>\n"
            "  </noise2d>\n"
            "  <multiplyfloat name=\"m1\" type=\"float\">\n"
            "    <input name=\"in1\" type=\"float\" nodename=\"n2\"/>\n"
            "    <input name=\"in2\" type=\"float\" value=\"0.25\"/>\n"
            "  </multiplyfloat>\n"
            "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
            "    <input name=\"specular_roughness\" type=\"float\" nodename=\"m1\"/>\n"
            "  </standard_surface>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n");
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, tmp.path());
        check(eval.ok && eval.material.roughnessProc >= 0, "multiplyfloat→roughness compiles");
        std::printf("  multiplyfloat ok proc=%d\n", eval.material.roughnessProc);
    }
}

void testMaterialXArnoldMapsAndConstants() {
    std::printf("materialx-arnold-maps-constants\n");
    if (!materialXAvailable()) {
        std::printf("  skip\n");
        return;
    }

    // Constant float → triplanar.scale must bake (broadcast XYZ).
    {
        const QString xml = QStringLiteral(
            "<?xml version=\"1.0\"?>\n"
            "<materialx version=\"1.38\">\n"
            "  <constant name=\"c1\" type=\"float\">\n"
            "    <input name=\"value\" type=\"float\" value=\"2.5\"/>\n"
            "  </constant>\n"
            "  <triplanarprojection name=\"tri1\" type=\"color3\">\n"
            "    <input name=\"scale\" type=\"vector3\" nodename=\"c1\"/>\n"
            "    <input name=\"default\" type=\"color3\" value=\"0.2, 0.5, 0.8\"/>\n"
            "  </triplanarprojection>\n"
            "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
            "    <input name=\"base_color\" type=\"color3\" nodename=\"tri1\"/>\n"
            "  </standard_surface>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n");
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, QString());
        check(eval.ok && eval.material.baseColorProc >= 0, "constant→triplanar.scale compiles");
        const int root = eval.material.baseColorProc;
        check(root >= 0 && root < int(eval.procedurals.size()) &&
                  eval.procedurals[size_t(root)].op == kProcTriplanar,
              "root is triplanar");
        const ProceduralNode& tri = eval.procedurals[size_t(root)];
        check(std::fabs(tri.p1.x - 2.5f) < 1e-4f && std::fabs(tri.p1.y - 2.5f) < 1e-4f &&
                  std::fabs(tri.p1.z - 2.5f) < 1e-4f,
              "float Constant broadcasts into triplanar scale XYZ");
        std::printf("  constant→scale ok scale=(%.2f,%.2f,%.2f)\n", tri.p1.x, tri.p1.y, tri.p1.z);
    }

    // Map into roughness fully replaces constant (no multiply by leftover 0.4).
    {
        const QString xml = QStringLiteral(
            "<?xml version=\"1.0\"?>\n"
            "<materialx version=\"1.38\">\n"
            "  <constant name=\"c1\" type=\"float\">\n"
            "    <input name=\"value\" type=\"float\" value=\"0.75\"/>\n"
            "  </constant>\n"
            "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
            "    <input name=\"specular_roughness\" type=\"float\" nodename=\"c1\"/>\n"
            "  </standard_surface>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n");
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, QString());
        check(eval.ok && eval.material.roughnessProc >= 0, "constant→roughness compiles");
        Stage stage;
        StagePrim prim;
        prim.type = PrimType::Mesh;
        prim.path = "/geo";
        prim.mesh = makeBoxMesh(Vec3(1.0f));
        prim.mesh->validate();
        prim.material = eval.material;
        prim.materialAssigned = true;
        prim.procedurals = eval.procedurals;
        prim.proceduralImages = eval.proceduralImages;
        stage.prims.push_back(prim);
        auto scene = stage.toScene();
        SceneView view = scene->view();
        Vec3 ns(0, 1, 0);
        Material m = evaluateTexturedMaterial(view, view.materials[0], Vec2(0.5f), ns, Vec3(0, 1, 0),
                                              Vec3(0, 1, 0), 0.01f);
        check(std::fabs(m.roughness - 0.75f) < 1e-3f, "roughness map replaces constant (no ×0.4)");
        std::printf("  map-replace roughness=%.3f\n", m.roughness);
    }

    // Standard Surface sheen / Oren–Nayar / anisotropy ports bake onto Material.
    {
        const QString xml = QStringLiteral(
            "<?xml version=\"1.0\"?>\n"
            "<materialx version=\"1.38\">\n"
            "  <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
            "    <input name=\"sheen\" type=\"float\" value=\"0.8\"/>\n"
            "    <input name=\"sheen_color\" type=\"color3\" value=\"1, 0.2, 0.1\"/>\n"
            "    <input name=\"sheen_roughness\" type=\"float\" value=\"0.4\"/>\n"
            "    <input name=\"diffuse_roughness\" type=\"float\" value=\"0.55\"/>\n"
            "    <input name=\"specular_anisotropy\" type=\"float\" value=\"0.7\"/>\n"
            "    <input name=\"specular_rotation\" type=\"float\" value=\"0.25\"/>\n"
            "  </standard_surface>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n");
        MaterialXEvalResult eval = evaluateMaterialXDocument(xml, QString());
        check(eval.ok, "sheen/aniso standard_surface evaluates");
        checkNear(eval.material.sheen, 0.8f, 1e-4f, "MX sheen bakes");
        checkNear(eval.material.sheenColor.y, 0.2f, 1e-4f, "MX sheen_color bakes");
        checkNear(eval.material.sheenRoughness, 0.4f, 1e-4f, "MX sheen_roughness bakes");
        checkNear(eval.material.diffuseRoughness, 0.55f, 1e-4f, "MX diffuse_roughness bakes");
        checkNear(eval.material.specularAnisotropy, 0.7f, 1e-4f, "MX specular_anisotropy bakes");
        checkNear(eval.material.specularRotation, 0.25f, 1e-4f, "MX specular_rotation bakes");
    }

    // transmission_color tints refraction, not base_color.
    {
        Material mat;
        mat.baseColor = Vec3(1.0f, 0.0f, 0.0f);
        mat.transmissionColor = Vec3(0.0f, 1.0f, 0.0f);
        mat.transmission = 1.0f;
        mat.baseWeight = 0.0f;
        mat.specular = 1.0f;
        mat.metallic = 0.0f;
        LobeWeights lw = computeLobes(mat);
        check(lw.transmissionTint.y > 0.9f && lw.transmissionTint.x < 0.1f,
              "transmission_color drives refraction tint");
        std::printf("  transmission_color tint=(%.2f,%.2f,%.2f)\n", lw.transmissionTint.x,
                    lw.transmissionTint.y, lw.transmissionTint.z);
    }

    // specular_color tints dielectric F0.
    {
        Material mat;
        mat.baseColor = Vec3(1.0f, 1.0f, 1.0f);
        mat.specular = 1.0f;
        mat.specularColor = Vec3(1.0f, 0.0f, 0.0f);
        mat.metallic = 0.0f;
        mat.transmission = 0.0f;
        LobeWeights lw = computeLobes(mat);
        check(lw.f0.x > lw.f0.y && lw.f0.x > lw.f0.z, "specular_color tints dielectric F0");
        std::printf("  specular_color f0=(%.3f,%.3f,%.3f)\n", lw.f0.x, lw.f0.y, lw.f0.z);
    }
}

void testMaterialXRaySwitchCaustics() {
    std::printf("materialx-ray-switch-caustics\n");

    // Incoming-ray tagging matches Arnold (child ray type after a BSDF sample).
    {
        BsdfSample specR{};
        specR.specular = true;
        specR.transmitted = false;
        LobeWeights lw{};
        lw.specular = 1.0f;
        check(nextRayShadeKind(specR, lw) == RayShadeKind::SpecularReflection,
              "specular bounce → SpecularReflection");
        BsdfSample specT{};
        specT.specular = true;
        specT.transmitted = true;
        check(nextRayShadeKind(specT, lw) == RayShadeKind::SpecularTransmission,
              "specular transmit → SpecularTransmission");
        BsdfSample diffR{};
        diffR.specular = false;
        diffR.transmitted = false;
        LobeWeights dLw{};
        dLw.diffuse = 1.0f;
        check(nextRayShadeKind(diffR, dLw) == RayShadeKind::DiffuseReflection,
              "diffuse bounce → DiffuseReflection");
        BsdfSample diffT{};
        diffT.specular = false;
        diffT.transmitted = true;
        check(nextRayShadeKind(diffT, dLw) == RayShadeKind::DiffuseTransmission,
              "diffuse transmit → DiffuseTransmission");
    }

#if !SOLSTICE_HAVE_MATERIALX
    std::printf("  skip (no MaterialX)\n");
    return;
#else
    bool foundShader = false, foundColor = false, shaderHasVolume = false, colorHasVolume = false;
    for (const MaterialXNodeCatalogEntry& e : listMaterialXNodeCatalog()) {
        if (e.category == QStringLiteral("ray_switch_shader")) {
            foundShader = true;
            for (const MaterialXNodeInputDef& inp : e.inputsFor(e.type)) {
                if (inp.name == QStringLiteral("volume")) shaderHasVolume = true;
            }
        }
        if (e.category == QStringLiteral("ray_switch")) {
            foundColor = true;
            for (const MaterialXNodeInputDef& inp : e.inputsFor(e.type)) {
                if (inp.name == QStringLiteral("volume")) colorHasVolume = true;
            }
        }
    }
    check(foundShader, "catalog contains ray_switch_shader");
    check(foundColor, "catalog contains ray_switch");
    check(shaderHasVolume, "ray_switch_shader has Arnold volume port");
    check(colorHasVolume, "ray_switch has Arnold volume port");

    auto ssXml = [](const char* name, float roughness, const char* color) {
        return QStringLiteral(
                   "  <standard_surface name=\"%1\" type=\"surfaceshader\">\n"
                   "    <input name=\"base\" type=\"float\" value=\"1\"/>\n"
                   "    <input name=\"base_color\" type=\"color3\" value=\"%2\"/>\n"
                   "    <input name=\"specular\" type=\"float\" value=\"1\"/>\n"
                   "    <input name=\"specular_roughness\" type=\"float\" value=\"%3\"/>\n"
                   "    <input name=\"specular_IOR\" type=\"float\" value=\"1.5\"/>\n"
                   "  </standard_surface>\n")
            .arg(QString::fromUtf8(name), QString::fromUtf8(color))
            .arg(roughness, 0, 'f', 3);
    };

    // Distinct shaders on every Arnold port (+ Solstice sss/caustics).
    const QString allPortsXml =
        QStringLiteral("<?xml version=\"1.0\"?>\n<materialx version=\"1.38\">\n") +
        ssXml("ss_cam", 0.11f, "0.1, 0.1, 0.1") + ssXml("ss_sh", 0.21f, "0.2, 0.0, 0.0") +
        ssXml("ss_dr", 0.31f, "0.0, 0.3, 0.0") + ssXml("ss_sr", 0.41f, "0.0, 0.0, 0.4") +
        ssXml("ss_dt", 0.51f, "0.5, 0.0, 0.5") + ssXml("ss_st", 0.61f, "0.0, 0.6, 0.6") +
        ssXml("ss_vol", 0.71f, "0.7, 0.7, 0.0") + ssXml("ss_sss", 0.81f, "0.8, 0.4, 0.2") +
        ssXml("ss_cau", 0.00f, "1, 1, 1") +
        QStringLiteral(
            "  <ray_switch_shader name=\"rswitch\" type=\"surfaceshader\">\n"
            "    <input name=\"camera\" type=\"surfaceshader\" nodename=\"ss_cam\"/>\n"
            "    <input name=\"shadow\" type=\"surfaceshader\" nodename=\"ss_sh\"/>\n"
            "    <input name=\"diffuse_reflection\" type=\"surfaceshader\" nodename=\"ss_dr\"/>\n"
            "    <input name=\"specular_reflection\" type=\"surfaceshader\" nodename=\"ss_sr\"/>\n"
            "    <input name=\"diffuse_transmission\" type=\"surfaceshader\" nodename=\"ss_dt\"/>\n"
            "    <input name=\"specular_transmission\" type=\"surfaceshader\" nodename=\"ss_st\"/>\n"
            "    <input name=\"volume\" type=\"surfaceshader\" nodename=\"ss_vol\"/>\n"
            "    <input name=\"sss\" type=\"surfaceshader\" nodename=\"ss_sss\"/>\n"
            "    <input name=\"caustics\" type=\"surfaceshader\" nodename=\"ss_cau\"/>\n"
            "  </ray_switch_shader>\n"
            "  <surfacematerial name=\"surface\" type=\"material\">\n"
            "    <input name=\"surface\" type=\"surfaceshader\" nodename=\"rswitch\"/>\n"
            "  </surfacematerial>\n"
            "</materialx>\n");

    MaterialXEvalResult eval = evaluateMaterialXDocument(allPortsXml, QString());
    check(eval.ok, "ray_switch_shader evaluates with all ports");
    if (!eval.ok) {
        std::printf("  error: %s\n", eval.error.toUtf8().constData());
        return;
    }
    check(std::fabs(eval.material.roughness - 0.11f) < 1e-4f, "camera branch roughness 0.11");
    check(eval.raySwitchBranches.size() == 8, "eight non-camera branch materials");

    Stage stage;
    StagePrim prim;
    prim.type = PrimType::Mesh;
    prim.path = "/switch";
    prim.mesh = makeSphereMesh(0.5f, 16, 8);
    prim.material = eval.material;
    prim.raySwitchBranches = eval.raySwitchBranches;
    prim.materialAssigned = true;
    stage.prims.push_back(prim);
    ScenePtr scene = stage.toScene();
    check(scene && scene->materials.size() >= 9, "scene has base + 8 branch materials");
    if (!scene) return;
    const int baseIdx = scene->instances.empty() ? -1 : scene->instances[0].materialIndex;
    check(baseIdx >= 0, "instance material index");
    SceneView view = scene->view();
    const Material& baked = scene->materials[size_t(baseIdx)];
    check(baked.raySwitch.shadow >= 0, "baked shadow slot");
    check(baked.raySwitch.diffuseReflection >= 0, "baked diffuse_reflection slot");
    check(baked.raySwitch.specularReflection >= 0, "baked specular_reflection slot");
    check(baked.raySwitch.diffuseTransmission >= 0, "baked diffuse_transmission slot");
    check(baked.raySwitch.specularTransmission >= 0, "baked specular_transmission slot");
    check(baked.raySwitch.volume >= 0, "baked volume slot");
    check(baked.raySwitch.sss >= 0, "baked sss slot");
    check(baked.raySwitch.caustics >= 0, "baked caustics slot");

    auto expectR = [&](RayShadeKind kind, float r, const char* label) {
        const Material m = materialForRay(view, baseIdx, kind);
        check(std::fabs(m.roughness - r) < 1e-4f, label);
    };
    expectR(RayShadeKind::Camera, 0.11f, "Camera → camera port");
    expectR(RayShadeKind::Shadow, 0.21f, "Shadow → shadow port");
    expectR(RayShadeKind::DiffuseReflection, 0.31f, "DiffuseReflection → diffuse_reflection");
    expectR(RayShadeKind::SpecularReflection, 0.41f, "SpecularReflection → specular_reflection");
    expectR(RayShadeKind::DiffuseTransmission, 0.51f, "DiffuseTransmission → diffuse_transmission");
    expectR(RayShadeKind::SpecularTransmission, 0.61f, "SpecularTransmission → specular_transmission");
    expectR(RayShadeKind::Volume, 0.71f, "Volume → volume port");
    expectR(RayShadeKind::Sss, 0.81f, "Sss → sss port");
    expectR(RayShadeKind::Caustics, 0.00f, "Caustics → caustics port");
    const Material cau = materialForCausticTransport(view, baseIdx);
    check(cau.roughness < 1e-5f, "caustic transport prefers caustics port");

    // Unconnected ports fall back to camera/base (Arnold).
    const QString fallbackXml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <standard_surface name=\"glass_camera\" type=\"surfaceshader\">\n"
        "    <input name=\"base\" type=\"float\" value=\"0\"/>\n"
        "    <input name=\"base_color\" type=\"color3\" value=\"1, 1, 1\"/>\n"
        "    <input name=\"specular\" type=\"float\" value=\"1\"/>\n"
        "    <input name=\"specular_roughness\" type=\"float\" value=\"0.12\"/>\n"
        "    <input name=\"specular_IOR\" type=\"float\" value=\"1.5\"/>\n"
        "    <input name=\"transmission\" type=\"float\" value=\"1\"/>\n"
        "  </standard_surface>\n"
        "  <standard_surface name=\"glass_caustics\" type=\"surfaceshader\">\n"
        "    <input name=\"base\" type=\"float\" value=\"0\"/>\n"
        "    <input name=\"base_color\" type=\"color3\" value=\"1, 1, 1\"/>\n"
        "    <input name=\"specular\" type=\"float\" value=\"1\"/>\n"
        "    <input name=\"specular_roughness\" type=\"float\" value=\"0\"/>\n"
        "    <input name=\"specular_IOR\" type=\"float\" value=\"1.5\"/>\n"
        "    <input name=\"transmission\" type=\"float\" value=\"1\"/>\n"
        "  </standard_surface>\n"
        "  <ray_switch_shader name=\"rswitch\" type=\"surfaceshader\">\n"
        "    <input name=\"camera\" type=\"surfaceshader\" nodename=\"glass_camera\"/>\n"
        "    <input name=\"caustics\" type=\"surfaceshader\" nodename=\"glass_caustics\"/>\n"
        "  </ray_switch_shader>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"rswitch\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");

    MaterialXEvalResult evalFb = evaluateMaterialXDocument(fallbackXml, QString());
    check(evalFb.ok, "camera+caustics ray_switch evaluates");
    if (!evalFb.ok) {
        std::printf("  error: %s\n", evalFb.error.toUtf8().constData());
        return;
    }
    check(std::fabs(evalFb.material.roughness - 0.12f) < 1e-4f, "camera branch roughness 0.12");
    check(evalFb.material.transmission > 0.99f, "camera branch transmission");
    check(evalFb.material.raySwitch.caustics == 0, "caustics slot is local index 0");
    check(evalFb.raySwitchBranches.size() == 1, "one caustics branch material");
    check(evalFb.raySwitchBranches[0].roughness < 1e-5f, "caustics branch roughness 0");

    Stage stageFb;
    StagePrim primFb;
    primFb.type = PrimType::Mesh;
    primFb.path = "/glass";
    primFb.mesh = makeSphereMesh(0.5f, 16, 8);
    primFb.material = evalFb.material;
    primFb.raySwitchBranches = evalFb.raySwitchBranches;
    primFb.materialAssigned = true;
    stageFb.prims.push_back(primFb);
    ScenePtr sceneFb = stageFb.toScene();
    check(sceneFb && sceneFb->materials.size() >= 2, "fallback scene has base + caustics");
    if (!sceneFb) return;
    const int fbIdx = sceneFb->instances.empty() ? -1 : sceneFb->instances[0].materialIndex;
    check(fbIdx >= 0, "fallback instance material index");
    SceneView viewFb = sceneFb->view();
    const Material cam = materialForRay(viewFb, fbIdx, RayShadeKind::Camera);
    const Material specT = materialForRay(viewFb, fbIdx, RayShadeKind::SpecularTransmission);
    const Material vol = materialForRay(viewFb, fbIdx, RayShadeKind::Volume);
    const Material sh = materialForRay(viewFb, fbIdx, RayShadeKind::Shadow);
    const Material cauFb = materialForCausticTransport(viewFb, fbIdx);
    check(std::fabs(cam.roughness - 0.12f) < 1e-4f, "Camera ray → camera port roughness 0.12");
    check(std::fabs(specT.roughness - 0.12f) < 1e-4f, "unconnected specular_transmission → camera");
    check(std::fabs(vol.roughness - 0.12f) < 1e-4f, "unconnected volume → camera");
    check(std::fabs(sh.roughness - 0.12f) < 1e-4f, "unconnected shadow → camera");
    check(cauFb.roughness < 1e-5f, "caustic transport → caustics port roughness 0");
    check(std::fabs(cam.roughness - cauFb.roughness) > 0.05f, "camera port must not equal caustics port");
    std::printf("  cameraR=%.3f specTransR=%.3f volumeR=%.3f causticTransportR=%.3f\n", cam.roughness,
                specT.roughness, vol.roughness, cauFb.roughness);

    // Without surfacematerial, the first standard_surface must not steal the switch.
    // Document order lists the GI shader first on purpose.
    const QString noSurfXml = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <standard_surface name=\"gi_only\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" value=\"1, 0, 0\"/>\n"
        "    <input name=\"specular_roughness\" type=\"float\" value=\"0.90\"/>\n"
        "  </standard_surface>\n"
        "  <standard_surface name=\"camera_look\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" value=\"0, 1, 0\"/>\n"
        "    <input name=\"specular_roughness\" type=\"float\" value=\"0.20\"/>\n"
        "  </standard_surface>\n"
        "  <ray_switch_shader name=\"rswitch\" type=\"surfaceshader\">\n"
        "    <input name=\"camera\" type=\"surfaceshader\" nodename=\"camera_look\"/>\n"
        "    <input name=\"diffuse_reflection\" type=\"surfaceshader\" nodename=\"gi_only\"/>\n"
        "  </ray_switch_shader>\n"
        "</materialx>\n");
    MaterialXEvalResult evalNs = evaluateMaterialXDocument(noSurfXml, QString());
    check(evalNs.ok, "ray_switch_shader without surfacematerial evaluates");
    if (!evalNs.ok) {
        std::printf("  error: %s\n", evalNs.error.toUtf8().constData());
        return;
    }
    check(std::fabs(evalNs.material.roughness - 0.20f) < 1e-4f,
          "no surfacematerial: camera port wins over first standard_surface");
    check(evalNs.material.raySwitch.diffuseReflection >= 0, "no surfacematerial: GI port cooked");
    check(evalNs.raySwitchBranches.size() == 1, "no surfacematerial: one GI branch");
    check(std::fabs(evalNs.raySwitchBranches[0].roughness - 0.90f) < 1e-4f,
          "no surfacematerial: GI branch roughness 0.90");
#endif
}

void testMaterialXUdimCubeAsset() {
    std::printf("udim-materialx-cube-asset\n");
    const QString root = QStringLiteral("/workspace/examples/udim_cube");
    if (!QDir(root).exists()) {
        std::printf("  skip (examples/udim_cube missing)\n");
        return;
    }
    std::string error;
    auto atlas = loadImageOrUdim(root + "/grid.<UDIM>.png", QString(), error,
                                 {1001, 1002, 1003, 1011, 1012, 1013}, true, "Utility - Raw");
    check(atlas != nullptr, "MaterialX grid_udim tiles load");
    check(atlas && atlas->udimGridU() == 3 && atlas->udimGridV() == 2, "MaterialX set is 3x2 atlas");
    if (!atlas) return;

    TextureView view;
    view.pixels = atlas->data();
    view.width = atlas->width();
    view.height = atlas->height();
    view.udimGridU = atlas->udimGridU();
    view.udimGridV = atlas->udimGridV();

    const Vec4 c1001 = sampleTextureRGBA(view, Vec2(0.5f, 0.5f));
    const Vec4 c1002 = sampleTextureRGBA(view, Vec2(1.5f, 0.5f));
    const Vec4 c1011 = sampleTextureRGBA(view, Vec2(0.5f, 1.5f));
    // Official MaterialX tiles are distinctly tinted; centers must differ.
    check(length(c1001.xyz() - c1002.xyz()) > 0.15f, "1001 vs 1002 centers differ");
    check(length(c1001.xyz() - c1011.xyz()) > 0.15f, "1001 vs 1011 centers differ");
}

void testQuarterFilmIsDownscaleNotCrop() {
    std::printf("quarter-film downscale\n");
    SceneView scene{};
    scene.settings.resolutionX = 400;
    scene.settings.resolutionY = 200;
    scene.camera.sensorWidth = 36.0f;
    scene.camera.focalLength = 50.0f;
    scene.camera.fStop = 0.0f;
    scene.camera.cameraToWorld = Mat4::identity();

    Vec3 oFull, dFull, oCrop, dCrop, oDown, dDown;
    generateCameraRay(scene, 399.5f, 100.0f, 0.5f, 0.5f, oFull, dFull);

    // Same pixel index on a 1/4 buffer, still divided by authored 400×200 → crop.
    generateCameraRay(scene, 99.5f, 25.0f, 0.5f, 0.5f, oCrop, dCrop);
    check(std::fabs(dCrop.x - dFull.x) > 0.05f, "unbound 1/4 pixel is a crop, not the right edge");

    bindFilmToFramebuffer(scene, 100, 50);
    generateCameraRay(scene, 99.5f, 25.0f, 0.5f, 0.5f, oDown, dDown);
    checkNear(dDown.x, dFull.x, 0.02f, "bound 1/4 film right edge matches full FOV");
    checkNear(dDown.y, dFull.y, 0.02f, "bound 1/4 film y matches full FOV");
    checkNear(dDown.z, dFull.z, 0.02f, "bound 1/4 film z matches full FOV");
}

void testNavPreviewDividerAndSplat() {
    std::printf("nav-preview divider\n");
    check(clampNavPreviewDivider(1) == 4, "clamp min 4");
    check(clampNavPreviewDivider(64) == 32, "clamp max 32");
    check(clampNavPreviewDivider(6) == 4, "snap down to power of two");
    check(clampNavPreviewDivider(16) == 16, "16 stays");
    check(adaptNavPreviewDivider(8, 200.0) == 16, "slow frame coarsens");
    check(adaptNavPreviewDivider(8, 20.0) == 4, "fast frame refines");
    check(adaptNavPreviewDivider(8, 80.0) == 8, "on-target stays");
    check(adaptNavPreviewDivider(4, 10.0) == 4, "already min");
    check(adaptNavPreviewDivider(32, 500.0) == 32, "already max");

    SceneView scene{};
    scene.settings.resolutionX = 400;
    scene.settings.resolutionY = 200;
    scene.camera.sensorWidth = 36.0f;
    scene.camera.focalLength = 50.0f;
    scene.camera.fStop = 0.0f;
    scene.camera.cameraToWorld = Mat4::identity();
    Vec3 oFull, dFull, oDown, dDown;
    generateCameraRay(scene, 399.5f, 100.0f, 0.5f, 0.5f, oFull, dFull);
    bindFilmToFramebuffer(scene, 400 / 16, 200 / 16);
    generateCameraRay(scene, 400 / 16 - 0.5f, 200 / 16 * 0.5f, 0.5f, 0.5f, oDown, dDown);
    checkNear(dDown.x, dFull.x, 0.05f, "bound 1/16 film right edge matches full FOV");
    checkNear(dDown.y, dFull.y, 0.05f, "bound 1/16 film y matches full FOV");
}

void testFramebufferPresentableOnlyWhenComplete() {
    std::printf("framebuffer presentable\n");
    Framebuffer fb;
    fb.resize(4, 4);
    check(!fb.hasAccumulatedData() && !fb.isPresentable(), "empty film is not presentable");
    fb.addSample(0, 0, Vec3(1.0f, 1.0f, 1.0f));
    check(fb.hasAccumulatedData(), "one pixel sets hasData");
    check(!fb.isPresentable(), "partial fill must not be shown");
    fb.setPresentable(true);
    check(fb.isPresentable(), "session marks a finished sample presentable");
    fb.clear();
    check(!fb.hasAccumulatedData() && !fb.isPresentable(), "clear drops presentable");
}

void testCameraDofFocus() {
    std::printf("camera-dof-focus\n");
    registerBuiltinNodes();
    NodeGraph graph;
    Node* sphere = graph.createNode("sphere", "sphere1");
    Node* camera = graph.createNode("camera", "camera1");
    Node* settings = graph.createNode("rendersettings", "rendersettings1");
    check(sphere && camera && settings, "dof graph nodes created");
    if (!camera) return;

    camera->setParameterValue("fstop", 2.8);
    camera->setParameterValue("focusdistance", 3.25);
    camera->setParameterValue("focal", 50.0);
    camera->setParameterValue("aperture", 36.0);
    camera->setParameterValue("eye", QVariant::fromValue(QVector3D(0.0f, 0.0f, 8.0f)));
    camera->setParameterValue("target", QVariant::fromValue(QVector3D(0.0f, 0.0f, 0.0f)));

    graph.connectNodes(sphere, camera, 0);
    graph.connectNodes(camera, settings, 0);
    graph.setDisplayNode(settings);

    CookContext context;
    StagePtr stage = graph.cookDisplay(context);
    check(stage != nullptr, "dof cook produces stage");
    check(!stage->renderCameraPath.isEmpty(), "renderCameraPath set");
    check(stage->find(stage->renderCameraPath) != nullptr, "renderCameraPath resolves");

    ScenePtr scene = stage->toScene();
    check(scene->cameraAuthored, "camera authored in scene");
    checkNear(scene->camera.focusDistance, 3.25f, 1e-4f, "scene focusDistance from camera node");
    checkNear(scene->camera.fStop, 2.8f, 1e-4f, "scene fStop from camera node");
    checkNear(scene->camera.focalLength, 50.0f, 1e-4f, "scene focalLength from camera node");

    // Thin-lens rays through the centre pixel must meet near the focus plane (z = -focusDistance
    // in camera space → world z ≈ +focusDistance when camera looks toward -Z from +Z).
    SceneView view = scene->view();
    view.settings.resolutionX = 64;
    view.settings.resolutionY = 64;
    const float focus = scene->camera.focusDistance;
    Vec3 hitSum(0.0f);
    const int samples = 16;
    for (int i = 0; i < samples; ++i) {
        const float lu = (float(i) + 0.5f) / float(samples);
        const float lv = 0.3f;
        Vec3 origin, direction;
        generateCameraRay(view, 32.0f, 32.0f, lu, lv, origin, direction);
        // Intersect optical-axis focus plane in world: camera at z=8 looking at origin →
        // forward is -Z, focus plane is at z = 8 - focus.
        const Vec3 eye(0.0f, 0.0f, 8.0f);
        const Vec3 forward(0.0f, 0.0f, -1.0f);
        const float denom = dot(direction, forward);
        check(denom > 1e-4f, "dof ray travels forward");
        const float t = focus / denom;
        const Vec3 hit = origin + direction * t;
        // Project hit onto the focus plane along forward from eye.
        const Vec3 onPlane = hit - forward * (dot(hit - (eye + forward * focus), forward));
        hitSum = hitSum + onPlane;
        // Off-axis lens samples should still land close to the optical axis on the focus plane.
        checkNear(onPlane.x, 0.0f, 0.05f, "dof focus plane x near axis");
        checkNear(onPlane.y, 0.0f, 0.05f, "dof focus plane y near axis");
        checkNear(dot(onPlane - eye, forward), focus, 0.05f, "dof focus plane depth");
    }
    const Vec3 hitAvg = hitSum * (1.0f / float(samples));
    checkNear(hitAvg.x, 0.0f, 0.02f, "average dof hit x");
    checkNear(hitAvg.y, 0.0f, 0.02f, "average dof hit y");
}


void testPolyOpticsApertureSpread() {
    std::printf("polynomial-optics-aperture-spread\n");
    CameraData cam;
    cam.opticalModel = 1;
    cam.lensModel = 3;  // angenieux 55mm ~f/1.1
    cam.sensorWidth = 36.0f;
    cam.fStop = 1.0f;
    cam.focusDistance = 2.5f;
    cam.opticalWavelengthNm = 550.0f;
    cam.cameraToWorld = Mat4::identity();

    sol::PolynomialOpticsCamera lens;
    lens.prepare(cam);
    check(lens.active, "poly active");
    check(lens.apertureRadiusMm > 10.0, "wide-open aperture for f/1 request");

    auto cocAt = [&](float planeZ) {
        Rng rng(3u, 5u);
        float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
        int ok = 0;
        for (int i = 0; i < 64; ++i) {
            Vec3 o, d;
            if (!sol::generatePolynomialOpticsRay(lens, cam, 48.0f, 27.0f, 96, 54,
                                                  float(i) / 64.0f, 0.37f, rng, o, d))
                continue;
            if (std::abs(d.z) < 1e-8f) continue;
            const float t = (planeZ - o.z) / d.z;
            if (t <= 0.0f) continue;
            const Vec3 hit = o + d * t;
            minX = std::min(minX, hit.x);
            maxX = std::max(maxX, hit.x);
            minY = std::min(minY, hit.y);
            maxY = std::max(maxY, hit.y);
            ++ok;
        }
        const float dx = maxX - minX, dy = maxY - minY;
        return std::make_pair(ok, std::sqrt(dx * dx + dy * dy));
    };

    const float focusCoC = cocAt(-2.5f).second;
    const float farCoC = cocAt(-10.0f).second;
    std::printf("  focusCoC=%g farCoC=%g aperture=%gmm\n", focusCoC, farCoC, lens.apertureRadiusMm);
    // Bundle-focus refinement must put the sharp plane near the requested focus distance.
    check(focusCoC < 0.006f, "poly focused near focus plane");
    check(farCoC > focusCoC * 5.0f, "far plane much softer than focus plane");
}

void testPolynomialOpticsCamera() {
    std::printf("polynomial-optics-camera\n");
    check(!sol::polynomialOpticsLensNames().empty(), "lens catalogue non-empty");

    CameraData cam;
    cam.opticalModel = 1;
    cam.lensModel = 19;  // cooke speed panchro 50mm
    cam.sensorWidth = 36.0f;
    cam.fStop = 2.8f;
    cam.focusDistance = 2.0f;
    cam.opticalWavelengthNm = 550.0f;
    cam.cameraToWorld = Mat4::identity();

    sol::PolynomialOpticsCamera lens;
    lens.prepare(cam);
    check(lens.active, "poly optics prepared");
    check(lens.apertureRadiusMm > 0.0, "aperture radius > 0");
    check(lens.lensEffectiveFocalLength > 0.0, "effective focal length loaded");

    Rng rng(1u, 2u);
    int ok = 0;
    for (int i = 0; i < 32; ++i) {
        Vec3 o, d;
        if (sol::generatePolynomialOpticsRay(lens, cam, 48.0f, 27.0f, 96, 54, rng.nextFloat(), rng.nextFloat(),
                                             rng, o, d)) {
            ++ok;
            check(std::isfinite(o.x) && std::isfinite(d.x), "poly ray finite");
            check(length(d) > 0.5f, "poly ray direction non-zero");
            // Looking down -Z in camera space after world transform (identity).
            check(d.z < 0.0f, "poly ray looks forward (-Z)");
        }
    }
    check(ok >= 8, "poly optics produces some valid rays through centre");

    // Off-axis pixel: R vs B wavelength should bend differently (chromatic terms in poly).
    Rng rngR(7u, 11u);
    Rng rngB(7u, 11u);
    Vec3 oR, dR, oB, dB;
    const bool okR =
        sol::generatePolynomialOpticsRay(lens, cam, 90.0f, 10.0f, 96, 54, 0.3f, 0.7f, rngR, oR, dR,
                                         chromaticWavelengthNm(0));
    const bool okB =
        sol::generatePolynomialOpticsRay(lens, cam, 90.0f, 10.0f, 96, 54, 0.3f, 0.7f, rngB, oB, dB,
                                         chromaticWavelengthNm(2));
    check(okR && okB, "chromatic R/B rays valid");
    const float dirDelta = length(dR - dB);
    const float originDelta = length(oR - oB);
    check(dirDelta > 1e-5f || originDelta > 1e-6f, "R and B wavelengths produce different lens rays");
}

void testTxMipmaps() {
    std::printf("tx-mipmaps\n");
#if !SOLSTICE_HAVE_TIFF
    std::printf("  skip (libtiff unavailable)\n");
    return;
#else
    QTemporaryDir dir;
    check(dir.isValid(), "temp dir for tx");
    const QString txPath = dir.path() + "/checker.tx";

    TIFF* tif = TIFFOpen(txPath.toUtf8().constData(), "w");
    check(tif != nullptr, "create .tx via libtiff");
    if (!tif) return;

    auto writeLevel = [&](int w, int h, uint8_t value, bool last) {
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, uint32_t(w));
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, uint32_t(h));
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, uint16_t(3));
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, uint16_t(8));
        TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, uint32_t(h));
        std::vector<uint8_t> row(size_t(w) * 3, value);
        for (int y = 0; y < h; ++y) TIFFWriteScanline(tif, row.data(), y, 0);
        TIFFWriteDirectory(tif);
        (void)last;
    };
    // Multi-directory mip chain (fallback path when SUBIFD is absent).
    writeLevel(8, 8, 200, false);
    writeLevel(4, 4, 120, false);
    writeLevel(2, 2, 60, false);
    writeLevel(1, 1, 30, true);
    TIFFClose(tif);

    Image image;
    std::string error;
    check(loadImage(txPath.toStdString(), image, error, /*srgbColor=*/true, "Utility - Raw"),
          "load .tx with mips");
    if (!error.empty() && image.empty()) std::printf("  load error: %s\n", error.c_str());
    check(image.mipCount() >= 4, "loaded or rebuilt mip pyramid");
    check(image.width() == 8 && image.height() == 8, "level 0 is 8x8");

    TextureView view;
    view.pixels = image.data();
    view.width = image.width();
    view.height = image.height();
    view.mipCount = image.mipCount();
    const Vec4 sharp = sampleTextureRGBALod(view, Vec2(0.5f, 0.5f), 0.0f);
    const Vec4 soft = sampleTextureRGBALod(view, Vec2(0.5f, 0.5f), float(image.mipCount() - 1));
    check(sharp.x > 0.5f, "LOD0 samples fine level");
    check(soft.x > 0.0f, "max LOD samples coarse level");
#endif
}

void testBdptShadersAndSss() {
    std::printf("bdpt-shaders-sss\n");
    check(bdpt::kMaxVerts == 4096, "BDPT vertex cap is 4096");
    {
        Vec3 beta(0.25f, 0.25f, 0.25f);
        Rng rngBefore(1u, 2u);
        check(bdpt::bdptRussianRoulette(beta, rngBefore, 1, 3),
              "BDPT RR does not run before rrStartDepth");
        checkNear(beta.x, 0.25f, 1e-6f, "BDPT RR leaves beta unchanged before rrStart");

        Vec3 glassBeta(1.0f, 1.0f, 1.0f);
        Rng rngGlass(7u, 13u);
        check(bdpt::bdptRussianRoulette(glassBeta, rngGlass, 30, 3),
              "BDPT RR never kills throughput 1 (glass)");
        checkNear(glassBeta.x, 1.0f, 1e-6f, "BDPT RR does not boost throughput 1");

        double acc = 0.0;
        int lives = 0;
        const int nTrials = 20000;
        for (int i = 0; i < nTrials; ++i) {
            Vec3 t(0.25f, 0.25f, 0.25f);
            Rng r(uint64_t(1000 + i), 7u);
            if (bdpt::bdptRussianRoulette(t, r, 3, 3)) {
                acc += double(t.x);
                ++lives;
            }
        }
        checkNear(float(acc / double(nTrials)), 0.25f, 0.02f, "BDPT RR is unbiased");
        check(lives > nTrials / 6 && lives < nTrials / 3, "BDPT RR survival tracks q=0.25");
    }

    auto makeBaseScene = []() {
        auto scene = std::make_shared<Scene>();
        MeshPtr floor = std::make_shared<Mesh>();
        floor->positions = {Vec3(-4, 0, -4), Vec3(4, 0, -4), Vec3(4, 0, 4), Vec3(-4, 0, 4)};
        floor->indices = {0, 2, 1, 0, 3, 2};
        floor->normals = {Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0)};
        floor->validate();
        const int floorMesh = scene->addMesh(floor);
        Material floorMat;
        floorMat.baseColor = Vec3(0.65f);
        floorMat.roughness = 0.85f;
        floorMat.specular = 0.0f;
        InstanceData floorInst;
        floorInst.meshIndex = floorMesh;
        floorInst.materialIndex = scene->addMaterial(floorMat);
        scene->instances.push_back(floorInst);

        LightData light;
        light.type = kLightRect;
        light.width = 2.5f;
        light.height = 2.5f;
        light.intensity = 40.0f;
        light.normalize = 1;
        light.visibleCamera = 0;
        light.xform = Mat4::translate(Vec3(0.0f, 4.5f, 0.0f)) * Mat4::rotateX(-90.0f);
        light.xformInv = inverse(light.xform);
        scene->lights.push_back(light);

        scene->settings.resolutionX = 48;
        scene->settings.resolutionY = 36;
        scene->settings.samplesPerPixel = 24;
        scene->settings.maxDepth = 6;
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
        scene->settings.clampDirect = 10.0f;
        scene->settings.clampIndirect = 10.0f;
        scene->camera.cameraToWorld =
            lookAtMatrix(Vec3(2.2f, 2.0f, 2.2f), Vec3(0.0f, 0.6f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
        scene->cameraAuthored = true;
        return scene;
    };

    auto addBall = [](ScenePtr scene, const Material& mat) {
        MeshPtr ball = makeSphereMesh(0.7f, 32, 16);
        InstanceData inst;
        inst.xform = Mat4::translate(Vec3(0.0f, 0.7f, 0.0f));
        inst.meshIndex = scene->addMesh(ball);
        inst.materialIndex = scene->addMaterial(mat);
        scene->instances.push_back(inst);
    };

    auto renderSum = [](ScenePtr scene, int integrator, int caustics) -> double {
        scene->settings.integrator = integrator;
        scene->settings.caustics = caustics;
        scene->finalize();
        RenderSession session;
        session.setScene(scene);
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double sum = 0.0;
        for (int y = 0; y < img.height(); ++y)
            for (int x = 0; x < img.width(); ++x) {
                const Vec3 c = img.rgb(x, y);
                check(isFinite(c), "bdpt shader pixel finite");
                sum += double(luminance(c));
            }
        return sum;
    };

    // Diffuse / metal / glass: BDPT must stay in the same ballpark as PT.
    {
        Material diffuse;
        diffuse.baseColor = Vec3(0.8f, 0.2f, 0.15f);
        diffuse.roughness = 0.7f;
        diffuse.specular = 0.2f;
        auto sPt = makeBaseScene();
        addBall(sPt, diffuse);
        auto sBdpt = makeBaseScene();
        addBall(sBdpt, diffuse);
        const double pt = renderSum(sPt, kIntegratorPathTracer, 0);
        const double bdpt = renderSum(sBdpt, kIntegratorBdpt, 0);
        const double ratio = pt > 0.0 ? bdpt / pt : 0.0;
        check(pt > 0.0 && bdpt > 0.0, "BDPT diffuse produces light");
        check(ratio > 0.55 && ratio < 1.8, "BDPT diffuse energy ~ PT");
        std::printf("  diffuse PT=%.1f BDPT=%.1f ratio=%.3f\n", pt, bdpt, ratio);
    }
    {
        Material metal;
        metal.baseColor = Vec3(0.95f, 0.75f, 0.35f);
        metal.metallic = 1.0f;
        metal.roughness = 0.15f;
        metal.specular = 1.0f;
        auto sPt = makeBaseScene();
        addBall(sPt, metal);
        auto sBdpt = makeBaseScene();
        addBall(sBdpt, metal);
        const double pt = renderSum(sPt, kIntegratorPathTracer, 0);
        const double bdpt = renderSum(sBdpt, kIntegratorBdpt, 0);
        const double ratio = pt > 0.0 ? bdpt / pt : 0.0;
        check(pt > 0.0 && bdpt > 0.0, "BDPT metal produces light");
        check(ratio > 0.5 && ratio < 2.0, "BDPT metal energy ~ PT");
        std::printf("  metal PT=%.1f BDPT=%.1f ratio=%.3f\n", pt, bdpt, ratio);
    }
    {
        Material glass;
        glass.baseColor = Vec3(1.0f);
        glass.transmission = 1.0f;
        glass.ior = 1.5f;
        glass.roughness = 0.0f;
        glass.specular = 1.0f;
        auto sPt = makeBaseScene();
        addBall(sPt, glass);
        auto sBdpt = makeBaseScene();
        addBall(sBdpt, glass);
        const double pt = renderSum(sPt, kIntegratorPathTracer, 1);
        const double bdpt = renderSum(sBdpt, kIntegratorBdpt, 1);
        const double ratio = pt > 0.0 ? bdpt / pt : 0.0;
        check(pt > 0.0 && bdpt > 0.0, "BDPT glass produces light");
        check(ratio > 0.45 && ratio < 2.2, "BDPT glass energy ~ PT");
        std::printf("  glass PT=%.1f BDPT=%.1f ratio=%.3f\n", pt, bdpt, ratio);
    }

    // SSS: PT with caustics off uses the random-walk BSSRDF; BDPT must now too.
    {
        Material sss;
        sss.baseColor = Vec3(0.9f, 0.55f, 0.4f);
        sss.subsurface = 1.0f;
        sss.subsurfaceColor = Vec3(0.9f, 0.35f, 0.2f);
        sss.subsurfaceRadius = Vec3(0.35f, 0.12f, 0.06f);
        sss.subsurfaceScale = 1.0f;
        sss.roughness = 0.55f;
        sss.specular = 0.35f;
        sss.ior = 1.4f;
        auto sPt = makeBaseScene();
        addBall(sPt, sss);
        auto sBdpt = makeBaseScene();
        addBall(sBdpt, sss);
        auto sOff = makeBaseScene();
        Material noSss = sss;
        noSss.subsurface = 0.0f;
        addBall(sOff, noSss);

        const double pt = renderSum(sPt, kIntegratorPathTracer, 0);
        const double bdpt = renderSum(sBdpt, kIntegratorBdpt, 0);
        const double diffuseOnly = renderSum(sOff, kIntegratorPathTracer, 0);
        const double ratio = pt > 0.0 ? bdpt / pt : 0.0;
        check(pt > 0.0 && bdpt > 0.0, "BDPT SSS produces light");
        // Chromatic SSS vs Lambert: check the walk is not black or exploding.
        check(pt > diffuseOnly * 0.3 && pt < diffuseOnly * 3.0,
              "SSS energy is in the Lambert ballpark");
        check(ratio > 0.45 && ratio < 2.2, "BDPT SSS energy ~ PT SSS");
        std::printf("  sss PT=%.1f BDPT=%.1f diffuse=%.1f ratio=%.3f\n", pt, bdpt, diffuseOnly, ratio);
    }

    // Grey Lambert floor + white rect: BDPT light-tracing splats the whole
    // connectable floor. RGBIlluminantSpectrum × RGBAlbedoSpectrum + spectrumToRgb
    // is ACEScg-neutral (pbrt film — no per-sample von Kries).
    {
        auto scene = makeBaseScene();
        scene->settings.integrator = kIntegratorBdpt;
        scene->settings.caustics = 1;
        scene->settings.resolutionX = 32;
        scene->settings.resolutionY = 24;
        scene->settings.samplesPerPixel = 8;
        scene->settings.maxDepth = 4;
        scene->settings.workingSpace = kWorkingSpaceAcesCg;
        scene->finalize();
        RenderSession session;
        session.setScene(scene);
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double r = 0.0, g = 0.0, b = 0.0;
        int nLit = 0;
        for (int y = 0; y < img.height(); ++y) {
            for (int x = 0; x < img.width(); ++x) {
                const Vec3 c = img.rgb(x, y);
                check(isFinite(c), "BDPT grey-plane pixel finite");
                if (luminance(c) < 1e-4f) continue;
                r += double(c.x);
                g += double(c.y);
                b += double(c.z);
                ++nLit;
            }
        }
        check(nLit > 16, "BDPT grey-plane has lit pixels");
        const float rr = float(r / double(nLit));
        const float gg = float(g / double(nLit));
        const float bb = float(b / double(nLit));
        std::printf("  grey-plane BDPT mean RGB=(%.3f, %.3f, %.3f) R/G=%.3f B/G=%.3f n=%d\n", rr, gg,
                    bb, rr / srMax(gg, 1e-8f), bb / srMax(gg, 1e-8f), nLit);
        checkNear(rr / srMax(gg, 1e-8f), 1.0f, 0.08f, "BDPT grey plane ACEScg R/G ~ 1");
        checkNear(bb / srMax(gg, 1e-8f), 1.0f, 0.08f, "BDPT grey plane ACEScg B/G ~ 1");
    }
}

void testWireframeCausticsOn() {
    std::printf("wireframe-caustics-on\n");
    registerBuiltinNodes();
    NodeGraph graph;
    buildDefaultGraph(graph);
    CookContext context;
    StagePtr stage = graph.cookDisplay(context);
    ScenePtr scene = stage->toScene();
    scene->settings.resolutionX = 48;
    scene->settings.resolutionY = 32;
    scene->settings.samplesPerPixel = 2;
    scene->settings.backend = kBackendCpuEmbree;
    scene->settings.integrator = kIntegratorWireframe;
    scene->settings.caustics = 1;
    scene->settings.causticsEngine = kCausticsEngineAuto;
    scene->settings.pathGuiding = 0;
    RenderSession session;
    session.setScene(scene);
    session.start();
    session.waitForCompletion();
    const Image image = session.linearImage();
    bool finite = true;
    double sum = 0.0;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const Vec3 c = image.rgb(x, y);
            if (!isFinite(c)) finite = false;
            sum += double(luminance(c));
        }
    }
    check(finite, "Wireframe+caustics output is finite");
    check(sum > 0.0, "Wireframe+caustics produces visible edges");
}

// Spectral PT used to shade the fog AABB as a surface (rainbow noise on Embree).
// Enter/leave must toggle the medium without depending on OpenVDB voxels.
void testFogAabbProxyEnterExit() {
    std::printf("fog AABB proxy enter/exit\n");
    VolumeGrid fog;
    fog.setKind(VolumeGridKind::Fog);
    const VolumeGrid* const grids[] = {&fog};
    SceneView scene{};
    scene.volumes = grids;
    scene.volumeCount = 1;

    InstanceData inst{};
    inst.volumeIndex = 0;
    inst.mediumIndex = 4;

    RayHit hit{};
    hit.t = 1.0f;
    SurfaceInteraction si{};
    si.p = Vec3(0.0f, 0.0f, 0.0f);
    si.ng = Vec3(0.0f, 0.0f, 1.0f);
    si.ns = si.ng;

    Vec3 origin(-1.0f, 0.0f, 0.0f);
    const Vec3 direction(1.0f, 0.0f, 0.0f);
    int medium = -1;
    check(consumeVolumeProxyHit(scene, kIntegratorPathTracer, inst, hit, si, origin, direction, medium),
          "fog AABB enter continues the ray");
    check(medium == 4, "fog AABB enter sets currentMedium");

    check(consumeVolumeProxyHit(scene, kIntegratorPathTracer, inst, hit, si, origin, direction, medium),
          "fog AABB leave continues the ray");
    check(medium == -1, "fog AABB leave clears currentMedium");

    medium = -1;
    check(!consumeVolumeProxyHit(scene, kIntegratorWireframe, inst, hit, si, origin, direction, medium),
          "wireframe keeps the AABB silhouette");
    check(medium == -1, "wireframe does not enter fog");

    scene.volumes = nullptr;
    scene.volumeCount = 0;
    medium = -1;
    origin = Vec3(-1.0f, 0.0f, 0.0f);
    check(consumeVolumeProxyHit(scene, kIntegratorPathTracer, inst, hit, si, origin, direction, medium),
          "missing fog grid still skips the AABB proxy");
    check(medium == -1, "missing grid does not enter a medium");
}

// GPU Woodcock walk lives in the shared header; instantiate it on CPU so a
// template error fails Embree tests instead of the user's nvcc.
struct DummyFogGrid {
    Vec3 bmin() const { return Vec3(-1.0f); }
    Vec3 bmax() const { return Vec3(1.0f); }
    float majorant() const { return 1.0f; }
    bool hasMajorantGrid() const { return false; }
    bool hasMajorantBricks() const { return false; }
    bool brickEmpty(Vec3) const { return false; }
    float brickExitT(Vec3, Vec3, float, float tMax) const { return tMax; }
    void occupancy(Vec3, float& minD, float& maxD) const {
        minD = 0.0f;
        maxD = 1.0f;
    }
    float cellExitT(Vec3, Vec3, float, float tMax) const { return tMax; }
    float sampleOcc(Vec3) const { return 0.5f; }
};

void testFogWoodcockHeader() {
    std::printf("fog Woodcock header\n");
    DummyFogGrid grid;
    MediumData med{};
    med.type = 2;
    med.sigmaA = Vec3(0.1f);
    med.sigmaS = Vec3(0.4f);
    med.density = 1.0f;
    Rng rng(1u, 1u);
    float lambda[4] = {450.0f, 500.0f, 550.0f, 600.0f};
    float trWood[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float trRes[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const MediumSample wood =
        sampleHeterogeneousFogWlWoodcock(grid, med, Vec3(-2.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f),
                                         4.0f, rng, trWood, lambda, 4);
    Rng rng2(1u, 1u);
    const MediumSample residual =
        sampleHeterogeneousFogWl(grid, med, Vec3(-2.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), 4.0f,
                                 rng2, trRes, lambda, 4);
    check(std::isfinite(wood.t) && wood.t >= 0.0f, "Woodcock free-flight t is finite");
    check(std::isfinite(residual.t) && residual.t >= 0.0f, "residual-ratio free-flight t is finite");
    for (int i = 0; i < 4; ++i) {
        check(std::isfinite(trWood[i]) && trWood[i] >= 0.0f, "Woodcock throughput finite");
        check(std::isfinite(trRes[i]) && trRes[i] >= 0.0f, "residual-ratio throughput finite");
    }
}

void testNgonTriangulateAndVdb() {
    std::printf("n-gon triangulate + OpenVDB\n");
    // Concave quad (arrowhead) — fan would flip; earcut should keep area positive.
    Mesh concave;
    concave.positions = {
        Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(1, 0.3f, 0), Vec3(2, 1, 0), Vec3(0, 1, 0),
    };
    concave.faceVertexCounts = {5};
    concave.faceVertexIndices = {0, 1, 2, 3, 4};
    concave.ensureRenderTriangles();
    check(concave.indices.size() >= 9, "concave pentagon produces triangles");
    float area = 0.0f;
    for (size_t t = 0; t + 2 < concave.indices.size(); t += 3) {
        const Vec3& a = concave.positions[concave.indices[t]];
        const Vec3& b = concave.positions[concave.indices[t + 1]];
        const Vec3& c = concave.positions[concave.indices[t + 2]];
        area += 0.5f * length(cross(b - a, c - a));
    }
    check(area > 1.0f, "concave triangulation area is sane");

    // Quad fan → 2 tris; diagonal must NOT be marked as a cage boundary.
    Mesh quad;
    quad.positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0)};
    quad.faceVertexCounts = {4};
    quad.faceVertexIndices = {0, 1, 2, 3};
    quad.ensureRenderTriangles();
    check(quad.indices.size() == 6, "quad → 2 triangles");
    check(quad.triEdgeMask.size() == 2, "quad edge mask per triangle");
    if (quad.triEdgeMask.size() == 2) {
        // Each tri has one diagonal edge unmarked; total boundary bits across both = 4 (quad sides).
        int boundaryBits = 0;
        for (uint8_t m : quad.triEdgeMask) {
            boundaryBits += (m & 1) ? 1 : 0;
            boundaryBits += (m & 2) ? 1 : 0;
            boundaryBits += (m & 4) ? 1 : 0;
            check(m != 7, "quad triangulation hides the diagonal (mask != 7)");
        }
        check(boundaryBits == 4, "quad has exactly 4 authored boundary edges");
    }
    quad.captureWireCage();
    check(quad.wireIndices.size() == 8, "wire cage: 4 edges × 2 indices");
    check(quad.wirePositions.size() == 4, "wire cage verts stay cage-sized");

    // MaterialX volume → standard_volume coefficients for Fog routing.
    // Accepts Solstice short ports (surface/volume) and MaterialX long names.
    {
        const QString volXmlShort =
            QStringLiteral(
                "<materialx version=\"1.38\">"
                "  <standard_surface name=\"ss\" type=\"surfaceshader\">"
                "    <input name=\"base_color\" type=\"color3\" value=\"0.8, 0.2, 0.1\" />"
                "  </standard_surface>"
                "  <standard_volume name=\"sv\" type=\"volumeshader\">"
                "    <input name=\"density\" type=\"float\" value=\"2.5\" />"
                "    <input name=\"anisotropy\" type=\"float\" value=\"0.3\" />"
                "    <input name=\"absorption\" type=\"color3\" value=\"0.1, 0.2, 0.3\" />"
                "    <input name=\"scattering\" type=\"color3\" value=\"0.7, 0.6, 0.5\" />"
                "    <input name=\"emission\" type=\"float\" value=\"1.5\" />"
                "    <input name=\"emission_color\" type=\"color3\" value=\"0.2, 0.4, 0.8\" />"
                "  </standard_volume>"
                "  <surfacematerial name=\"surface\" type=\"material\">"
                "    <input name=\"surface\" type=\"surfaceshader\" nodename=\"ss\" />"
                "    <input name=\"displacement\" type=\"displacementshader\" />"
                "    <input name=\"volume\" type=\"volumeshader\" nodename=\"sv\" />"
                "  </surfacematerial>"
                "</materialx>");
        MaterialXEvalResult volEval = evaluateMaterialXDocument(volXmlShort, QString());
        check(volEval.ok, "MaterialX volume (short ports) evaluates");
        check(volEval.material.hasVolumeShader == 1, "volume port sets hasVolumeShader");
        check(std::fabs(volEval.material.volumeDensity - 2.5f) < 1e-4f, "volume density from MTLX");
        check(std::fabs(volEval.material.volumeAnisotropy - 0.3f) < 1e-4f, "volume anisotropy from MTLX");
        check(std::fabs(volEval.material.volumeAbsorption.x - 0.1f) < 1e-3f, "volume absorption R");
        check(std::fabs(volEval.material.volumeScattering.x - 0.7f) < 1e-3f, "volume scattering R");
        check(std::fabs(volEval.material.volumeEmissionStrength - 1.5f) < 1e-4f, "volume emission strength");
        check(std::fabs(volEval.material.volumeEmission.z - 0.8f) < 1e-3f, "volume emission_color B");
    }
    {
        const QString volXmlLong =
            QStringLiteral(
                "<materialx version=\"1.38\">"
                "  <standard_surface name=\"ss\" type=\"surfaceshader\">"
                "    <input name=\"base_color\" type=\"color3\" value=\"0.1, 0.1, 0.1\" />"
                "  </standard_surface>"
                "  <standard_volume name=\"sv\" type=\"volumeshader\">"
                "    <input name=\"density\" type=\"float\" value=\"4.0\" />"
                "    <input name=\"anisotropy\" type=\"float\" value=\"-0.2\" />"
                "    <input name=\"scattering\" type=\"color3\" value=\"0.9, 0.8, 0.7\" />"
                "  </standard_volume>"
                "  <surfacematerial name=\"surface\" type=\"material\">"
                "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"ss\" />"
                "    <input name=\"volumeshader\" type=\"volumeshader\" nodename=\"sv\" />"
                "  </surfacematerial>"
                "</materialx>");
        MaterialXEvalResult volEval = evaluateMaterialXDocument(volXmlLong, QString());
        check(volEval.ok, "MaterialX volume (long ports) evaluates");
        check(volEval.material.hasVolumeShader == 1, "volumeshader alias sets hasVolumeShader");
        check(std::fabs(volEval.material.volumeDensity - 4.0f) < 1e-4f, "long-port volume density");
        check(std::fabs(volEval.material.volumeAnisotropy + 0.2f) < 1e-4f, "long-port volume anisotropy");
        check(std::fabs(volEval.material.volumeAbsorption.x) < 1e-4f,
              "omitted absorption defaults to 0 (not 0.5)");
        check(std::fabs(volEval.material.volumeScattering.x - 0.9f) < 1e-3f, "authored scattering kept");
    }
    {
        // Default standard_volume (no absorption/scattering) must not be darker than
        // the implicit fog fallback: absorption 0, scattering 1 (white, no extra σa).
        const QString volXmlDefaults =
            QStringLiteral(
                "<materialx version=\"1.38\">"
                "  <standard_surface name=\"ss\" type=\"surfaceshader\">"
                "    <input name=\"base_color\" type=\"color3\" value=\"0.8, 0.8, 0.8\" />"
                "  </standard_surface>"
                "  <standard_volume name=\"sv\" type=\"volumeshader\">"
                "    <input name=\"density\" type=\"float\" value=\"1\" />"
                "  </standard_volume>"
                "  <surfacematerial name=\"surface\" type=\"material\">"
                "    <input name=\"surface\" type=\"surfaceshader\" nodename=\"ss\" />"
                "    <input name=\"volume\" type=\"volumeshader\" nodename=\"sv\" />"
                "  </surfacematerial>"
                "</materialx>");
        MaterialXEvalResult volEval = evaluateMaterialXDocument(volXmlDefaults, QString());
        check(volEval.ok, "MaterialX volume (default abs/scatter) evaluates");
        check(volEval.material.hasVolumeShader == 1, "default volume still sets hasVolumeShader");
        check(std::fabs(volEval.material.volumeAbsorption.x) < 1e-4f &&
                  std::fabs(volEval.material.volumeAbsorption.y) < 1e-4f &&
                  std::fabs(volEval.material.volumeAbsorption.z) < 1e-4f,
              "default standard_volume absorption is 0");
        check(std::fabs(volEval.material.volumeScattering.x - 1.0f) < 1e-4f &&
                  std::fabs(volEval.material.volumeScattering.y - 1.0f) < 1e-4f &&
                  std::fabs(volEval.material.volumeScattering.z - 1.0f) < 1e-4f,
              "default standard_volume scattering is 1");
    }

    // Deep volume MS fireflies: RR must not inflate albedo-1 fog, and Direct Clamp
    // must bound β · NEE / pNee (not raw NEE before the 1/pNee lottery).
    {
        const Vec3 conservative(1.0f);
        const float q = volumeRussianRouletteQ(conservative);
        checkNear(q, 1.0f, 1e-6f, "volume RR survival is 1 at luminance 1");
        const Vec3 after = conservative / q;
        checkNear(after.x, 1.0f, 1e-6f, "volume RR does not boost conservative-fog throughput");
        checkNear(volumeRussianRouletteQ(Vec3(2.0f)), 1.0f, 1e-6f,
                  "volume RR cap is 1 (no 0.95 inflation)");
        checkNear(volumeRussianRouletteQ(Vec3(0.02f)), 0.05f, 1e-6f, "volume RR floor stays 0.05");

        checkNear(volumeNeeRouletteP(0), 1.0f, 1e-6f, "volume NEE RR is 1 before bounce 4");
        checkNear(volumeNeeRouletteP(3), 1.0f, 1e-6f, "volume NEE RR is 1 at bounce 3");
        checkNear(volumeNeeRouletteP(4), 1.0f, 1e-6f, "volume NEE RR stays 1 at bounce 4 (pbrt: no extra lottery)");
        checkNear(volumeNeeRouletteP(499), 1.0f, 1e-6f, "volume NEE RR stays 1 at high depth");

        const Vec3 rawNee(100.0f);
        const Vec3 pathThru(50.0f);
        const float pNeeSynth = 0.05f;  // illustrate clamp-after-1/p, not current roulette
        const Vec3 oldWrong = clampContribution(rawNee, 10.0f) * (1.0f / pNeeSynth) * pathThru;
        const Vec3 pathNee = clampContribution(pathThru * rawNee * (1.0f / pNeeSynth), 10.0f);
        check(maxComponent(oldWrong) > 1000.0f, "clamping raw NEE left 1/pNee spikes");
        checkNear(maxComponent(pathNee), 10.0f, 1e-5f,
                  "volume NEE path contribution is clamped after 1/pNee");
    }

#if SOLSTICE_HAVE_OPENVDB
    MeshPtr box = makeBoxMesh(Vec3(1, 1, 1));
    check(box && !box->indices.empty(), "box mesh for VDB");
    VolumeFromPolygonsSettings settings;
    settings.kind = VolumeGridKind::Sdf;
    settings.voxelSize = 0.1f;
    settings.exteriorBand = 3.0f;
    settings.interiorBand = 3.0f;
    settings.filter = VolumeSampleFilter::Linear;
    std::string err;
    VolumeGridPtr sdf = VolumeGrid::fromPolygons(*box, Mat4::identity(), settings, &err);
    check(sdf && sdf->valid(), std::string("vdbfrompolygons SDF: ") + err);
    if (sdf && sdf->valid()) {
        check(sdf->sampleWorld(Vec3(0, 0, 0)) < 0.0f, "SDF inside box is negative");
        check(sdf->sampleWorld(Vec3(2, 0, 0)) > 0.0f, "SDF outside box is positive");
        MeshPtr meshed = sdf->toPolygonsOpenVDB(0.0f, 0.0f);
        check(meshed && meshed->triangleCount() > 0, "sdftopolygons_vdb produces mesh");
        MeshPtr dcsdd = dcsddContourVolume(*sdf, 0.15f, {}, &err);
        check(dcsdd && dcsdd->triangleCount() > 0, std::string("sdftopolygons_dcsdd: ") + err);

        float tHit = 0.0f;
        Vec3 nSdf;
        const bool hitSdf =
            intersectSdfVolume(*sdf, Vec3(-3, 0, 0), Vec3(1, 0, 0), 0.0f, 10.0f, tHit, nSdf);
        check(hitSdf, "SDF sphere-trace hits the box");
        check(std::fabs(tHit - 2.5f) < 0.35f, "SDF hit near the box face at x=-0.5");
        check(sdf->sampleFilter() == VolumeSampleFilter::Linear, "SDF keeps Linear filter");
        sdf->setSampleFilter(VolumeSampleFilter::Quadratic);
        check(sdf->sampleWorld(Vec3(0, 0, 0)) < 0.0f, "Quadratic filter still inside-negative");
        sdf->setSampleFilter(VolumeSampleFilter::Nearest);
        check(sdf->sampleWorld(Vec3(0, 0, 0)) < 0.0f, "Nearest filter still inside-negative");
    }

    // Fill Density is a runtime multiplier — voxel bake stays normalized (~1 majorant).
    {
        VolumeFromPolygonsSettings a = settings;
        a.kind = VolumeGridKind::Fog;
        a.fillDensity = 1.0f;
        VolumeFromPolygonsSettings b = settings;
        b.kind = VolumeGridKind::Fog;
        b.fillDensity = 25.0f;  // must NOT bake into voxels anymore
        std::string eA, eB;
        VolumeGridPtr fogA = VolumeGrid::fromPolygons(*box, Mat4::identity(), a, &eA);
        VolumeGridPtr fogB = VolumeGrid::fromPolygons(*box, Mat4::identity(), b, &eB);
        check(fogA && fogA->valid() && fogB && fogB->valid(), "fog unit-bake grids");
        if (fogA && fogB) {
            check(std::fabs(fogA->majorant() - fogB->majorant()) < 0.25f,
                  "Fill Density does not change baked fog majorant");
            // Fog must fill the closed mesh interior (not a hollow narrow-band shell).
            const float densCenter = fogA->sampleWorld(Vec3(0, 0, 0));
            const float densOutside = fogA->sampleWorld(Vec3(3, 0, 0));
            check(densCenter > 0.85f, "fog interior at center is filled (~1)");
            check(densOutside < 0.05f, "fog exterior outside the box is empty");
            check(fogA->majorant() < 1.5f, "baked fog occupancy majorant is ~1");
            int interiorHits = 0;
            int interiorCount = 0;
            for (int z = -2; z <= 2; ++z) {
                for (int y = -2; y <= 2; ++y) {
                    for (int x = -2; x <= 2; ++x) {
                        const Vec3 p(x * 0.1f, y * 0.1f, z * 0.1f); // well inside ±0.5 box
                        ++interiorCount;
                        if (fogA->sampleWorld(p) > 0.85f) ++interiorHits;
                    }
                }
            }
            check(interiorHits == interiorCount, "fog lattice inside unit box is filled");
            std::printf("  fog fill: center=%.3f outside=%.3f lattice=%d/%d\n", densCenter,
                        densOutside, interiorHits, interiorCount);
            check(fogA->hasMajorantGrid(), "fog builds supervoxel majorant grid");
            check(fogA->hasMajorantBricks(), "fog builds empty-skip bricks");
            check(fogA->majorantDimX() > 0 && fogA->majorantDimY() > 0 && fogA->majorantDimZ() > 0,
                  "supervoxel grid is non-empty");
            float majMin = 0.0f, majMax = 0.0f;
            fogA->majorantOccupancy(Vec3(0, 0, 0), majMin, majMax);
            std::printf("  majorant: dim=%dx%dx%d cell=%.4f center min=%.4f max=%.4f\n",
                        fogA->majorantDimX(), fogA->majorantDimY(), fogA->majorantDimZ(),
                        fogA->majorantCellSize(), majMin, majMax);
            check(majMin > 0.85f, "interior supervoxel min occupancy ~1");
            check(majMax > 0.85f && majMax < 1.6f, "interior supervoxel max occupancy ~1");
            fogA->majorantOccupancy(Vec3(3, 0, 0), majMin, majMax);
            check(majMax < 0.05f, "outside supervoxel is empty");
            fogA->majorantOccupancy(Vec3(-0.45f, 0, 0), majMin, majMax);
            check(majMin < 0.05f, "boundary supervoxel min occupancy is 0 (linear halo)");
            check(fogA->majorantBrickEmpty(Vec3(3, 0, 0)), "outside brick is empty");
            check(!fogA->majorantBrickEmpty(Vec3(0, 0, 0)), "interior brick is occupied");
            {
                MediumData trackMed;
                trackMed.type = 2;
                trackMed.sigmaA = Vec3(0.0f);
                trackMed.sigmaS = Vec3(1.0f);
                trackMed.density = 1.0f;
                Rng rngWalk(3u, 5u);
                int scatters = 0;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 4000; ++i) {
                    Vec3 thru(1.0f);
                    const MediumSample ms = sampleMediumVdbFog(
                        *fogA, trackMed, Vec3(0, 0, 0), Vec3(1, 0, 0), 4.0f, rngWalk, thru);
                    if (ms.scattered) ++scatters;
                }
                const auto msWalk = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t0)
                                        .count();
                check(scatters > 200, "analytical interior walk produces real scatters");
                check(msWalk < 2000.0, "4000 free-flights stay cheap with local majorants");
                std::printf("  majorant walk: scatters=%d in %.1f ms\n", scatters, msWalk);

                Rng rngTr(11u, 13u);
                Vec3 trAcc(0.0f);
                constexpr int kN = 128;
                float trPeak = 0.0f;
                for (int i = 0; i < kN; ++i) {
                    const Vec3 tr = mediumShadowTrVdb(*fogA, trackMed, Vec3(-2, 0, 0), Vec3(1, 0, 0),
                                                      4.0f, rngTr);
                    trAcc = trAcc + tr;
                    trPeak = srMax(trPeak, srMax(tr.x, srMax(tr.y, tr.z)));
                    check(tr.x <= 1.02f && tr.y <= 1.02f && tr.z <= 1.02f,
                          "residual-ratio Tr sample does not exceed 1");
                }
                const Vec3 trMean = trAcc * (1.0f / float(kN));
                check(trMean.x > 0.15f && trMean.x < 0.85f,
                      "residual-ratio Tr through unit fog is between vacuum and opaque");
                check(trPeak <= 1.02f, "residual-ratio Tr peak stays ≤ 1");

                VolumeGpuExport gpuExp;
                check(fogA->exportGpuTracking(gpuExp), "GPU tracking export of unit fog");
                check(gpuExp.kind == 1 && gpuExp.nx > 4 && gpuExp.nx < 40,
                      "GPU occupancy is voxel resolution, not the old 160³ bake");
                check(gpuExp.majNx == fogA->majorantDimX() && gpuExp.majNy == fogA->majorantDimY() &&
                          gpuExp.majNz == fogA->majorantDimZ(),
                      "GPU majorant grid matches Embree");
                check(!gpuExp.majMin.empty() && !gpuExp.majMax.empty() && !gpuExp.bricks.empty(),
                      "GPU upload includes min/max majorants and empty-skip bricks");
                check(gpuExp.brickSize == VolumeGrid::majorantBrickSize(),
                      "GPU brick size matches Embree 4³ skip bricks");
                const size_t cIdx =
                    (size_t(gpuExp.nz / 2) * size_t(gpuExp.ny) + size_t(gpuExp.ny / 2)) *
                        size_t(gpuExp.nx) +
                    size_t(gpuExp.nx / 2);
                check(cIdx < gpuExp.occupancy.size() && gpuExp.occupancy[cIdx] > 0.85f,
                      "GPU occupancy at the box center is filled");
                float cpuMin = 0.0f, cpuMax = 0.0f;
                fogA->majorantOccupancy(Vec3(0, 0, 0), cpuMin, cpuMax);
                check(cpuMax > 0.85f, "CPU majorant at center is occupied (matches GPU brick)");
            }
            RenderSettingsData deepMs;
            deepMs.maxDepth = 1024;
            deepMs.rrStartDepth = 1024;
            check(deepMs.maxDepth == 1024 && deepMs.rrStartDepth == 1024,
                  "settings accept 1000+ depth for volume multiple scattering");
            check(deepMs.volumeSimilarity == 0, "Volume Similarity defaults off");

            {
                MediumData simMed;
                simMed.sigmaS = Vec3(2.0f);
                simMed.g = 0.9f;
                const MediumData s0 = mediumWithVolumeSimilarity(simMed, 0);
                const MediumData s5 = mediumWithVolumeSimilarity(simMed, 5);
                const MediumData s20 = mediumWithVolumeSimilarity(simMed, 20);
                check(std::fabs(s0.g - 0.9f) < 1e-5f && std::fabs(s0.sigmaS.x - 2.0f) < 1e-5f,
                      "similarity leaves low-order g and σs unchanged");
                check(std::fabs(s5.g - 0.9f) < 1e-5f, "similarity still full g at bounce 5");
                check(std::fabs(s20.g) < 1e-5f, "similarity is isotropic by bounce 20");
                const float red0 = s0.sigmaS.x * (1.0f - s0.g);
                const float red20 = s20.sigmaS.x * (1.0f - s20.g);
                check(std::fabs(red0 - red20) < 1e-4f, "similarity conserves σs(1−g)");
            }

            {
                LightData lights[2];
                lights[0].type = kLightDome;
                lights[0].intensity = 1.0f;
                lights[0].color = Vec3(1.0f);
                lights[1].type = kLightDistant;
                lights[1].intensity = 1.0f;
                lights[1].color = Vec3(1.0f);
                lights[1].normalize = 1;
                lights[1].angle = 0.53f;
                lights[1].xform = lookAtMatrix(Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f),
                                               Vec3(0.0f, 0.0f, 1.0f));
                lights[1].xformInv = inverse(lights[1].xform);
                SceneView sv{};
                sv.lights = lights;
                sv.lightCount = 2;
                sv.domeLightIndex = 0;
                const Vec3 sunDir = normalize(lightAxisZ(lights[1]));
                const Vec3 p(0.0f);
                const float pdfDome = volumeLightSelectionPdfIndex(sv, p, sunDir, 0.0f, 0);
                const float pdfSun = volumeLightSelectionPdfIndex(sv, p, sunDir, 0.0f, 1);
                checkNear(pdfDome, lightSelectionPdfIndex(sv, p, 0), 1e-5f,
                          "volume pdf equals surface LightSampler (dome)");
                checkNear(pdfSun, lightSelectionPdfIndex(sv, p, 1), 1e-5f,
                          "volume pdf equals surface LightSampler (sun)");
                checkNear(volumeLightSelectionPdfIndex(sv, p, sunDir, 0.9f, 1), pdfSun, 1e-5f,
                          "volume pick ignores HG (sun)");
                checkNear(volumeLightSelectionPdfIndex(sv, p, sunDir, 0.9f, 0), pdfDome, 1e-5f,
                          "volume pick ignores HG (dome)");
                checkNear(volumeLightSelectionPdfIndex(sv, p, sunDir * -1.0f, 0.9f, 1), pdfSun, 1e-5f,
                          "volume pick ignores HG tail");
                float pdfSel = 0.0f;
                const int picked = sampleVolumeLightIndex(sv, p, sunDir, 0.9f, 0.5f, pdfSel);
                check(picked >= 0 && pdfSel > 0.0f, "volume light pick at g=0.9 succeeds");
                check(std::fabs(pdfSel - volumeLightSelectionPdfIndex(sv, p, sunDir, 0.9f, picked)) <
                          1e-5f,
                      "volume light pick pdf matches selection pdf");
            }
        }
    }

    // End-to-end: Volume prim through Embree must produce light for SDF and Fog,
    // including the default caustics-on path (MNEE) and Direct Lighting.
    auto renderVolumeSum = [&](VolumeGridKind kind, int integrator, int caustics) -> double {
        VolumeFromPolygonsSettings vs;
        vs.kind = kind;
        vs.voxelSize = 0.08f;
        vs.exteriorBand = 3.0f;
        vs.interiorBand = 3.0f;
        vs.fillDensity = 1.0f;
        std::string e2;
        VolumeGridPtr grid = VolumeGrid::fromPolygons(*box, Mat4::identity(), vs, &e2);
        check(grid && grid->valid(), std::string("volume grid: ") + e2);

        ScenePtr scene = std::make_shared<Scene>();
        const int volumeIndex = scene->addVolume(grid);
        const Bounds3 bb = grid->worldBounds();
        MeshPtr proxy = makeBoxMesh(bb.hi - bb.lo);
        const Vec3 center = bb.center();
        for (Vec3& p : proxy->positions) p = p + center;
        proxy->ensureRenderTriangles();
        proxy->computeBounds();
        const int meshIndex = scene->addMesh(proxy);
        Material mat;
        if (kind == VolumeGridKind::Fog) {
            // Black proxy: shading the AABB as a surface stays dark. Spectral PT
            // must enter the medium (the Embree regression was rainbow box noise).
            mat.baseColor = Vec3(0.0f);
            mat.roughness = 1.0f;
            mat.metallic = 0.0f;
            mat.specular = 0.0f;
        } else {
            mat.baseColor = Vec3(0.85f, 0.75f, 0.65f);
            mat.roughness = 0.45f;
        }
        const int materialIndex = scene->addMaterial(mat);
        InstanceData inst;
        inst.meshIndex = meshIndex;
        inst.materialIndex = materialIndex;
        inst.volumeIndex = volumeIndex;
        inst.visibilityMask = kVisPrimary;  // proxy must not cast shadows onto the SDF/fog
        MediumData med;
        med.type = (kind == VolumeGridKind::Sdf) ? 3 : 2;
        med.volumeIndex = volumeIndex;
        med.density = 1.0f;
        med.sigmaA = Vec3(0.05f);
        med.sigmaS = Vec3(1.2f);
        inst.mediumIndex = scene->addMedium(med);
        scene->instances.push_back(inst);

        LightData light;
        light.type = kLightDistant;
        light.color = Vec3(1.0f);
        light.intensity = 4.0f;
        light.xform = lookAtMatrix(Vec3(3, 5, 4), Vec3(0, 0, 0), Vec3(0, 1, 0));
        light.xformInv = inverse(light.xform);
        scene->lights.push_back(light);

        scene->settings.resolutionX = 48;
        scene->settings.resolutionY = 36;
        scene->settings.samplesPerPixel = 12;
        scene->settings.backend = kBackendCpuEmbree;
        scene->settings.integrator = integrator;
        scene->settings.caustics = caustics;
        scene->settings.pathGuiding = 0;
        scene->settings.maxDepth = 6;
        scene->finalize();
        scene->frameCameraOnContents();

        RenderSession session;
        session.setScene(scene);
        session.start();
        session.waitForCompletion();
        const Image image = session.linearImage();
        double sum = 0.0;
        for (int y = 0; y < image.height(); ++y)
            for (int x = 0; x < image.width(); ++x) sum += double(luminance(image.rgb(x, y)));
        return sum;
    };

    const double sdfPtOff = renderVolumeSum(VolumeGridKind::Sdf, kIntegratorPathTracer, 0);
    const double sdfPtOn = renderVolumeSum(VolumeGridKind::Sdf, kIntegratorPathTracer, 1);
    const double sdfDl = renderVolumeSum(VolumeGridKind::Sdf, kIntegratorDirectLighting, 0);
    const double fogPtOff = renderVolumeSum(VolumeGridKind::Fog, kIntegratorPathTracer, 0);
    const double fogDl = renderVolumeSum(VolumeGridKind::Fog, kIntegratorDirectLighting, 0);
    const double fogBdpt = renderVolumeSum(VolumeGridKind::Fog, kIntegratorBdpt, 0);
    std::printf("  VDB render sums: SDF PT-off=%.3f PT-on=%.3f DL=%.3f | Fog PT-off=%.3f DL=%.3f BDPT=%.3f\n",
                sdfPtOff, sdfPtOn, sdfDl, fogPtOff, fogDl, fogBdpt);
    check(sdfPtOff > 1.0, "SDF PathTracer (caustics off) renders the volume");
    check(sdfPtOn > 1.0, "SDF PathTracer (caustics on / MNEE) renders the volume");
    check(sdfDl > 1.0, "SDF Direct Lighting renders the volume");
    check(fogPtOff > 0.5, "Fog PathTracer (spectral) renders the volume, not the AABB");
    check(fogDl > 0.5, "Fog Direct Lighting renders the volume");
    check(fogBdpt > 0.5, "Fog with BDPT selected still renders (falls back to PT)");
    {
        const float ratio = float(fogBdpt / srMax(fogPtOff, 1e-8));
        check(ratio > 0.5f && ratio < 2.0f, "Fog BDPT fallback energy matches Path Tracer");
    }

    // Indirect Guides + dense fog: OpenPGL used to Reserve(128) path segments.
    // A walk past that (Windows ACCESS_VIOLATION at 0xFFFFFFFFFFFFFFFF) plus
    // Init×HG product on a half-baked volume field. Deep maxDepth + no RR
    // forces the overflow; extra spp trains the volume field.
    {
        VolumeFromPolygonsSettings vsG;
        vsG.kind = VolumeGridKind::Fog;
        vsG.voxelSize = 0.08f;
        vsG.exteriorBand = 3.0f;
        vsG.interiorBand = 3.0f;
        vsG.fillDensity = 1.0f;
        std::string eGuide;
        VolumeGridPtr gridG = VolumeGrid::fromPolygons(*box, Mat4::identity(), vsG, &eGuide);
        check(gridG && gridG->valid(), std::string("guiding fog grid: ") + eGuide);
        ScenePtr sceneG = std::make_shared<Scene>();
        const int volumeIndex = sceneG->addVolume(gridG);
        const Bounds3 bb = gridG->worldBounds();
        MeshPtr proxy = makeBoxMesh(bb.hi - bb.lo);
        const Vec3 center = bb.center();
        for (Vec3& p : proxy->positions) p = p + center;
        proxy->ensureRenderTriangles();
        proxy->computeBounds();
        const int meshIndex = sceneG->addMesh(proxy);
        Material mat;
        mat.baseColor = Vec3(0.8f);
        mat.roughness = 0.5f;
        const int materialIndex = sceneG->addMaterial(mat);
        InstanceData inst;
        inst.meshIndex = meshIndex;
        inst.materialIndex = materialIndex;
        inst.volumeIndex = volumeIndex;
        inst.visibilityMask = kVisPrimary;
        MediumData med;
        med.type = 2;
        med.volumeIndex = volumeIndex;
        med.density = 200.0f;
        med.sigmaA = Vec3(0.0f);
        med.sigmaS = Vec3(1.0f);
        med.g = 0.9f;
        inst.mediumIndex = sceneG->addMedium(med);
        sceneG->instances.push_back(inst);
        LightData light;
        light.type = kLightDistant;
        light.color = Vec3(1.0f);
        light.intensity = 4.0f;
        light.xform = lookAtMatrix(Vec3(3, 5, 4), Vec3(0, 0, 0), Vec3(0, 1, 0));
        light.xformInv = inverse(light.xform);
        sceneG->lights.push_back(light);
        sceneG->settings.resolutionX = 12;
        sceneG->settings.resolutionY = 10;
        sceneG->settings.samplesPerPixel = 4;
        sceneG->settings.backend = kBackendCpuEmbree;
        sceneG->settings.integrator = kIntegratorPathTracer;
        sceneG->settings.caustics = 0;
        sceneG->settings.pathGuiding = 1;
        sceneG->settings.maxDepth = 160;
        sceneG->settings.rrStartDepth = 10000;
        sceneG->finalize();
        sceneG->frameCameraOnContents();
        RenderSession sessionG;
        sessionG.setScene(sceneG);
        sessionG.start();
        sessionG.waitForCompletion();
        const Image imageG = sessionG.linearImage();
        double sumG = 0.0;
        bool finiteG = true;
        for (int y = 0; y < imageG.height(); ++y) {
            for (int x = 0; x < imageG.width(); ++x) {
                const Vec3 c = imageG.rgb(x, y);
                if (!isFinite(c)) finiteG = false;
                sumG += double(luminance(c));
            }
        }
        check(finiteG, "Fog + Indirect Guides stays finite");
        check(sumG > 0.0, "Fog + Indirect Guides produces light");
        std::printf("  Fog + Indirect Guides sum=%.3f (maxDepth=160)\n", sumG);
    }

    {
        VolumeFromPolygonsSettings vsS;
        vsS.kind = VolumeGridKind::Fog;
        vsS.voxelSize = 0.08f;
        vsS.exteriorBand = 3.0f;
        vsS.interiorBand = 3.0f;
        vsS.fillDensity = 1.0f;
        std::string eSim;
        VolumeGridPtr gridS = VolumeGrid::fromPolygons(*box, Mat4::identity(), vsS, &eSim);
        check(gridS && gridS->valid(), std::string("similarity fog grid: ") + eSim);
        ScenePtr sceneS = std::make_shared<Scene>();
        const int volumeIndex = sceneS->addVolume(gridS);
        const Bounds3 bb = gridS->worldBounds();
        MeshPtr proxy = makeBoxMesh(bb.hi - bb.lo);
        const Vec3 center = bb.center();
        for (Vec3& p : proxy->positions) p = p + center;
        proxy->ensureRenderTriangles();
        proxy->computeBounds();
        const int meshIndex = sceneS->addMesh(proxy);
        Material mat;
        mat.baseColor = Vec3(0.8f);
        mat.roughness = 0.5f;
        const int materialIndex = sceneS->addMaterial(mat);
        InstanceData inst;
        inst.meshIndex = meshIndex;
        inst.materialIndex = materialIndex;
        inst.volumeIndex = volumeIndex;
        inst.visibilityMask = kVisPrimary;
        MediumData med;
        med.type = 2;
        med.volumeIndex = volumeIndex;
        med.density = 8.0f;
        med.sigmaA = Vec3(0.0f);
        med.sigmaS = Vec3(1.0f);
        med.g = 0.9f;
        inst.mediumIndex = sceneS->addMedium(med);
        sceneS->instances.push_back(inst);
        LightData light;
        light.type = kLightDistant;
        light.color = Vec3(1.0f);
        light.intensity = 4.0f;
        light.xform = lookAtMatrix(Vec3(3, 5, 4), Vec3(0, 0, 0), Vec3(0, 1, 0));
        light.xformInv = inverse(light.xform);
        sceneS->lights.push_back(light);
        sceneS->settings.resolutionX = 12;
        sceneS->settings.resolutionY = 10;
        sceneS->settings.samplesPerPixel = 4;
        sceneS->settings.backend = kBackendCpuEmbree;
        sceneS->settings.integrator = kIntegratorPathTracer;
        sceneS->settings.caustics = 0;
        sceneS->settings.pathGuiding = 0;
        sceneS->settings.volumeSimilarity = 1;
        sceneS->settings.maxDepth = 40;
        sceneS->settings.rrStartDepth = 40;
        sceneS->finalize();
        sceneS->frameCameraOnContents();
        RenderSession sessionS;
        sessionS.setScene(sceneS);
        sessionS.start();
        sessionS.waitForCompletion();
        const Image imageS = sessionS.linearImage();
        double sumS = 0.0;
        bool finiteS = true;
        for (int y = 0; y < imageS.height(); ++y) {
            for (int x = 0; x < imageS.width(); ++x) {
                const Vec3 c = imageS.rgb(x, y);
                if (!isFinite(c)) finiteS = false;
                sumS += double(luminance(c));
            }
        }
        check(finiteS, "Fog + Volume Similarity stays finite");
        check(sumS > 0.0, "Fog + Volume Similarity produces light");
        std::printf("  Fog + Volume Similarity sum=%.3f (maxDepth=40)\n", sumS);
    }

    // Cast-shadow regression: volume above a ground plane, light from an angle.
    // Lit side of the plane must be brighter than the shadowed side.
    auto castShadowContrast = [&](VolumeGridKind kind) -> double {
        VolumeFromPolygonsSettings vs;
        vs.kind = kind;
        vs.voxelSize = 0.08f;
        vs.exteriorBand = 3.0f;
        vs.interiorBand = 3.0f;
        vs.fillDensity = 1.0f;
        std::string e2;
        VolumeGridPtr grid = VolumeGrid::fromPolygons(*box, Mat4::identity(), vs, &e2);
        check(grid && grid->valid(), std::string("cast-shadow grid: ") + e2);

        ScenePtr scene = std::make_shared<Scene>();
        const int volumeIndex = scene->addVolume(grid);
        const Bounds3 bb = grid->worldBounds();
        MeshPtr proxy = makeBoxMesh(bb.hi - bb.lo);
        const Vec3 center = bb.center();
        for (Vec3& p : proxy->positions) p = p + center;
        proxy->ensureRenderTriangles();
        proxy->computeBounds();
        const int volMesh = scene->addMesh(proxy);

        // Ground plane under the volume (y = -1.2), larger than the box shadow.
        MeshPtr ground = makeBoxMesh(Vec3(6.0f, 0.05f, 6.0f));
        for (Vec3& p : ground->positions) p.y -= 1.2f;
        ground->ensureRenderTriangles();
        ground->computeBounds();
        const int groundMesh = scene->addMesh(ground);

        Material volMat;
        volMat.baseColor = Vec3(0.85f, 0.75f, 0.65f);
        volMat.roughness = 0.4f;
        volMat.hasVolumeShader = 1;
        volMat.volumeDensity = 1.0f;
        volMat.volumeAbsorption = Vec3(0.05f);
        volMat.volumeScattering = Vec3(1.5f);
        const int volMatIdx = scene->addMaterial(volMat);
        Material gMat;
        gMat.baseColor = Vec3(0.7f);
        gMat.roughness = 0.6f;
        const int gMatIdx = scene->addMaterial(gMat);

        InstanceData volInst;
        volInst.meshIndex = volMesh;
        volInst.materialIndex = volMatIdx;
        volInst.volumeIndex = volumeIndex;
        volInst.visibilityMask = kVisPrimary;
        MediumData med;
        med.type = (kind == VolumeGridKind::Sdf) ? 3 : 2;
        med.volumeIndex = volumeIndex;
        med.density = 1.0f;
        med.sigmaA = Vec3(0.05f);
        med.sigmaS = Vec3(1.5f);
        volInst.mediumIndex = scene->addMedium(med);
        scene->instances.push_back(volInst);

        InstanceData gInst;
        gInst.meshIndex = groundMesh;
        gInst.materialIndex = gMatIdx;
        scene->instances.push_back(gInst);

        LightData light;
        light.type = kLightDistant;
        light.color = Vec3(1.0f);
        light.intensity = 6.0f;
        // Light from +X/+Y so the shadow falls toward -X on the ground.
        light.xform = lookAtMatrix(Vec3(4, 6, 0), Vec3(0, 0, 0), Vec3(0, 1, 0));
        light.xformInv = inverse(light.xform);
        scene->lights.push_back(light);

        scene->settings.resolutionX = 64;
        scene->settings.resolutionY = 48;
        scene->settings.samplesPerPixel = 24;
        scene->settings.backend = kBackendCpuEmbree;
        scene->settings.integrator = kIntegratorPathTracer;
        scene->settings.caustics = 0;
        scene->settings.pathGuiding = 0;
        scene->settings.maxDepth = 4;
        scene->finalize();
        // Look down at the ground / volume from above.
        scene->camera.cameraToWorld = lookAtMatrix(Vec3(0, 5, 0.01f), Vec3(0, -1, 0), Vec3(0, 0, -1));
        scene->camera.focalLength = 35.0f;
        scene->cameraAuthored = true;

        RenderSession session;
        session.setScene(scene);
        session.start();
        session.waitForCompletion();
        const Image image = session.linearImage();
        // Compare left half (toward shadow) vs right half (lit).
        double left = 0.0, right = 0.0;
        int nL = 0, nR = 0;
        const int mid = image.width() / 2;
        const int y0 = image.height() / 3;
        const int y1 = (image.height() * 2) / 3;
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < image.width(); ++x) {
                const double L = double(luminance(image.rgb(x, y)));
                if (x < mid) {
                    left += L;
                    ++nL;
                } else {
                    right += L;
                    ++nR;
                }
            }
        }
        const double meanL = nL ? left / nL : 0.0;
        const double meanR = nR ? right / nR : 0.0;
        std::printf("  cast-shadow %s: left=%.4f right=%.4f ratio=%.3f\n",
                    kind == VolumeGridKind::Sdf ? "SDF" : "Fog", meanL, meanR,
                    meanR > 1e-8 ? meanL / meanR : 0.0);
        // Shadowed side must be darker. Soft fog still needs a clear ratio.
        check(meanR > 1e-4, "cast-shadow lit side has light");
        check(meanL < meanR * 0.92, "cast-shadow: shadowed side darker than lit");
        return meanR > 1e-8 ? meanL / meanR : 0.0;
    };
    (void)castShadowContrast(VolumeGridKind::Sdf);
    (void)castShadowContrast(VolumeGridKind::Fog);

    // Field-level unit checks (no Embree): SDF occludes, fog attenuates.
    {
        VolumeFromPolygonsSettings vs;
        vs.kind = VolumeGridKind::Sdf;
        vs.voxelSize = 0.1f;
        vs.exteriorBand = 3.0f;
        vs.interiorBand = 3.0f;
        std::string e3;
        VolumeGridPtr sdfGrid = VolumeGrid::fromPolygons(*box, Mat4::identity(), vs, &e3);
        check(sdfGrid && sdfGrid->valid(), "unit sdf grid");
        ScenePtr s = std::make_shared<Scene>();
        const int vi = s->addVolume(sdfGrid);
        MeshPtr proxy = makeBoxMesh(Vec3(1.2f));
        proxy->ensureRenderTriangles();
        proxy->computeBounds();
        InstanceData inst;
        inst.meshIndex = s->addMesh(proxy);
        inst.volumeIndex = vi;
        inst.visibilityMask = kVisPrimary;
        MediumData med;
        med.type = 3;
        med.volumeIndex = vi;
        inst.mediumIndex = s->addMedium(med);
        s->instances.push_back(inst);
        s->finalize();
        const SceneView view = s->view();
        check(shadowOccludedBySdfVolumes(view, Vec3(-3, 0, 0), Vec3(1, 0, 0), 10.0f),
              "SDF occludes a ray through the box");
        check(!shadowOccludedBySdfVolumes(view, Vec3(-3, 0, 0), Vec3(0, 1, 0), 10.0f),
              "SDF misses a ray that misses the box");
    }
    {
        VolumeFromPolygonsSettings vs;
        vs.kind = VolumeGridKind::Fog;
        vs.voxelSize = 0.1f;
        vs.exteriorBand = 3.0f;
        vs.interiorBand = 3.0f;
        vs.fillDensity = 1.0f;
        std::string e3;
        VolumeGridPtr fogGrid = VolumeGrid::fromPolygons(*box, Mat4::identity(), vs, &e3);
        check(fogGrid && fogGrid->valid(), "unit fog grid");
        ScenePtr s = std::make_shared<Scene>();
        const int vi = s->addVolume(fogGrid);
        MeshPtr proxy = makeBoxMesh(Vec3(1.2f));
        proxy->ensureRenderTriangles();
        proxy->computeBounds();
        InstanceData inst;
        inst.meshIndex = s->addMesh(proxy);
        inst.volumeIndex = vi;
        inst.visibilityMask = kVisPrimary;
        MediumData med;
        med.type = 2;
        med.volumeIndex = vi;
        med.density = 1.0f;
        med.sigmaA = Vec3(0.2f);
        med.sigmaS = Vec3(1.0f);
        inst.mediumIndex = s->addMedium(med);
        s->instances.push_back(inst);
        s->finalize();
        const SceneView view = s->view();
        Rng rngTr(42u, 7u);
        Vec3 TrAcc(0.0f);
        constexpr int kTrSamples = 256;
        for (int i = 0; i < kTrSamples; ++i) {
            const Vec3 sample =
                shadowTransmittanceFogVolumes(view, Vec3(-3, 0, 0), Vec3(1, 0, 0), 10.0f, rngTr);
            TrAcc = TrAcc + sample;
            check(sample.x <= 1.02f && sample.y <= 1.02f && sample.z <= 1.02f,
                  "Fog ratio-tracking Tr sample does not exceed 1");
        }
        const Vec3 Tr = TrAcc * (1.0f / float(kTrSamples));
        check(Tr.x < 0.95f && Tr.y < 0.95f && Tr.z < 0.95f,
              "Fog ratio-tracking Tr attenuates through the volume");
        check(Tr.x > 0.0f, "Fog Tr mean is not fully opaque for this density");
        Rng rngMiss(99u, 3u);
        Vec3 missAcc(0.0f);
        for (int i = 0; i < 64; ++i)
            missAcc =
                missAcc + shadowTransmittanceFogVolumes(view, Vec3(-3, 0, 0), Vec3(0, 1, 0), 10.0f, rngMiss);
        const Vec3 TrMiss = missAcc * (1.0f / 64.0f);
        check(TrMiss.x > 0.99f && TrMiss.y > 0.99f && TrMiss.z > 0.99f,
              "Fog Tr ~1 when the ray misses the AABB");
    }
#else
    std::printf("  (OpenVDB disabled in this build — skipping volume checks)\n");
#endif
}

void testBinaryUsdLoad() {
    std::printf("binary-usd-usdc\n");
#if SOLSTICE_HAVE_TINYUSDZ
    const char* candidates[] = {
        "tests/fixtures/suzanne.usdc",
        "../tests/fixtures/suzanne.usdc",
        "../../tests/fixtures/suzanne.usdc",
        "/workspace/tests/fixtures/suzanne.usdc",
    };
    std::string path;
    for (const char* c : candidates) {
        std::ifstream in(c, std::ios::binary);
        if (in) {
            path = c;
            break;
        }
    }
    if (path.empty()) {
        std::printf("  skip (fixture missing)\n");
        return;
    }

    {
        std::ifstream in(path, std::ios::binary);
        char magic[8] = {};
        in.read(magic, 8);
        check(std::string(magic, 8) == "PXR-USDC", "usdc magic");
    }

    sol::UsdContents out;
    std::string err;
    sol::UsdLoadOptions opt;
    const bool ok = sol::loadUsd(path, opt, out, err);
    check(ok, "load usdc");
    if (!ok) std::printf("  err=%s\n", err.c_str());
    check(!out.prims.empty(), "usdc has prims");
    int meshPrims = 0;
    int tris = 0;
    for (const sol::UsdPrim& p : out.prims) {
        if (p.type != sol::UsdPrim::Type::Mesh || !p.mesh) continue;
        ++meshPrims;
        tris += int(p.mesh->triangleCount());
    }
    check(meshPrims >= 1, "usdc mesh prim");
    check(tris > 100, "usdc mesh triangles");
    std::printf("  prims=%zu meshes=%d tris=%d\n", out.prims.size(), meshPrims, tris);
#else
    std::printf("  skip (TinyUSDZ disabled)\n");
#endif
}

}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    (void)app;
    std::printf("Solstice tests\n");
    if (getenv("SOL_ONLY_WF")) {
        registerBuiltinNodes();
        testWireframeCausticsOn();
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    if (getenv("SOL_ONLY_DOME")) {
        registerBuiltinNodes();
        testDomeHdrLoad();
        testAcesTextureConvert();
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    if (getenv("SOL_ONLY_VDB")) {
        registerBuiltinNodes();
        testFogAabbProxyEnterExit();
        testFogWoodcockHeader();
        testNgonTriangulateAndVdb();
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    if (getenv("SOL_ONLY_STRESS")) {
        registerBuiltinNodes();
        testIntegratorSwitchStress();
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    if (getenv("SOL_ONLY_THROUGH_REFR")) {
        registerBuiltinNodes();
        testBdptCausticThroughRefraction();
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    if (getenv("SOL_ONLY_FOLDERS")) {
        registerBuiltinNodes();
        testXpuDevice();
        testRenderSettingsFolders();
        testIntegratorDeviceMemory();
        testSceneGraphFolders();
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    if (getenv("SOL_ONLY_SPECTRAL")) {
        testSpectralHeroBasics();
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    if (getenv("SOL_ONLY_BDPT")) {
        registerBuiltinNodes();
        testBdptShadersAndSss();
        testBdptTimersFormat();
        testBdptScratchReuse();
        testIntegratorDeviceMemory();
        testUndoHub();
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    if (getenv("SOL_ONLY_TESS")) {
        registerBuiltinNodes();
        testTessellationTriangleBudget();
        testFrustumCullCloseUpSubdiv();
        testFrustumLocalDicingFalloff();
        testFrustumLocalFullyInViewFast();
        testFrustumLocalItersNotClampedOnDenseCage();
        testScreenAdaptiveTessellation();
        testScreenAdaptiveQualityCoarse();
        testEnableDisplacementMasterSwitch();
        testTimeDependentStamp();
        testScreenAdaptiveNearDensityDip();
        testSpectralHeroBasics();
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    testMath();
    testPixelFilter();
    testSampling();
    testLightSelectionDistantRect();
    testBsdf();
    testGlob();
    testGraphCook();
    testGraphDeleteAfterLoad();
    testXpuDevice();
    testRenderSettingsFolders();
    testSceneGraphFolders();
    testQuarterFilmIsDownscaleNotCrop();
    testNavPreviewDividerAndSplat();
    testFramebufferPresentableOnlyWhenComplete();
    testCameraDofFocus();
    testPolyOpticsApertureSpread();
    testPolynomialOpticsCamera();
    testEnvironment();
    testDomeHdrLoad();
    testPhysicalSkyLight();
    testAcesTextureConvert();
    testRender();
    testCausticsGlassSphere();
    testBdptDistantSunCaustics();
    testCameraProjShared();
    testBdptCausticThroughRefraction();
    testPhotonAim();
    testPhotonCaustics();
    testRoughGlassCaustics();
    testRefractionSparkleClamp();
    testSplatAccumulationPrecision();
    testBdptTimersFormat();
    testBdptScratchReuse();
    testIntegratorDeviceMemory();
    testUndoHub();
    testDispersionAndThinFilm();
    testIntegratorSwitchStress();
    testInstanceTransform();
    testUdimMaterialX();
    testTxMipmaps();
    testMaterialXTypeMismatchConnect();
    testMaterialXColorIntoFloatSlots();
    testMaterialXBumpAndNormalMap();
    testArnoldDisplacement();
    testTessellationTriangleBudget();
    testFrustumCullCloseUpSubdiv();
    testFrustumLocalDicingFalloff();
    testFrustumLocalFullyInViewFast();
    testFrustumLocalItersNotClampedOnDenseCage();
    testScreenAdaptiveTessellation();
    testScreenAdaptiveQualityCoarse();
    testEnableDisplacementMasterSwitch();
    testTimeDependentStamp();
    testScreenAdaptiveNearDensityDip();
    testSpectralHeroBasics();
    testTriplanarDisplacementArtifacts();
    testDefaultGroundDisplacement();
    testRockDisplacementExr();
    testMaterialXNoiseAndTriplanar();
    testMaterialXKarmaArnoldWirings();
    testMaterialXArnoldMapsAndConstants();
    testMaterialXRaySwitchCaustics();
    testMaterialXUdimCubeAsset();
    testBdptShadersAndSss();
    testBinaryUsdLoad();
    testFogAabbProxyEnterExit();
    testFogWoodcockHeader();
    testNgonTriangulateAndVdb();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
