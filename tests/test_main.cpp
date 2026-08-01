// Unit tests for the maths, sampling, node graph and renderer plumbing.
#include <chrono>
#include <cmath>
#include <functional>
#include <utility>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <QColor>
#include <QDir>
#include <QImage>
#include <QTemporaryDir>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QVector3D>

#include "app/default_scene.h"
#include "core/image.h"
#include "core/rng.h"
#include "io/alembic_loader.h"
#include "io/image_io.h"
#include "io/materialx_graph.h"
#include "io/usd_loader.h"
#include "nodes/node_graph.h"
#include "nodes/node_registry.h"
#include "nodes/stage.h"
#include "render/cpu/polynomial_optics.h"
#include "render/framebuffer.h"
#include "render/integrator.h"
#include "render/render_session.h"
#include "render/shading.h"
#include "scene/scene.h"
#include "scene/displace.h"
#include "scene/tessellate.h"
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

    // Integrator smoke tests: PT+MNEE caustics and the BDPT integrator.
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
    smokeIntegrator(kIntegratorPathTracer, 1, "PT + MNEE caustics");
    smokeIntegrator(kIntegratorPathTracer, 0, "PT caustics off");
    smokeIntegrator(kIntegratorBdpt, 1, "BDPT integrator");
    // Path guiding smoke: same scene with OpenPGL training enabled (no-op when
    // the build lacks OpenPGL).
    scene->settings.pathGuiding = 1;
    smokeIntegrator(kIntegratorPathTracer, 1, "PT + guiding");
    smokeIntegrator(kIntegratorBdpt, 1, "BDPT + guiding");
    scene->settings.pathGuiding = 0;
    scene->settings.integrator = kIntegratorPathTracer;
    scene->settings.caustics = 1;
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
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
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
    check(pointRatio > 0.8 && pointRatio < 1.25, "BDPT and PT point-light caustics agree");
    std::printf("  pointOn=%.1f pointOff=%.1f pointBDPT=%.1f ratio=%.3f\n", sumPointOn, sumPointOff,
                sumPointBdpt, pointRatio);
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
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
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
    check(ratio > 0.55 && ratio < 1.8, "BDPT and PT through-glass caustic energy agree");
    std::printf("  bdptOn=%.1f bdptOff=%.1f ptOn=%.1f ratio=%.3f\n", sumBdptOn, sumBdptOff, sumPtOn, ratio);
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
        scene->settings.pathGuiding = 0;
        scene->settings.envVisibleCamera = 0;
        scene->settings.clampIndirect = 0.0f;
        scene->camera.cameraToWorld =
            lookAtMatrix(Vec3(2.4f, 2.6f, 2.4f), Vec3(0.0f, 0.35f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
        scene->cameraAuthored = true;
        scene->finalize();
        return scene;
    };
    auto render = [&](float abbe, double& chroma, bool& finite) {
        RenderSession session;
        session.setScene(buildScene(abbe));
        session.start();
        session.waitForCompletion();
        const Image img = session.linearImage();
        double sum = 0.0;
        chroma = 0.0;
        finite = true;
        for (int y = 0; y < img.height(); ++y)
            for (int x = 0; x < img.width(); ++x) {
                const Vec3 c = img.rgb(x, y);
                if (!isFinite(c)) finite = false;
                sum += double(luminance(c));
                const float mean = (c.x + c.y + c.z) / 3.0f;
                chroma += double(std::fabs(c.x - mean) + std::fabs(c.y - mean) + std::fabs(c.z - mean));
            }
        return sum;
    };
    double chromaOff = 0.0, chromaOn = 0.0;
    bool finOff = true, finOn = true;
    const double sumOff = render(0.0f, chromaOff, finOff);
    const double sumOn = render(20.0f, chromaOn, finOn);
    check(finOff && finOn, "dispersion renders are finite");
    check(sumOn > sumOff * 0.8 && sumOn < sumOff * 1.25, "dispersion conserves energy");
    check(chromaOn > chromaOff * 1.5, "dispersion separates RGB channels (rainbow)");
    std::printf("  sumOff=%.1f sumOn=%.1f chromaOff=%.1f chromaOn=%.1f\n", sumOff, sumOn, chromaOff,
                chromaOn);
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
    auto atlas = loadImageOrUdim(root + "/grid.1001.png", QString(), error, {});
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
#if !SOLSTICE_HAVE_MATERIALX
    std::printf("  skip (no MaterialX)\n");
    return;
#else
    // Camera look = muddy glass (rough); caustics transport = sharp glass.
    // Photon / MNEE / LT must pick RayShadeKind::Caustics, not the camera look.
    const QString xml = QStringLiteral(
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

    MaterialXEvalResult eval = evaluateMaterialXDocument(xml, QString());
    check(eval.ok, "ray_switch_shader evaluates");
    if (!eval.ok) {
        std::printf("  error: %s\n", eval.error.toUtf8().constData());
        return;
    }
    check(std::fabs(eval.material.roughness - 0.12f) < 1e-4f, "camera branch roughness 0.12");
    check(eval.material.transmission > 0.99f, "camera branch transmission");
    check(eval.material.raySwitch.caustics == 0, "caustics slot is local index 0");
    check(eval.raySwitchBranches.size() == 1, "one caustics branch material");
    check(eval.raySwitchBranches[0].roughness < 1e-5f, "caustics branch roughness 0");
    check(eval.raySwitchBranches[0].transmission > 0.99f, "caustics branch transmission");

    // Catalog must list Solstice ray_switch nodes even when MaterialX libs load.
    bool foundShader = false, foundColor = false;
    for (const MaterialXNodeCatalogEntry& e : listMaterialXNodeCatalog()) {
        if (e.category == QStringLiteral("ray_switch_shader")) foundShader = true;
        if (e.category == QStringLiteral("ray_switch")) foundColor = true;
    }
    check(foundShader, "catalog contains ray_switch_shader");
    check(foundColor, "catalog contains ray_switch");

    // Stage bake remaps local → scene-absolute indices; materialForRay picks slots.
    Stage stage;
    StagePrim prim;
    prim.type = PrimType::Mesh;
    prim.path = "/glass";
    prim.mesh = makeSphereMesh(0.5f, 16, 8);
    prim.material = eval.material;
    prim.raySwitchBranches = eval.raySwitchBranches;
    prim.materialAssigned = true;
    stage.prims.push_back(prim);

    ScenePtr scene = stage.toScene();
    check(scene && scene->materials.size() >= 2, "scene has base + caustics materials");
    if (!scene) return;
    const int baseIdx = scene->instances.empty() ? -1 : scene->instances[0].materialIndex;
    check(baseIdx >= 0, "instance material index");
    const Material& base = scene->materials[size_t(baseIdx)];
    check(base.raySwitch.caustics >= 0, "baked caustics slot is absolute");
    check(std::fabs(base.roughness - 0.12f) < 1e-4f, "baked camera roughness");

    SceneView view = scene->view();
    const Material cam = materialForRay(view, baseIdx, RayShadeKind::Camera);
    const Material specT = materialForRay(view, baseIdx, RayShadeKind::SpecularTransmission);
    const Material cau = materialForCausticTransport(view, baseIdx);
    check(std::fabs(cam.roughness - 0.12f) < 1e-4f, "Camera ray → camera port roughness 0.12");
    // Unconnected specular_transmission falls back to camera/base (Arnold-like).
    check(std::fabs(specT.roughness - 0.12f) < 1e-4f, "unconnected specular_transmission → camera");
    check(cau.roughness < 1e-5f, "caustic transport → caustics port roughness 0");
    check(std::fabs(cam.roughness - cau.roughness) > 0.05f,
          "camera port must not equal caustics port");
    std::printf("  cameraR=%.3f specTransR=%.3f causticTransportR=%.3f slot=%d\n", cam.roughness,
                specT.roughness, cau.roughness, base.raySwitch.caustics);
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
                                 {1001, 1002, 1003, 1011, 1012, 1013});
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
    check(loadImage(txPath.toStdString(), image, error), "load .tx with mips");
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
        // SSS should look different from pure diffuse base (softens / tints).
        check(std::fabs(pt - diffuseOnly) / std::max(pt, diffuseOnly) > 0.02,
              "SSS changes energy vs diffuse-only");
        check(ratio > 0.45 && ratio < 2.2, "BDPT SSS energy ~ PT SSS");
        std::printf("  sss PT=%.1f BDPT=%.1f diffuse=%.1f ratio=%.3f\n", pt, bdpt, diffuseOnly, ratio);
    }
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

int main() {
    std::printf("Solstice tests\n");
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
    if (getenv("SOL_ONLY_TESS")) {
        registerBuiltinNodes();
        testTessellationTriangleBudget();
        testFrustumCullCloseUpSubdiv();
        testFrustumLocalDicingFalloff();
        testFrustumLocalFullyInViewFast();
        testFrustumLocalItersNotClampedOnDenseCage();
        testScreenAdaptiveTessellation();
        testScreenAdaptiveQualityCoarse();
        testScreenAdaptiveNearDensityDip();
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    testMath();
    testSampling();
    testBsdf();
    testGlob();
    testGraphCook();
    testCameraDofFocus();
    testPolyOpticsApertureSpread();
    testPolynomialOpticsCamera();
    testEnvironment();
    testRender();
    testCausticsGlassSphere();
    testBdptCausticThroughRefraction();
    testPhotonCaustics();
    testRoughGlassCaustics();
    testRefractionSparkleClamp();
    testSplatAccumulationPrecision();
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
    testScreenAdaptiveNearDensityDip();
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
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
