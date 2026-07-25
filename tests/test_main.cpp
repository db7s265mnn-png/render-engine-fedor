// Unit tests for the maths, sampling, node graph and renderer plumbing.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <QDir>
#include <QImage>
#include <QTemporaryDir>

#include "app/default_scene.h"
#include "core/image.h"
#include "core/rng.h"
#include "io/alembic_loader.h"
#include "io/image_io.h"
#include "io/materialx_graph.h"
#include "nodes/node_graph.h"
#include "nodes/node_registry.h"
#include "render/integrator.h"
#include "render/render_session.h"
#include "render/shading.h"

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
    check(stage->countOfType(PrimType::Mesh) == 2, "two meshes in the default stage");
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
        // 1001 → red at UV tile (0,0); 1002 → green at (1,0); 1011 → blue at (0,1)
        checkNear(atlas->at(0, 0).x, 1.0f, 0.05f, "tile 1001 baked into atlas origin");
        checkNear(atlas->at(4, 0).y, 1.0f, 0.05f, "tile 1002 baked into U=1");
        checkNear(atlas->at(0, 4).z, 1.0f, 0.05f, "tile 1011 baked into V=1");
    }

    if (!materialXAvailable()) {
        std::printf("  skip MaterialX xml roundtrip (MaterialX unavailable)\n");
        return;
    }

    QVector<MaterialXGraphNode> nodes;
    MaterialXGraphNode image;
    image.name = "image_color";
    image.category = "image";
    image.type = "color3";
    image.inputs.push_back({"file", "filename", pattern, {}});
    nodes.push_back(image);
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
}

}  // namespace

int main() {
    std::printf("Solstice tests\n");
    testMath();
    testSampling();
    testBsdf();
    testGlob();
    testGraphCook();
    testEnvironment();
    testRender();
    testInstanceTransform();
    testUdimMaterialX();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
