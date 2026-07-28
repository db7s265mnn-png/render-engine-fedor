// Unit tests for the maths, sampling, node graph and renderer plumbing.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <QColor>
#include <QDir>
#include <QImage>
#include <QTemporaryDir>
#include <QVector3D>

#include "app/default_scene.h"
#include "core/image.h"
#include "core/rng.h"
#include "io/alembic_loader.h"
#include "io/image_io.h"
#include "io/materialx_graph.h"
#include "nodes/node_graph.h"
#include "nodes/node_registry.h"
#include "nodes/stage.h"
#include "render/integrator.h"
#include "render/render_session.h"
#include "render/shading.h"
#include "scene/scene.h"
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
        "    <input name=\"filex\" type=\"filename\" value=\"\"/>\n"
        "    <input name=\"filey\" type=\"filename\" value=\"\"/>\n"
        "    <input name=\"filez\" type=\"filename\" value=\"\"/>\n"
        "    <input name=\"default\" type=\"color3\" value=\"0.2, 0.5, 0.8\"/>\n"
        "    <input name=\"blend\" type=\"float\" value=\"1\"/>\n"
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
    std::printf("  triplanar ok=%d err=%s proc=%d\n", int(eval.ok), eval.error.toStdString().c_str(),
                eval.material.baseColorProc);

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

    // unifiednoise3d with high frequency must compile (no spinbox clamp on authoring).
    const QString unified = QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <unifiednoise3d name=\"u1\" type=\"float\">\n"
        "    <input name=\"freq\" type=\"vector3\" value=\"64, 64, 64\"/>\n"
        "    <input name=\"offset\" type=\"vector3\" value=\"0, 0, 0\"/>\n"
        "    <input name=\"type\" type=\"integer\" value=\"0\"/>\n"
        "  </unifiednoise3d>\n"
        "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" nodename=\"u1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
    eval = evaluateMaterialXDocument(unified, QString());
    check(eval.ok, "evaluate unifiednoise3d freq=64 ok");
    check(eval.material.baseColorProc >= 0, "unifiednoise3d compiles");
    if (eval.material.baseColorProc >= 0 && size_t(eval.material.baseColorProc) < eval.procedurals.size()) {
        const ProceduralNode& node = eval.procedurals[size_t(eval.material.baseColorProc)];
        check(node.op == kProcUnified3d, "unifiednoise3d opcode");
        check(std::fabs(node.p0.x - 64.0f) < 1e-4f, "freq 64 preserved (no clamp)");
    }
    std::printf("  noise3d ok=%d proc=%d unified freq=%g\n", int(eval.ok), eval.material.baseColorProc,
                eval.material.baseColorProc >= 0 && size_t(eval.material.baseColorProc) < eval.procedurals.size()
                    ? double(eval.procedurals[size_t(eval.material.baseColorProc)].p0.x)
                    : 0.0);
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

}  // namespace

int main() {
    std::printf("Solstice tests\n");
    testMath();
    testSampling();
    testBsdf();
    testGlob();
    testGraphCook();
    testCameraDofFocus();
    testEnvironment();
    testRender();
    testInstanceTransform();
    testUdimMaterialX();
    testTxMipmaps();
    testMaterialXTypeMismatchConnect();
    testMaterialXNoiseAndTriplanar();
    testMaterialXUdimCubeAsset();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
