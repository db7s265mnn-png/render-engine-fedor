#include "app/default_scene.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QVector3D>

#include "io/alembic_loader.h"
#include "io/materialx_graph.h"
#include "nodes/node_registry.h"

namespace sol {
namespace {

void place(Node* node, double x, double y) {
    if (node) node->setPosition(QPointF(x, y));
}

QString findBundledAsset(const QString& fileName) {
    QStringList roots = {
        QDir::currentPath() + "/examples",
        QDir::currentPath() + "/assets",
        QDir::currentPath() + "/../examples",
        QDir::currentPath() + "/../assets",
    };
    if (QCoreApplication::instance()) {
        const QString appDir = QCoreApplication::applicationDirPath();
        roots.prepend(appDir + "/../assets");
        roots.prepend(appDir + "/../examples");
        roots.prepend(appDir + "/assets");
        roots.prepend(appDir + "/examples");
    }
    for (const QString& root : roots) {
        const QFileInfo info(QDir(root).absoluteFilePath(fileName));
        if (info.exists() && info.isFile()) return info.absoluteFilePath();
    }
    return {};
}

}  // namespace

void buildDefaultGraph(NodeGraph& graph) {
    registerBuiltinNodes();
    graph.clear();

    const QString buddhaPath =
        alembicSupportAvailable() ? findBundledAsset("buddha.abc") : QString();
    const QString hdriPath = findBundledAsset("ferndale_studio_07_2k.hdr");

    // 3×3 hero grid (world spacing between instance centres).
    constexpr float kGridSpacing = 2.5f;
    QList<Node*> heroes;
    heroes.reserve(9);
    for (int i = 0; i < 9; ++i) {
        const int col = i % 3;
        const int row = i / 3;
        const float tx = (col - 1) * kGridSpacing;
        const float tz = (row - 1) * kGridSpacing;
        const double uiX = -200.0 + col * 140.0;
        const double uiY = -560.0 + row * 80.0;

        Node* hero = nullptr;
        if (!buddhaPath.isEmpty()) {
            hero = graph.createNode("alembic", QStringLiteral("buddha%1").arg(i + 1));
            if (hero) {
                hero->setParameterValue("file", buddhaPath);
                hero->setParameterValue("primpath", QStringLiteral("/geo/buddha%1").arg(i + 1));
                hero->setParameterValue("translate", QVariant::fromValue(QVector3D(tx, 0.0f, tz)));
                hero->setParameterValue("scale", QVariant::fromValue(QVector3D(1.0f, 1.0f, 1.0f)));
            }
        } else {
            hero = graph.createNode("sphere", QStringLiteral("sphere%1").arg(i + 1));
            if (hero) {
                hero->setParameterValue("radius", 1.0);
                hero->setParameterValue("primname", QStringLiteral("sphere%1").arg(i + 1));
                hero->setParameterValue("translate", QVariant::fromValue(QVector3D(tx, 1.0f, tz)));
            }
        }
        place(hero, uiX, uiY);
        heroes.append(hero);
    }

    Node* ground = graph.createNode("grid", "ground1");
    place(ground, -480, -480);
    if (ground) {
        ground->setParameterValue("sizex", 40.0);
        ground->setParameterValue("sizez", 40.0);
        ground->setParameterValue("primname", "ground");
    }

    Node* groundMaterial = graph.createNode("material", "groundmat1");
    place(groundMaterial, -480, -380);
    if (groundMaterial) {
        groundMaterial->setParameterValue("pattern", "/geo/ground");
        groundMaterial->setParameterValue(
            "mtlx",
            QStringLiteral(
                "<?xml version=\"1.0\"?>\n"
                "<materialx version=\"1.39\">\n"
                "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\">\n"
                "    <input name=\"base_color\" type=\"color3\" value=\"0.55, 0.55, 0.58\" />\n"
                "    <input name=\"specular_roughness\" type=\"float\" value=\"0.6\" />\n"
                "    <input name=\"metalness\" type=\"float\" value=\"0\" />\n"
                "  </standard_surface>\n"
                "  <surfacematerial name=\"surface\" type=\"material\">\n"
                "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\" />\n"
                "  </surfacematerial>\n"
                "</materialx>\n"));
    }

    // Merge tree (merge has 4 inputs): 4 + 4 + 1 → heromat.
    Node* mergeA = graph.createNode("merge", "merge_heroes_a");
    place(mergeA, -200, -300);
    Node* mergeB = graph.createNode("merge", "merge_heroes_b");
    place(mergeB, -40, -300);
    Node* mergeHeroes = graph.createNode("merge", "merge_heroes");
    place(mergeHeroes, -120, -220);

    for (int i = 0; i < 4 && i < heroes.size(); ++i)
        if (heroes[i]) graph.connectNodes(heroes[i], mergeA, i);
    for (int i = 0; i < 4 && (i + 4) < heroes.size(); ++i)
        if (heroes[i + 4]) graph.connectNodes(heroes[i + 4], mergeB, i);
    if (mergeA) graph.connectNodes(mergeA, mergeHeroes, 0);
    if (mergeB) graph.connectNodes(mergeB, mergeHeroes, 1);
    if (heroes.size() > 8 && heroes[8]) graph.connectNodes(heroes[8], mergeHeroes, 2);

    Node* heroMaterial = graph.createNode("material", "heromat1");
    place(heroMaterial, -120, -140);
    if (heroMaterial) {
        heroMaterial->setParameterValue("pattern", "*");
        // MaterialX graph (Solaris-style): standard_surface → surfacematerial "surface".
        QString mtlx = createDefaultMaterialXDocument();
        if (mtlx.isEmpty()) {
            mtlx = QStringLiteral(
                "<?xml version=\"1.0\"?>\n"
                "<materialx version=\"1.39\">\n"
                "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\">\n"
                "    <input name=\"base_color\" type=\"color3\" value=\"0.82, 0.72, 0.58\" />\n"
                "    <input name=\"specular_roughness\" type=\"float\" value=\"0.35\" />\n"
                "    <input name=\"subsurface\" type=\"float\" value=\"0.35\" />\n"
                "    <input name=\"subsurface_color\" type=\"color3\" value=\"0.9, 0.55, 0.35\" />\n"
                "  </standard_surface>\n"
                "  <surfacematerial name=\"surface\" type=\"material\">\n"
                "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\" />\n"
                "  </surfacematerial>\n"
                "</materialx>\n");
        } else {
            mtlx.replace("0.8, 0.8, 0.8", "0.82, 0.72, 0.58");
            mtlx.replace("name=\"subsurface\" type=\"float\" value=\"0\"",
                         "name=\"subsurface\" type=\"float\" value=\"0.35\"");
            mtlx.replace("name=\"subsurface_color\" type=\"color3\" value=\"1, 0.75, 0.55\"",
                         "name=\"subsurface_color\" type=\"color3\" value=\"0.9, 0.55, 0.35\"");
        }
        heroMaterial->setParameterValue("mtlx", mtlx);
    }

    Node* merge = graph.createNode("merge", "merge1");
    place(merge, -180, -60);

    Node* dome = graph.createNode("domelight", "domelight1");
    place(dome, -180, 20);
    if (dome) {
        dome->setParameterValue("intensity", hdriPath.isEmpty() ? 0.6 : 1.0);
        if (!hdriPath.isEmpty()) dome->setParameterValue("texture", hdriPath);
    }

    Node* sun = graph.createNode("distantlight", "sunlight1");
    place(sun, -180, 100);
    if (sun) {
        sun->setParameterValue("enabled", true);
        sun->setParameterValue("intensity", hdriPath.isEmpty() ? 2.5 : 0.35);
        sun->setParameterValue("angle", 1.5);
        sun->setParameterValue("rotate", QVariant::fromValue(QVector3D(-42.0f, -35.0f, 0.0f)));
    }

    Node* key = graph.createNode("rectlight", "rectlight1");
    place(key, -180, 180);
    if (key) {
        key->setParameterValue("enabled", true);
        key->setParameterValue("width", 4.0);
        key->setParameterValue("height", 3.0);
        key->setParameterValue("intensity", hdriPath.isEmpty() ? 24.0 : 4.0);
        key->setParameterValue("translate", QVariant::fromValue(QVector3D(-5.0f, 6.0f, 5.0f)));
        key->setParameterValue("rotate", QVariant::fromValue(QVector3D(-40.0f, -35.0f, 0.0f)));
    }

    Node* camera = graph.createNode("camera", "camera1");
    place(camera, -180, 260);
    if (camera) {
        camera->setParameterValue("eye", QVariant::fromValue(QVector3D(8.0f, 4.0f, 10.0f)));
        camera->setParameterValue("target", QVariant::fromValue(QVector3D(0.0f, 0.9f, 0.0f)));
        camera->setParameterValue("focal", 50.0);
    }

    Node* settings = graph.createNode("rendersettings", "rendersettings1");
    place(settings, -180, 340);

    graph.connectNodes(ground, groundMaterial, 0);
    graph.connectNodes(mergeHeroes, heroMaterial, 0);
    graph.connectNodes(groundMaterial, merge, 0);
    graph.connectNodes(heroMaterial, merge, 1);
    graph.connectNodes(merge, dome, 0);
    graph.connectNodes(dome, sun, 0);
    graph.connectNodes(sun, key, 0);
    graph.connectNodes(key, camera, 0);
    graph.connectNodes(camera, settings, 0);
    graph.setDisplayNode(settings);
    graph.setModified(false);
}

void buildAlembicGraph(NodeGraph& graph, const QString& alembicPath, const QString& hdriPath) {
    registerBuiltinNodes();
    graph.clear();

    Node* alembic = graph.createNode("alembic", "alembic1");
    place(alembic, -180, -420);
    if (alembic) alembic->setParameterValue("file", alembicPath);

    Node* material = graph.createNode("material", "material1");
    place(material, -180, -320);
    if (material) {
        material->setParameterValue("pattern", "*");
        material->setParameterValue("roughness", 0.4);
    }

    Node* dome = graph.createNode("domelight", "domelight1");
    place(dome, -180, -220);
    if (dome && !hdriPath.isEmpty()) dome->setParameterValue("texture", hdriPath);

    Node* sun = graph.createNode("distantlight", "sunlight1");
    place(sun, -180, -140);

    // No camera node on purpose: without one the renderer frames the imported
    // geometry automatically, which is what you want right after an import.
    Node* settings = graph.createNode("rendersettings", "rendersettings1");
    place(settings, -180, -60);

    graph.connectNodes(alembic, material, 0);
    graph.connectNodes(material, dome, 0);
    graph.connectNodes(dome, sun, 0);
    graph.connectNodes(sun, settings, 0);
    graph.setDisplayNode(settings);
    graph.setModified(false);
}

}  // namespace sol
