#include "app/default_scene.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QVector3D>

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

    const QString buddhaPath = findBundledAsset("buddha.abc");
    const QString hdriPath = findBundledAsset("ferndale_studio_07_2k.hdr");

    Node* ground = graph.createNode("grid", "ground1");
    place(ground, -320, -420);
    if (ground) {
        ground->setParameterValue("sizex", 24.0);
        ground->setParameterValue("sizez", 24.0);
        ground->setParameterValue("primname", "ground");
    }

    Node* groundMaterial = graph.createNode("material", "groundmat1");
    place(groundMaterial, -320, -320);
    if (groundMaterial) {
        groundMaterial->setParameterValue("pattern", "/geo/ground");
        groundMaterial->setParameterValue("basecolor", QVariant::fromValue(QVector3D(0.55f, 0.55f, 0.58f)));
        groundMaterial->setParameterValue("roughness", 0.6);
    }

    Node* hero = nullptr;
    if (!buddhaPath.isEmpty()) {
        hero = graph.createNode("alembic", "buddha1");
        place(hero, -40, -420);
        if (hero) {
            hero->setParameterValue("file", buddhaPath);
            hero->setParameterValue("translate", QVariant::fromValue(QVector3D(0.0f, 0.0f, 0.0f)));
            hero->setParameterValue("scale", QVariant::fromValue(QVector3D(1.0f, 1.0f, 1.0f)));
        }
    } else {
        hero = graph.createNode("sphere", "sphere1");
        place(hero, -40, -420);
        if (hero) {
            hero->setParameterValue("radius", 1.2);
            hero->setParameterValue("primname", "sphere");
            hero->setParameterValue("translate", QVariant::fromValue(QVector3D(0.0f, 1.2f, 0.0f)));
        }
    }

    Node* heroMaterial = graph.createNode("material", "heromat1");
    place(heroMaterial, -40, -320);
    if (heroMaterial) {
        heroMaterial->setParameterValue("pattern", "*");
        heroMaterial->setParameterValue("basecolor", QVariant::fromValue(QVector3D(0.82f, 0.72f, 0.58f)));
        heroMaterial->setParameterValue("roughness", 0.35);
        heroMaterial->setParameterValue("subsurface", 0.35);
        heroMaterial->setParameterValue("subsurface_color", QVariant::fromValue(QVector3D(0.9f, 0.55f, 0.35f)));
    }

    Node* merge = graph.createNode("merge", "merge1");
    place(merge, -180, -200);

    Node* dome = graph.createNode("domelight", "domelight1");
    place(dome, -180, -100);
    if (dome) {
        dome->setParameterValue("intensity", hdriPath.isEmpty() ? 0.6 : 1.0);
        if (!hdriPath.isEmpty()) dome->setParameterValue("texture", hdriPath);
    }

    Node* sun = graph.createNode("distantlight", "sunlight1");
    place(sun, -180, -20);
    if (sun) {
        sun->setParameterValue("enabled", true);
        sun->setParameterValue("intensity", hdriPath.isEmpty() ? 2.5 : 0.35);
        sun->setParameterValue("angle", 1.5);
        sun->setParameterValue("rotate", QVariant::fromValue(QVector3D(-42.0f, -35.0f, 0.0f)));
    }

    Node* key = graph.createNode("rectlight", "rectlight1");
    place(key, -180, 60);
    if (key) {
        key->setParameterValue("enabled", true);
        key->setParameterValue("width", 4.0);
        key->setParameterValue("height", 3.0);
        key->setParameterValue("intensity", hdriPath.isEmpty() ? 24.0 : 4.0);
        key->setParameterValue("translate", QVariant::fromValue(QVector3D(-3.5f, 4.5f, 3.0f)));
        key->setParameterValue("rotate", QVariant::fromValue(QVector3D(-40.0f, -35.0f, 0.0f)));
    }

    Node* camera = graph.createNode("camera", "camera1");
    place(camera, -180, 140);
    if (camera) {
        camera->setParameterValue("eye", QVariant::fromValue(QVector3D(4.5f, 2.4f, 5.5f)));
        camera->setParameterValue("target", QVariant::fromValue(QVector3D(0.0f, 1.0f, 0.0f)));
        camera->setParameterValue("focal", 50.0);
    }

    Node* settings = graph.createNode("rendersettings", "rendersettings1");
    place(settings, -180, 220);

    graph.connectNodes(ground, groundMaterial, 0);
    graph.connectNodes(hero, heroMaterial, 0);
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
