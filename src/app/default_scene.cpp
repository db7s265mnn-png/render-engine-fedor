#include "app/default_scene.h"

#include <QVector3D>

#include "nodes/node_registry.h"

namespace sol {
namespace {

void place(Node* node, double x, double y) {
    if (node) node->setPosition(QPointF(x, y));
}

}  // namespace

void buildDefaultGraph(NodeGraph& graph) {
    registerBuiltinNodes();
    graph.clear();

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

    Node* sphere = graph.createNode("sphere", "sphere1");
    place(sphere, -40, -420);
    if (sphere) {
        sphere->setParameterValue("radius", 1.2);
        sphere->setParameterValue("primname", "sphere");
        sphere->setParameterValue("translate", QVariant::fromValue(QVector3D(0.0f, 1.2f, 0.0f)));
    }

    Node* sphereMaterial = graph.createNode("material", "spheremat1");
    place(sphereMaterial, -40, -320);
    if (sphereMaterial) {
        sphereMaterial->setParameterValue("pattern", "/geo/sphere");
        sphereMaterial->setParameterValue("basecolor", QVariant::fromValue(QVector3D(0.85f, 0.35f, 0.2f)));
        sphereMaterial->setParameterValue("roughness", 0.25);
        sphereMaterial->setParameterValue("metallic", 0.0);
    }

    Node* merge = graph.createNode("merge", "merge1");
    place(merge, -180, -200);

    Node* dome = graph.createNode("domelight", "domelight1");
    place(dome, -180, -100);
    if (dome) dome->setParameterValue("intensity", 0.6);

    Node* sun = graph.createNode("distantlight", "sunlight1");
    place(sun, -180, -20);
    if (sun) {
        sun->setParameterValue("intensity", 2.5);
        sun->setParameterValue("angle", 1.5);
        sun->setParameterValue("rotate", QVariant::fromValue(QVector3D(-42.0f, -35.0f, 0.0f)));
    }

    Node* key = graph.createNode("rectlight", "rectlight1");
    place(key, -180, 60);
    if (key) {
        key->setParameterValue("width", 4.0);
        key->setParameterValue("height", 3.0);
        key->setParameterValue("intensity", 24.0);
        key->setParameterValue("translate", QVariant::fromValue(QVector3D(-3.5f, 4.5f, 3.0f)));
        key->setParameterValue("rotate", QVariant::fromValue(QVector3D(-40.0f, -35.0f, 0.0f)));
    }

    Node* camera = graph.createNode("camera", "camera1");
    place(camera, -180, 140);
    if (camera) {
        camera->setParameterValue("eye", QVariant::fromValue(QVector3D(5.5f, 3.2f, 7.5f)));
        camera->setParameterValue("target", QVariant::fromValue(QVector3D(0.0f, 1.1f, 0.0f)));
        camera->setParameterValue("focal", 45.0);
    }

    Node* settings = graph.createNode("rendersettings", "rendersettings1");
    place(settings, -180, 220);

    graph.connectNodes(ground, groundMaterial, 0);
    graph.connectNodes(sphere, sphereMaterial, 0);
    graph.connectNodes(groundMaterial, merge, 0);
    graph.connectNodes(sphereMaterial, merge, 1);
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
