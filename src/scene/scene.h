// Host side scene container. Cooking a node graph produces one of these and
// the render backends consume its SceneView.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/image.h"
#include "scene/types.h"

namespace sol {

struct Mesh {
    std::string name;
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    std::vector<uint32_t> indices;
    Bounds3 bounds;
    // Arnold displacement bounds_padding — reapplied by computeBounds() after validate.
    float boundsPadding = 0.0f;
    // Deformation motion blur: positions for keys 1..N-1 (key 0 is `positions`).
    // Each entry must match `positions.size()` when used.
    std::vector<std::vector<Vec3>> motionPositions;
    // Flat packed copy of all keys for MeshView (key0 = positions, then motionPositions...).
    mutable std::vector<Vec3> motionPositionsPacked_;

    size_t triangleCount() const { return indices.size() / 3; }
    void computeBounds();
    // Area weighted vertex normals; only fills in normals when they are missing.
    void computeNormalsIfMissing();
    void validate();
    MeshView view() const;
};

using MeshPtr = std::shared_ptr<Mesh>;

struct EnvironmentMap {
    std::string path;
    Image image;
    Distribution2D distribution;
    std::vector<float> luminanceFunc;

    void buildSamplingTables();
    EnvMapView view() const;
};

// A named entry in the scene graph tree, mirroring a USD style prim path.
struct PrimRecord {
    std::string path;
    std::string type;      // "Mesh", "RectLight", "Camera", ...
    std::string sourceNode;
    int instanceIndex = -1;
    int lightIndex = -1;
    long long pointCount = 0;
    long long triangleCount = 0;
};

class Scene {
public:
    std::vector<MeshPtr> meshes;
    std::vector<InstanceData> instances;
    std::vector<Material> materials;
    std::vector<LightData> lights;
    std::vector<std::shared_ptr<EnvironmentMap>> envMaps;
    std::vector<std::shared_ptr<Image>> textures;
    std::vector<ProceduralNode> procedurals;
    std::vector<PrimRecord> prims;

    CameraData camera;
    RenderSettingsData settings;
    bool cameraAuthored = false;
    // Timeline scrub / playback: prefer a fast accel rebuild over final quality.
    bool fastRebuild = false;

    // Motion blur storage (filled by attachMotionBlurKeys).
    std::vector<Mat4> motionXforms;
    std::vector<Mat4> cameraMotionXforms;

    // Shutter-center instance transforms for viewport interaction (pick / frame /
    // tumble pivot). Empty when motion blur is off — then use InstanceData::xform.
    // Beauty Embree still consumes motionXforms; this buffer ignores shutter keys.
    std::vector<Mat4> pickXforms;

    int addMesh(MeshPtr mesh);
    int addMaterial(const Material& material);
    int addEnvMap(std::shared_ptr<EnvironmentMap> env);
    int addTexture(std::shared_ptr<Image> image);

    // Generates renderable proxy geometry for area lights, recomputes bounds
    // and prepares the flat arrays consumed by SceneView.
    void finalize();

    // Rebuild MeshView table after mutating mesh motion buffers.
    void refreshMeshViews();

    SceneView view() const;
    Bounds3 bounds() const { return bounds_; }
    bool empty() const { return instances.empty() && lights.empty(); }

    size_t totalTriangles() const;
    void frameCameraOnContents();

private:
    void buildLightProxies();

    Bounds3 bounds_;
    std::vector<MeshView> meshViews_;
    std::vector<EnvMapView> envViews_;
    std::vector<TextureView> textureViews_;
    int domeLightIndex_ = -1;
};

using ScenePtr = std::shared_ptr<Scene>;

// Primitive builders shared by the geometry nodes and by light proxies.
MeshPtr makeSphereMesh(float radius, int segmentsU = 48, int segmentsV = 24);
MeshPtr makeGridMesh(float sizeX, float sizeZ, int divisionsX = 1, int divisionsZ = 1);
MeshPtr makeBoxMesh(Vec3 size);
MeshPtr makeRectMesh(float width, float height);
MeshPtr makeDiskMesh(float radius, int segments = 48);
MeshPtr makeTubeMesh(float radius, float height, int segments = 32);

}  // namespace sol
