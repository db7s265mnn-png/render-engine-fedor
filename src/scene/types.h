// Plain-old-data scene description shared by the Embree and OptiX backends.
// No STL containers here: the very same structs are uploaded to the GPU.
#pragma once

#include "core/math.h"

namespace sol {

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------
struct Material {
    Vec3 baseColor{0.8f, 0.8f, 0.8f};
    float roughness = 0.4f;

    float metallic = 0.0f;
    float ior = 1.5f;
    float specular = 0.5f;       // dielectric specular level (0..1 -> F0 0..0.08)
    float transmission = 0.0f;   // 0 = opaque, 1 = glass

    Vec3 emissionColor{0.0f, 0.0f, 0.0f};
    float emissionStrength = 0.0f;

    float opacity = 1.0f;
    float subsurface = 0.0f;     // 0 = opaque BRDF, 1 = full random-walk SSS
    int doubleSided = 1;
    int pad0 = 0;

    Vec3 subsurfaceColor{1.0f, 0.75f, 0.55f};
    float subsurfaceScale = 0.15f;  // world-space mean free path scale

    Vec3 subsurfaceRadius{1.0f, 0.35f, 0.2f};  // relative RGB scattering radii
    float pad1 = 0.0f;

    // Indices into SceneView::textures (-1 = none).
    int baseColorTex = -1;
    int roughnessTex = -1;
    int metallicTex = -1;
    int opacityTex = -1;

    int emissionTex = -1;
    int normalTex = -1;
    int subsurfaceTex = -1;
    int pad2 = 0;
};

// RGBA32F texture view shared by CPU / GPU shading.
// UDIM sets are baked into a single atlas covering UV [0,udimGridU]×[0,udimGridV]
// (Houdini/MaterialX <UDIM> → Mari index 1001 + U + V*10).
struct TextureView {
    const float* pixels = nullptr;
    int width = 0;
    int height = 0;
    int udimGridU = 0;  // 0 = regular texture
    int udimGridV = 0;

    SR_HD bool valid() const { return pixels != nullptr && width > 0 && height > 0; }
    SR_HD bool isUdimAtlas() const { return udimGridU > 0 && udimGridV > 0; }
};

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
// A view onto triangle mesh data. The pointers live either in host memory
// (Embree backend) or in device memory (OptiX backend).
struct MeshView {
    const Vec3* positions = nullptr;
    const Vec3* normals = nullptr;   // may be null -> geometric normals used
    const Vec2* uvs = nullptr;       // may be null
    const uint32_t* indices = nullptr;  // 3 per triangle
    uint32_t triangleCount = 0;
    uint32_t vertexCount = 0;
};

// Visibility bits for primary vs shadow rays (Embree mask / OptiX visibilityMask).
constexpr int kVisShadow = 0x1;
constexpr int kVisPrimary = 0x2;
constexpr int kVisAll = kVisShadow | kVisPrimary;

struct InstanceData {
    Mat4 xform;      // object -> world
    Mat4 xformInv;   // world -> object
    int meshIndex = -1;
    int materialIndex = -1;
    int lightIndex = -1;   // >= 0 when this instance is an area light proxy
    int visibleCamera = 1;
    // Embree/OptiX visibility: shadow rays use kVisShadow only so area-light
    // proxies can opt out of casting self-shadows via kVisPrimary alone.
    int visibilityMask = kVisAll;
};

// ---------------------------------------------------------------------------
// Lights
// ---------------------------------------------------------------------------
enum LightType : int {
    kLightDistant = 0,  // sun, direction = -Z of the light transform
    kLightRect = 1,     // XY rectangle, emits along -Z
    kLightDisk = 2,     // XY disk, emits along -Z
    kLightSphere = 3,   // sphere of given radius
    kLightDome = 4,     // environment / HDR
    kLightPoint = 5,    // tiny sphere, analytic
};

// Sampled environment map (equirectangular) with its 2D CDF tables.
struct EnvMapView {
    const float* pixels = nullptr;  // RGBA32F, width*height*4
    int width = 0;
    int height = 0;
    const float* condCdf = nullptr;         // (width+1) * height
    const float* condIntegral = nullptr;    // height
    const float* margCdf = nullptr;         // height + 1
    const float* margFunc = nullptr;        // height
    const float* func = nullptr;            // width * height
    float integral = 0.0f;

    SR_HD bool valid() const { return pixels != nullptr && width > 0 && height > 0; }
    SR_HD bool sampled() const { return valid() && func != nullptr && integral > 0.0f; }
};

struct LightData {
    Mat4 xform;      // light -> world
    Mat4 xformInv;   // world -> light

    int type = kLightDistant;
    float intensity = 1.0f;
    float exposure = 0.0f;
    float pad0 = 0.0f;

    Vec3 color{1.0f, 1.0f, 1.0f};
    float width = 1.0f;   // rect width

    float height = 1.0f;  // rect height
    float radius = 0.5f;  // disk / sphere radius
    float angle = 0.53f;  // distant light angular diameter in degrees
    float spread = 1.0f;  // reserved

    int normalize = 1;       // divide by light area so size does not change exposure
    int twoSided = 0;        // rect / disk emit on both sides
    int visibleCamera = 1;   // primary rays see the light
    int envIndex = -1;       // index into the scene environment map table

    int shadowEnable = 1;
    // When 0, the light's proxy geometry is skipped by shadow rays (no self-shadow).
    int selfShadowEnable = 0;
    int samples = 1;
    int pad1 = 0;

    SR_HD Vec3 emittedRadiance() const { return color * (intensity * exp2f(exposure)); }
};

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
struct CameraData {
    Mat4 cameraToWorld;
    float focalLength = 50.0f;    // mm
    float sensorWidth = 36.0f;    // mm (horizontal aperture)
    float fStop = 0.0f;           // 0 disables depth of field
    float focusDistance = 5.0f;

    float shutterOpen = 0.0f;
    float shutterClose = 0.0f;
    float nearClip = 0.001f;
    float farClip = 1e7f;
};

// ---------------------------------------------------------------------------
// Render settings
// ---------------------------------------------------------------------------
enum ToneMapper : int { kToneNone = 0, kToneReinhard = 1, kToneAces = 2 };
enum RenderBackendType : int { kBackendCpuEmbree = 0, kBackendGpuOptix = 1 };
enum IntegratorMode : int { kIntegratorPathTracer = 0, kIntegratorDirectLighting = 1, kIntegratorAmbientOcclusion = 2 };

struct RenderSettingsData {
    int resolutionX = 960;
    int resolutionY = 540;
    int samplesPerPixel = 64;
    int maxDepth = 8;

    int rrStartDepth = 3;
    int lightSamples = 1;
    int seed = 0;
    int integrator = kIntegratorPathTracer;

    float clampIndirect = 20.0f;   // <= 0 disables firefly clamping
    float exposure = 0.0f;
    float gamma = 2.2f;
    int toneMapper = kToneAces;

    int backend = kBackendCpuEmbree;
    int envVisibleCamera = 1;
    int tileSize = 32;
    int threads = 0;               // 0 = hardware concurrency

    float aoDistance = 1.0f;
    float pad1 = 0.0f;
    float pad2 = 0.0f;
    float pad3 = 0.0f;
};

// ---------------------------------------------------------------------------
// The flattened scene as seen by an integrator.
// ---------------------------------------------------------------------------
struct SceneView {
    const MeshView* meshes = nullptr;
    const InstanceData* instances = nullptr;
    const Material* materials = nullptr;
    const LightData* lights = nullptr;
    const EnvMapView* envMaps = nullptr;
    const TextureView* textures = nullptr;

    int meshCount = 0;
    int instanceCount = 0;
    int materialCount = 0;
    int lightCount = 0;
    int envMapCount = 0;
    int textureCount = 0;
    int domeLightIndex = -1;  // first dome light, used for ray misses

    CameraData camera;
    RenderSettingsData settings;
    Bounds3 worldBounds;
};

}  // namespace sol
