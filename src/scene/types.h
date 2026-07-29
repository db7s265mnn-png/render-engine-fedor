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
    // Mix weight vs diffuse (Standard Surface): base_mix = (1-w)*diffuse + w*SSS.
    float subsurface = 0.0f;
    int doubleSided = 1;
    // Allow specular→diffuse light transport (reflective caustics).
    int reflectiveCaustics = 1;

    Vec3 subsurfaceColor{1.0f, 0.75f, 0.55f};
    // Arnold: MFP (scene units / metres) = subsurfaceScale * subsurfaceRadius.
    // Scale = 1 means Radius is already in metres (Houdini MKS).
    // Radius is RGB: each channel has its own MFP (spectral random-walk / hero channel).
    float subsurfaceScale = 1.0f;

    Vec3 subsurfaceRadius{1.0f, 0.35f, 0.2f};  // RGB mean free path (skin: R > G > B)
    // Standard Surface `base` weight (multiplies diffuse only). MaterialX default is 0.8;
    // 1.0 keeps non-MaterialX materials unchanged until `base` is authored.
    float baseWeight = 1.0f;

    // Indices into SceneView::textures (-1 = none).
    int baseColorTex = -1;
    int roughnessTex = -1;
    int metallicTex = -1;
    int opacityTex = -1;

    int emissionTex = -1;
    int normalTex = -1;
    int subsurfaceTex = -1;
    // Indices into SceneView::procedurals (-1 = none). Procedurals win over textures.
    int baseColorProc = -1;

    int roughnessProc = -1;
    int metallicProc = -1;
    int opacityProc = -1;
    int emissionProc = -1;

    int normalProc = -1;
    int subsurfaceProc = -1;
    // Fake-caustics style shadow control (MaterialX): 1 = fully opaque shadow, 0 = none.
    // Only applies to transmissive surfaces when refractiveCaustics is on.
    float shadowOpacity = 1.0f;
    // Allow transmission→... light transport / transparent shadows (refractive caustics).
    int refractiveCaustics = 1;

    // MaterialX normalmap.scale / bump.scale (tangent XY strength).
    float normalScale = 1.0f;
    // Height-field bump (MaterialX <bump>). When set, wins over normalTex/normalProc.
    int bumpTex = -1;
    int bumpProc = -1;
    int _padBump = 0;
};

// Shade-time MaterialX procedural opcode (see render/procedural.h for evaluation).
enum ProceduralOp : int {
    kProcConst = 0,
    kProcUv = 1,
    kProcPosition = 2,
    kProcNormal = 3,
    kProcNoise2d = 4,
    kProcNoise3d = 5,
    kProcFractal = 6,
    kProcCell2d = 7,
    kProcCell3d = 8,
    kProcImage = 9,
    kProcTriplanar = 10,
    kProcMul = 11,
    kProcAdd = 12,
    kProcSub = 13,
    kProcDiv = 14,
    kProcMix = 15,
    kProcClamp = 16,
    kProcSaturate = 17,
    kProcInvert = 18,
    kProcAbs = 19,
    kProcPower = 20,
    kProcCombine = 21,
    kProcExtract = 22,
    kProcRampLR = 23,
    kProcRampTB = 24,
    kProcChecker = 25,
    kProcUnified2d = 26,
    kProcUnified3d = 27,
    // MaterialX place2d / rotate2d: UV = rotate((uv-pivot)*scale)+pivot+offset
    // in0=texcoord child, p0=scale.xy, p1=offset.xy, p2=pivot.xy, s0=rotate degrees
    kProcPlace2d = 28,
};

struct ProceduralNode {
    int op = kProcConst;
    int channels = 3;
    int in0 = -1;
    int in1 = -1;
    int in2 = -1;
    int in3 = -1;
    Vec4 p0{0.0f, 0.0f, 0.0f, 1.0f};
    Vec4 p1{0.0f, 0.0f, 0.0f, 0.0f};
    Vec4 p2{0.0f, 1.0f, 0.0f, 0.0f};
    float s0 = 0.0f;
    float s1 = 2.0f;
    float s2 = 0.5f;
    float s3 = 1.0f;
};

// RGBA32F texture view shared by CPU / GPU shading.
// UDIM sets are baked into a single atlas covering UV [0,udimGridU]×[0,udimGridV]
// (Houdini/MaterialX <UDIM> → Mari index 1001 + U + V*10).
// Optional mip pyramid is packed contiguously after level 0 (maketx /.tx).
struct TextureView {
    const float* pixels = nullptr;
    int width = 0;
    int height = 0;
    int udimGridU = 0;  // 0 = regular texture
    int udimGridV = 0;
    int mipCount = 1;   // 1 = level 0 only
    int padMip = 0;

    SR_HD bool valid() const { return pixels != nullptr && width > 0 && height > 0; }
    SR_HD bool isUdimAtlas() const { return udimGridU > 0 && udimGridV > 0; }
    SR_HD bool hasMips() const { return mipCount > 1; }
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

    // 0 = thin lens (default), 1 = polynomial optics (Embree only; OptiX falls back).
    int opticalModel = 0;
    // Index into polynomialOpticsLensNames() when opticalModel == 1.
    int lensModel = 19;  // cooke__speed_panchro__1920__50mm
    float opticalWavelengthNm = 550.0f;
    // When set with polynomial optics: each camera sample picks R, G or B wavelength
    // so longitudinal/lateral chromatic aberration appears as coloured fringing.
    int chromaticAberration = 0;

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
// Caustics solver used when Integrator = Path Tracer (CPU / Embree).
// 0 = BDPT + OpenPGL guiding (D+A), 1 = MNEE / manifold next-event (C).
enum CausticsSolver : int { kCausticsBdptGuided = 0, kCausticsMnee = 1 };

struct RenderSettingsData {
    int resolutionX = 960;
    int resolutionY = 540;
    int samplesPerPixel = 64;
    int maxDepth = 8;

    int rrStartDepth = 3;
    int lightSamples = 2;          // NEE samples per bounce (MIS with BSDF)
    int seed = 0;
    int integrator = kIntegratorPathTracer;

    float clampIndirect = 10.0f;   // <= 0 disables firefly clamping of indirect contribs
    float exposure = 0.0f;
    float gamma = 2.2f;
    int toneMapper = kToneAces;

    int backend = kBackendCpuEmbree;
    int envVisibleCamera = 1;
    int tileSize = 32;
    int threads = 0;               // 0 = hardware concurrency

    float aoDistance = 1.0f;
    int pathGuiding = 1;           // OpenPGL on CPU (Embree); recommended with BDPT (D+A)
    // BDPT+Guiding vs MNEE — see CausticsSolver. OptiX falls back to unidirectional PT.
    int causticsSolver = kCausticsBdptGuided;
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
    const ProceduralNode* procedurals = nullptr;

    int meshCount = 0;
    int instanceCount = 0;
    int materialCount = 0;
    int lightCount = 0;
    int envMapCount = 0;
    int textureCount = 0;
    int proceduralCount = 0;
    int domeLightIndex = -1;  // first dome light, used for ray misses

    CameraData camera;
    RenderSettingsData settings;
    Bounds3 worldBounds;
};

}  // namespace sol
