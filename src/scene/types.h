// Plain-old-data scene description shared by the Embree and OptiX backends.
// No STL containers here: the very same structs are uploaded to the GPU.
#pragma once

#include "core/math.h"

namespace sol {

// Arnold-like ray switch: absolute indices into SceneView::materials (-1 = use
// the material that owns this table — typically the camera/base assignment).
// Port selection follows the *incoming* ray type (aiRaySwitch). Solstice adds
// `caustics` for photon / MNEE / BDPT light-tracing only — never for camera rays.
struct RaySwitchTable {
    int camera = -1;
    int shadow = -1;
    int diffuseReflection = -1;
    int specularReflection = -1;
    int diffuseTransmission = -1;
    int specularTransmission = -1;
    int sss = -1;
    // Solstice extension: light-side caustic transport only (not camera rays).
    int caustics = -1;
};

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

    // Arnold standard_surface: separate specular / transmission tints.
    Vec3 specularColor{1.0f, 1.0f, 1.0f};
    Vec3 transmissionColor{1.0f, 1.0f, 1.0f};

    Vec3 emissionColor{0.0f, 0.0f, 0.0f};
    float emissionStrength = 0.0f;

    float opacity = 1.0f;
    // Mix weight vs diffuse (Standard Surface): base_mix = (1-w)*diffuse + w*SSS.
    float subsurface = 0.0f;
    int doubleSided = 1;
    // Arnold-style: when 0 this transmissive/specular material still shades
    // normally but does not cast caustics (MNEE / BDPT LT / photon map).
    // Shadows then use shadowOpacity even with render-settings caustics ON.
    int contributeCaustics = 1;

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
    int specularColorTex = -1;

    int transmissionColorTex = -1;
    // Indices into SceneView::procedurals (-1 = none). Procedurals win over textures.
    int baseColorProc = -1;

    int roughnessProc = -1;
    int metallicProc = -1;
    int opacityProc = -1;
    int emissionProc = -1;

    int normalProc = -1;
    int subsurfaceProc = -1;
    int specularColorProc = -1;
    int transmissionColorProc = -1;
    // Fake shadow control for transmissive surfaces when render-settings caustics
    // are OFF: 1 = fully opaque shadow, 0 = fully open. With caustics ON shadow
    // rays treat glass as opaque and light arrives via MNEE / BDPT instead.
    float shadowOpacity = 1.0f;

    // MaterialX normalmap.scale / bump.scale (tangent XY strength).
    float normalScale = 1.0f;
    // Height-field bump (MaterialX <bump>). When set, wins over normalTex/normalProc.
    int bumpTex = -1;
    int bumpProc = -1;

    // Geometric displacement (MaterialX <displacement> on surfacematerial).
    // Tessellation (subdiv / dicing) lives on geometry; the shader supplies height only.
    int displacementTex = -1;
    int displacementProc = -1;
    float displacementScale = 1.0f;
    float displacementHeight = 0.0f;  // unused (no constant-height mode)
    float displacementZeroValue = 0.5f;
    int subdivIterations = 0;   // authored on geometry; densify may ignore under Screen Adaptive
    int autobump = 1;           // Pref-space displace normal (Arnold-like), all ray types
    int displacementVector = 0; // 0 = along normal (float), 1 = vector displace
    // PT Spectral metal complex IOR (RGB ≈ η/κ at ~650/550/450 nm). Used only by
    // the spectral integrator; RGB Path Tracer ignores these.
    Vec3 conductorEta{1.5f, 1.5f, 1.5f};
    Vec3 conductorK{0.0f, 0.0f, 0.0f};

    // Chromatic dispersion of the transmission lobe (Abbe number, Arnold-style
    // `dispersion_abbe`): 0 = off; typical glass 20–60, lower = stronger rainbow.
    float dispersionAbbe = 0.0f;
    // Thin-film iridescence on the specular reflection lobe (soap bubble / oil):
    // film thickness in nanometres (0 = off) and film IOR.
    float thinFilmThickness = 0.0f;
    float thinFilmIor = 1.4f;
    // Arnold Advanced → Internal Reflections (1 = on). When off, rays inside a
    // dielectric skip Fresnel reflections (TIR still reflects — nowhere else to go).
    float internalReflections = 1.0f;

    // Separate dielectric coat (pbrt-style overlay, not mixed into the base lobes).
    // thickness is an optical depth in the coat medium (Beer–Lambert); 0 = clear.
    float coat = 0.0f;
    float coatRoughness = 0.1f;
    float coatIor = 1.5f;
    float coatThickness = 0.0f;
    Vec3 coatColor{1.0f, 1.0f, 1.0f};

    // MaterialX volumeshader (connected to surfacematerial.volumeshader).
    // When hasVolumeShader != 0, fog/VDB path uses these coefficients.
    int hasVolumeShader = 0;
    float volumeDensity = 1.0f;
    float volumeAnisotropy = 0.0f;  // HG g
    Vec3 volumeAbsorption{0.0f, 0.0f, 0.0f};
    Vec3 volumeScattering{1.0f, 1.0f, 1.0f};
    Vec3 volumeEmission{0.0f, 0.0f, 0.0f};
    float volumeEmissionStrength = 0.0f;

    // MaterialX ray_switch_shader → per-ray-type material indices (scene-absolute
    // after Stage::toScene). -1 = this material.
    RaySwitchTable raySwitch;
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
// Participating media
// ---------------------------------------------------------------------------
struct MediumData {
    int type = 0;            // 0 = none, 1 = homogeneous, 2 = OpenVDB fog, 3 = OpenVDB SDF surface
    int volumeIndex = -1;    // index into Scene::volumes / volumePaths
    Vec3 sigmaA{0.0f};       // absorption coefficient (homogeneous / volume shader)
    Vec3 sigmaS{0.0f};       // scattering coefficient (homogeneous / volume shader)
    float g = 0.0f;          // Henyey-Greenstein asymmetry parameter [-1, 1]
    float density = 1.0f;    // density scale (multiplies sigmaA/sigmaS and VDB samples)
    Vec3 emission{0.0f};     // volume emission * strength (RGB)
    int pad0 = 0;
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
    // Object-space AABB (for SMS seed cones around caustic casters).
    Vec3 boundsLo{0.0f, 0.0f, 0.0f};
    Vec3 boundsHi{0.0f, 0.0f, 0.0f};
    // Pref / Nref (pre-displace cage). Null = same as positions/normals.
    const Vec3* restPositions = nullptr;
    const Vec3* restNormals = nullptr;
    // Deformation motion blur: packed positions [key * vertexCount + i], key in [0, motionKeyCount).
    // Key 0 usually aliases `positions`. Null / count 1 = static mesh.
    const Vec3* motionPositions = nullptr;
    int motionKeyCount = 1;
    // Optional per-triangle edge mask (see Mesh::triEdgeMask). Null = draw all edges.
    const uint8_t* triEdgeMask = nullptr;
    // Optional cage wire overlay (authored n-gon edges). Null = unused.
    const uint32_t* wireIndices = nullptr;
    const Vec3* wirePositions = nullptr;
    const Vec3* wireNormals = nullptr;
    uint32_t wireEdgeCount = 0;     // number of edges (== wireIndices pairs)
    uint32_t wireVertexCount = 0;
};

// Visibility bits for primary vs shadow rays (Embree mask / OptiX visibilityMask).
constexpr int kVisShadow = 0x1;
constexpr int kVisPrimary = 0x2;
constexpr int kVisAll = kVisShadow | kVisPrimary;

struct InstanceData {
    Mat4 xform;      // object -> world (shutter center / key 0)
    Mat4 xformInv;   // world -> object
    int meshIndex = -1;
    int materialIndex = -1;
    int lightIndex = -1;   // >= 0 when this instance is an area light proxy
    int visibleCamera = 1;
    // Embree/OptiX visibility: shadow rays use kVisShadow only so area-light
    // proxies can opt out of casting self-shadows via kVisPrimary alone.
    int visibilityMask = kVisAll;
    // Transform motion blur: indices into SceneView::motionXforms.
    int motionKeyOffset = 0;
    int motionKeyCount = 1;
    // Index into SceneView::media; -1 = no participating medium on this instance.
    int mediumIndex = -1;
    // Index into Scene::volumes (SDF / fog grid). -1 = none.
    int volumeIndex = -1;
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
    // Kelvin; 0 = off (use RGB color only). Spectral integrators: blackbody × tint.
    float colorTemperatureK = 0.0f;

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
    // Arnold-style: when 0 this light still does direct lighting but does not
    // participate in caustic transport (MNEE / BDPT light-tracing delta chains /
    // BSDF specular→light after a diffuse bounce).
    int contributeCaustics = 1;
    // Physical Sky distant sun: draw the solar disc on camera / reflection misses.
    // Regular distant lights stay invisible (NEE only). The sky env map has no disc.
    int cameraSunDisc = 0;

    SR_HD Vec3 emittedRadiance() const { return color * (intensity * exp2f(exposure)); }
};

// ---------------------------------------------------------------------------
// Light BVH (PBRT-style; finite lights only — dome/distant kept separately)
// ---------------------------------------------------------------------------
// Flat binary BVH node.  Leaf: childOrLight = scene light index, rightChild = -1.
// Interior: childOrLight = left child index, rightChild = right child index.
// All fields are plain int / float so the struct is GPU-safe.
struct LightBvhNode {
    Vec3  bMin{0.0f, 0.0f, 0.0f};   // world AABB min
    Vec3  bMax{0.0f, 0.0f, 0.0f};   // world AABB max
    float power       = 0.f;          // total emitted power in this subtree
    int   childOrLight = -1;          // leaf: scene light index; interior: left child node index
    int   rightChild   = -1;          // interior only; -1 for leaves
    int   isLeaf       = 0;
    // pbrt-v4 LightBounds emission cone: axis w, θ_o (emit cone), θ_e (falloff).
    Vec3  coneAxis{0.0f, 0.0f, 1.0f};
    float cosThetaO = -1.0f;          // cos of emit-cone half-angle (–1 = 4π)
    float cosThetaE = 0.0f;           // cos of extra falloff (0 = π/2 Lambert)
    int   twoSided  = 0;
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

    // 0 = thin lens (Embree + OptiX). 1 = polynomial optics (Embree only).
    int opticalModel = 0;
    // Index into polynomialOpticsLensNames() when opticalModel == 1.
    int lensModel = 19;  // cooke__speed_panchro__1920__50mm
    float opticalWavelengthNm = 550.0f;
    // When set with polynomial optics: each camera sample picks R, G or B wavelength
    // so longitudinal/lateral chromatic aberration appears as coloured fringing.
    int chromaticAberration = 0;

    float shutterOpen = 0.0f;   // normalized shutter time at open (usually 0)
    float shutterClose = 0.0f;  // normalized shutter time at close (usually 1 when MB on)
    float nearClip = 0.001f;
    float farClip = 1e7f;
};

// ---------------------------------------------------------------------------
// Render settings
// ---------------------------------------------------------------------------
enum ToneMapper : int { kToneNone = 0, kToneReinhard = 1, kToneAces = 2 };  // legacy (removed from UI)
enum RenderBackendType : int {
    kBackendCpuEmbree = 0,
    kBackendGpuOptix = 1,
    kBackendXpu = 2,  // Embree + OptiX together (Karma / RenderMan XPU)
};

SR_INL SR_HD bool renderDeviceUsesGpu(int backend) {
    return backend == kBackendGpuOptix || backend == kBackendXpu;
}
SR_INL SR_HD bool renderDeviceUsesCpu(int backend) {
    return backend == kBackendCpuEmbree || backend == kBackendXpu;
}
SR_INL SR_HD bool renderDeviceIsXpu(int backend) { return backend == kBackendXpu; }

// XPU work schedule (Render Settings → Engine, visible only when backend is XPU).
// Overlap (default): GPU fills even spp until Embree finishes one odd spp, then one D2H add.
// Mixture: Karma-style independent full-frame estimators, host blend, automatic spp share.
enum XpuSchedule : int {
    kXpuScheduleOverlap = 0,
    kXpuScheduleMixture = 1,
};
// BDPT / spectral / wireframe are CPU / Embree only; GPU and XPU require Path Tracer
// and stop with an error (no Embree fallback).
// Menu order matches these values: Path Tracer, BDPT, Direct Lighting, AO,
// PT Spectral, BDPT Spectral, Wireframe.
enum IntegratorMode : int {
    kIntegratorPathTracer = 0,
    kIntegratorBdpt = 1,
    kIntegratorDirectLighting = 2,
    kIntegratorAmbientOcclusion = 3,
    kIntegratorSpectralPath = 4,  // PT Spectral (CPU / Embree)
    kIntegratorSpectralBdpt = 5,  // BDPT Spectral (CPU / Embree)
    kIntegratorWireframe = 6,     // Geometry edge overlay (barycentric)
};

// How refractive / reflective caustics are estimated when settings.caustics != 0.
// Menu order matches these values: MNEE, MNEE+Photon, Photon / VCM.
enum CausticsEngine : int {
    // Manifold next-event (PT) / BDPT LT+MNEE — best for near-delta glass.
    kCausticsEngineMnee = 0,
    // Smart pick: delta glass → MNEE; rough refractive casters → Photon.
    // BDPT keeps light-tracing; MNEE upgrades under-glass SDS when not Photon.
    kCausticsEngineAuto = 1,  // UI label: MNEE+Photon
    // Caustic-only photon map gather (VCM-style density estimation) — better for
    // rough glass and caustics seen through thick refractive bases.
    kCausticsEnginePhoton = 2,
};

// Camera AA / DoF primary samples. Path bounce dims use Owen-scrambled Sobol (PBRT4).
enum PixelSampler : int {
    kPixelSamplerSobol = 0,        // Owen-scrambled Sobol
    kPixelSamplerBlueNoise = 1,    // 64×64 BN CP tile
    kPixelSamplerXorshift = 2,     // Marsaglia xorshift32 white jitter
    kPixelSamplerGenPnt2D = 3,     // plastic-number R2 (Roberts), n = sampleIndex
    // Diagnostic: pixel center + U(-1,1)*manualTestMult per axis, clamped to [0,1).
    kPixelSamplerManualTest = 4,
};

// How the image is scheduled / written (Render Settings → Sampling Type).
enum SamplingEngine : int {
    // PBRT-style FilmTile buckets: local accum + mergeFilmTile, strong (x,y,spp) seed.
    kSamplingEngineBuckets = 0,
    // No buckets — parallel scanlines, strong seed.
    kSamplingEngineProgressive = 1,
};

// Render Settings → Diagnostic: replace beauty with a sampling/seed field.
enum SamplingDebug : int {
    kSamplingDebugOff = 0,
    kSamplingDebugPixelJitter = 1,  // R=jx G=jy — shows BN period vs Sobol/Xorshift
    kSamplingDebugPathRng = 2,      // first path-RNG float as gray — seed seams
    kSamplingDebugBucket = 3,       // color by render bucket (tileSize)
    kSamplingDebugPixelHash = 4,    // RGB from hashPixelSample(x,y,spp,seed)
};

// Film reconstruction filter (how a continuous sample is weighted into pixels).
enum PixelFilter : int {
    kPixelFilterBox = 0,       // 1×1 box — historical default (hard pixels)
    kPixelFilterTriangle = 1,  // tent
    kPixelFilterGaussian = 2,  // truncated Gaussian
    kPixelFilterMitchell = 3,  // Mitchell–Netravali B=C=1/3
};

// Render working colour space (Film). TX auto-convert always targets ACEScg (Arnold).
enum WorkingColorSpace : int {
    kWorkingSpaceSrgbLinear = 0,
    kWorkingSpaceAcesCg = 1,
};

// mplay-style colour management: Classic = linear→sRGB OETF (no tone map);
// OCIO = config Display/View.
enum ColorManagementMode : int {
    kColorClassic = 0,
    kColorOcio = 1,  // default
};

// Monitor view transform (chrome strip). Labels: sRGB / Rec.709 / Rec.2020 / Raw.
enum ViewTransform : int {
    kViewSrgbAces = 0,     // sRGB (enum name kept for compatibility)
    kViewRec709Aces = 1,   // Rec.709
    kViewRaw = 2,          // no display transform (linear clamp)
    kViewRec2020 = 3,      // Rec.2020
};

// Framebuffer resolve / viewport quantize / EXR save bit depth (accum stays float).
enum OutputBitDepth : int {
    kBitDepth8 = 8,
    kBitDepth16 = 16,
    kBitDepth32 = 32,
};

// Chromatic dispersion sampling (material dispersion_abbe / lens CA).
enum DispersionMode : int {
    // Current: one random hero RGB channel per sample; mask whole path to that channel.
    kDispersionHero = 0,
    // Lazy hero mask (only if path hit dispersing media) + stratified channel +
    // dispersion IOR only on the first N glass interfaces.
    kDispersionOptimized = 1,
    // Trace R+G+B heroes and average (≈3× cost when dispersion is present).
    kDispersionSpectral3 = 2,
    // No IOR split: single path + artistic chromatic tint on transmission.
    kDispersionFake = 3,
};

enum SubdivType : int {
    kSubdivNone = 0,
    kSubdivCatclark = 1,
    kSubdivLinear = 2,
};

enum DicingCameraMode : int {
    kDicingCameraRender = 0,
    kDicingCameraCustom = 1,
};

struct RenderSettingsData {
    int resolutionX = 960;
    int resolutionY = 540;
    int samplesPerPixel = 64;
    int maxDepth = 8;       // surfaces + volume scatters; UI up to 4096 for dense MS
    int rrStartDepth = 3;   // raise near maxDepth for deep volume multiple scattering
    int lightSamples = 2;          // UI; integrators take one NEE sample (pbrt-v4)
    int seed = 0;
    int integrator = kIntegratorPathTracer;

    // Arnold-style sample clamps in linear pixel radiance (<= 0 disables).
    // Direct Clamp: eye-path contributions (PT/BDPT eye, NEE, MNEE, photons).
    // Indirect Clamp: BDPT light-tracing splat deposits (converted to radiance via / (W·H)).
    float clampDirect = 10.0f;
    float clampIndirect = 10.0f;
    // Working colour space (Film). Display view is separate (viewport chrome).
    int workingSpace = kWorkingSpaceAcesCg;
    // Classic vs OCIO (updated live from the render view chrome). Default OCIO.
    int colorManagement = kColorOcio;
    // Viewport view transform (updated live from the render view chrome).
    int viewTransform = kViewSrgbAces;
    // Resolve / save bit depth: 8, 16, or 32 (accumulation remains float).
    int bitDepth = kBitDepth16;
    // Film reconstruction filter (Box = current 1-pixel behaviour).
    int pixelFilter = kPixelFilterBox;
    float filterRadius = 0.5f;  // pixels; 0 = use defaultFilterRadius(pixelFilter)
    // Karma XPU variance oracle. 0 = off (always take Samples Per Pixel).
    // Default 0.01 matches Karma's Variance Threshold.
    float noiseThreshold = 0.01f;

    int backend = kBackendCpuEmbree;
    int xpuSchedule = kXpuScheduleOverlap;  // ignored unless backend == XPU
    int envVisibleCamera = 1;
    int tileSize = 32;             // bucket size; 0 = PBRT-style auto
    int pixelSampler = kPixelSamplerSobol;  // camera AA / DoF generator
    // Manual-Test only: pixel jitter = 0.5 + U(-1,1)*mult, clamped to [0,1).
    float manualTestMult = 0.0f;
    int samplingEngine = kSamplingEngineBuckets;  // Buckets / Progressive
    int threads = 0;               // 0 = hardware concurrency

    float aoDistance = 1.0f;
    // Wireframe integrator: edge half-width in screen pixels (anti-aliased).
    float wireframeThickness = 1.0f;
    int pathGuiding = 0;           // OpenPGL indirect guides (CPU / Embree); PT + BDPT
    // Opt-in, biased: Hyperion similarity on deep volume MS (Engine checkbox).
    // Off (default) keeps authored g / σs. On: lerp g→0 from volume bounce 5..20.
    int volumeSimilarity = 0;
    // Enable caustic light transport (specular→diffuse). Off = dark glass shadows
    // (soften per-material with shadow_opacity / contribute_caustics).
    int caustics = 1;
    // Which estimator carries caustics when enabled (see CausticsEngine).
    int causticsEngine = kCausticsEngineAuto;  // MNEE+Photon smart pick
    // Firefly cap for paths that look through glass/mirrors at a light (SDS) and for
    // BDPT near-specular NEE/connections. Those never converge with more samples when
    // the light is small; a safety floor of 10 is always applied when this is left at 0
    // (see causticFireflyCap). Raise it to tighten further. Light-tracing caustics on
    // diffuse surfaces use Indirect Clamp, not this.
    float causticClamp = 0.0f;
    // Photon / VCM caustic map (used when causticsEngine == Photon).
    int photonCount = 100000;
    float photonRadius = 0.08f;    // gather radius in scene units (shrinks over spp)
    // Progressive pass index (set per sample by the CPU backend).
    int progressiveSample = 0;
    // Dispersion sampling strategy (see DispersionMode).
    int dispersionMode = kDispersionHero;
    // Optimized mode: max dispersing glass interfaces that change IOR (enter+exit = 2).
    int dispersionMaxInterfaces = 2;

    // Arnold-style motion blur (CPU / Embree). Shutter is centered on the frame.
    int motionBlur = 0;
    int motionKeys = 2;            // transform / deformation samples across the shutter
    float shutterLength = 0.5f;    // fraction of a frame (Arnold default 0.5)

    // Displacement tessellation (applied once at Render / headless start).
    int frustumCull = 1;              // skip subdiv for meshes outside padded frustum
    float frustumPadding = 10.0f;     // screen-space margin as % of width/height
    int screenAdaptive = 0;           // Karma-like dicing by projected edge length
    int dicingCameraMode = kDicingCameraRender;  // render camera or custom
    // Safety fuse for Screen Adaptive / densify (millions of triangles). Default 10M.
    int dicingPolyLimitM = 10;
    // Master switch: off = render cages, skip subdiv + geometric displacement.
    int enableDisplacement = 1;

    // PT Spectral / BDPT Spectral.
    int spectralSamples = 4;      // hero λ count (UI: 2..16)
    int spectralBins = 16;        // fixed bins for multilayer spectral EXR (8..32)
    int spectralExr = 0;          // write spectral multilayer EXR on save when set
    // Beauty conversion color space. Default ACEScg — same as workingSpace.
    int spectralColorSpace = 1;   // kSpectralColorSpaceAcesCg
    // Wavelength sampling: 0 = visible importance (pbrt), 1 = uniform stratified.
    int spectralWavelengthSampling = 0;
    // Film: false-color debug from spectral bins (PT Spectral). Diagnostic group.
    int filmFalseColor = 0;
    int filmFalseColorBin = 0;    // which bin to visualise (0 .. spectralBins-1)
    // Sampling / seed diagnostics (skip light transport; write debug RGB).
    int samplingDebug = 0;        // SamplingDebug enum

    // Texture TX cache: convert source textures to .tx mipmaps (maketx → ACEScg).
    int enableTxCache = 1;
    char txCacheDir[256] = "tx_cache";
    // OCIO: when ocioUseEnv != 0, read config from the OCIO environment variable.
    // Otherwise use ocioConfigPath (Render Settings → Film).
    int ocioUseEnv = 1;
    char ocioConfigPath[512] = "";
};

// ---------------------------------------------------------------------------
// The flattened scene as seen by an integrator.
// ---------------------------------------------------------------------------
class VolumeGrid;

struct SceneView {
    const MeshView* meshes = nullptr;
    const InstanceData* instances = nullptr;
    const Material* materials = nullptr;
    const LightData* lights = nullptr;
    const EnvMapView* envMaps = nullptr;
    const TextureView* textures = nullptr;
    const ProceduralNode* procedurals = nullptr;
    const MediumData* media = nullptr;
    // Host CPU only — OpenVDB grids (indexed by InstanceData::volumeIndex).
    // OptiX uses LaunchParams::volumes (occupancy + the same majorant bricks).
    const VolumeGrid* const* volumes = nullptr;

    int meshCount = 0;
    int instanceCount = 0;
    int materialCount = 0;
    int lightCount = 0;
    int envMapCount = 0;
    int textureCount = 0;
    int proceduralCount = 0;
    int mediumCount = 0;
    int volumeCount = 0;
    int domeLightIndex = -1;  // first dome light, used for ray misses
    // Any material with dispersionAbbe > 0 — enables hero-channel sampling.
    int hasDispersion = 0;

    CameraData camera;
    RenderSettingsData settings;
    Bounds3 worldBounds;

    // Motion blur keys (host pointers; Embree reads them when building / shading).
    const Mat4* motionXforms = nullptr;        // packed: instance keys via InstanceData offsets
    const Mat4* cameraMotionXforms = nullptr;  // camera keys across shutter [0,1]
    int cameraMotionKeyCount = 1;

    // Light BVH for position-aware light selection (finite lights only).
    // Uploaded to OptiX; flux-only fallback if the tree is empty.
    const LightBvhNode* lightBvh          = nullptr;
    int                 lightBvhNodeCount  = 0;
    // Infinite lights (dome, distant) are kept outside the BVH.
    const int* infiniteLightIndices        = nullptr;
    int        infiniteLightCount          = 0;
    // Precomputed total-power sums used for the infinite-vs-finite split decision.
    float infiniteLightPower               = 0.f;
    float finiteLightPower                 = 0.f;
};

}  // namespace sol
