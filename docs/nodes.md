# Node reference

Every node takes the stage from its input, edits it and passes it on. Press `Tab` in the
network editor to add one, and set the blue display flag on the node you want to render.

Prim patterns accept space separated globs with `*` and `?`, matched against the full prim
path, for example `/geo/body* /geo/head`.

## Geometry

### Alembic Import (`alembic`)
Reads polygon meshes and subdivision cages from an `.abc` archive, including the xform
hierarchy.

| Parameter | Meaning |
| --- | --- |
| Alembic File | path to the archive; relative paths resolve against the scene file |
| Prim Path | scene graph root for the imported prims, default `/geo` |
| Path Filter | only import Alembic objects whose path matches this glob |
| Time | sample time in seconds, snapped to the nearest stored sample |
| Import Scale | unit conversion applied to point positions |
| Import Normals / UVs | read the stored attributes instead of computing smooth normals |
| Transform | translate, rotate, scale applied on top of the archive transforms |

Face varying normals or UVs force the mesh to be split per face corner; missing normals are
computed by welding equal positions, so imported meshes stay smooth.

Imported meshes keep an **n-gon cage** (`faceVertexCounts` / `faceVertexIndices`). Embree and
OptiX still ray-trace triangles densified with Mapbox earcut (concave-safe), matching Houdini-style
polygon handling.

### Sphere, Grid, Box, Tube
Polygonal primitives with a *Prim Name* and the usual transform block. Grid is the usual
ground plane, Sphere and Box are handy stand-ins while lighting.

## Volume

Houdini-like VDB SOPs. Output of `vdbfrompolygons` is a **Volume prim only** (no mesh passthrough).

| Node | What it does |
| --- | --- |
| **VDB from Polygons** (`vdbfrompolygons`) | Mode: SDF / Fog Volume; Voxel Size; Exterior / Interior Band |
| **VDB File** (`vdbfile`) | Load `.vdb` from disk into a Volume prim |
| **SDF to Polygons (VDB)** (`sdftopolygons_vdb`) | OpenVDB `volumeToMesh` |
| **SDF to Polygons (DCSDD)** (`sdftopolygons_dcsdd`) | Dual Contouring of Signed Distance Data (Carrera et al. 2026) |

CPU path tracer renders SDF level sets by sphere tracing and Fog volumes with delta tracking.
Assign a MaterialX graph with `surfacematerial.volume` → `standard_volume` (density, anisotropy,
absorption, scattering, emission, emission_color) for volume shading; the surface shader shades SDF
hits. Material container ports are short names (`surface`, `displacement`, `volume`); MaterialX
long names (`surfaceshader`, `displacementshader`, `volumeshader`) still resolve when loading XML.

## Utility

| Node | What it does |
| --- | --- |
| **Transform** | multiplies the transform of every prim matching *Prim Pattern* |
| **Merge** | combines up to four stages; paths are made unique automatically |
| **Switch** | passes through the input selected by *Input Index* |
| **Prune** | deactivates prims matching a pattern (or everything else with *Invert*) |
| **Null** | pass through, useful as a bookmark or a stable display point |

## Material

### Material (`material`)
Creates a principled surface and assigns it to every mesh prim matching *Assign To*.

| Parameter | Meaning |
| --- | --- |
| Base Color | diffuse albedo, or the reflectance tint when metallic |
| Roughness | 0 is a mirror, 1 is fully rough; values below ~0.002 become a perfect specular |
| Metallic | blends from a dielectric to a conductor |
| Specular | dielectric specular level, 0.5 corresponds to an F0 of 0.04 |
| IOR | index of refraction used by the specular and transmission lobes |
| Transmission | 0 opaque, 1 glass; combine with roughness for frosted glass |
| Opacity | stochastic cut-out transparency |
| Emission Color / Strength | makes the surface emit light (sampled indirectly, not by NEE) |

Chain several material nodes with different patterns to shade different parts of an import.

## Lighting

All lights share Color, Intensity, Exposure (a power of two multiplier), Cast Shadows and a
transform block.

| Node | Shape specific parameters | Notes |
| --- | --- | --- |
| **Dome Light** | HDRI Texture, Visible To Camera | equirectangular map, +Y up, importance sampled; without a texture it acts as a uniform sky |
| **Distant Light** | Angular Diameter, Normalize | travels along the -Z axis of its transform; with Normalize on, Intensity behaves as irradiance and the disc size only changes shadow softness |
| **Rect Light** | Width, Height, Normalize, Two Sided, Visible To Camera | XY rectangle emitting along -Z |
| **Disk Light** | Radius, Normalize, Two Sided, Visible To Camera | XY disk emitting along -Z |
| **Sphere Light** | Radius, Normalize, Visible To Camera | emits in every direction |

*Normalize* divides the intensity by the emitting area (or solid angle), so resizing a light
keeps its exposure. *Visible To Camera* off makes primary rays pass straight through the
light while it keeps illuminating the scene.

## Camera

### Camera (`camera`)
Defines the render camera. The node that authors the camera also selects it as the render
camera for the stage.

| Parameter | Meaning |
| --- | --- |
| Focal Length / Sensor Width | field of view in millimetres, as on a real camera |
| F-Stop | 0 disables depth of field, otherwise the lens radius is focal / (2 · f-stop) |
| Focus Distance | distance of the focal plane in scene units |
| Use Look At | place the camera with Eye/Look At/Up instead of the transform block |

*Render → Copy View To Camera Node* writes the current viewport position into these
parameters, and *Look Through Camera Node* goes back to the authored camera.

## Render

### Render Settings (`rendersettings`)

| Group | Parameters |
| --- | --- |
| Image | Resolution X/Y, Pixel Filter, Filter Radius, Output Path, Bit Depth, TX cache |
| Sampling | Samples Per Pixel, Pixel Oracle (Uniform / Variance), Noise Threshold, Light Samples, Pixel Sampler, Sampling Type, Bucket Size, Seed, Direct / Indirect Clamp. Variance overlay shows `N% skip`; at 0.01 a noisy 128 spp frame is often 0%. |
| Engine | Render Device (CPU Embree, GPU OptiX, or XPU), Integrator, spectral options, CPU Threads, AO Distance, Dispersion, Indirect Guides, Volume Similarity |
| Depth | Max Ray Depth, Russian Roulette Depth |
| Caustics | Caustics, Caustics Engine, Caustic Firefly Clamp, Photon Count / Radius |
| Motion Blur | Enable Motion Blur, Motion Keys, Shutter Length |
| Displacement | Frustum Cull, Screen Adaptive, Enable Displacement, Dicing Camera |
| Film | Working Space, OCIO, Environment Visible To Camera |
| Diagnostic | Sampling Debug, Spectral False Color |

Folders are Houdini-style tabs in the parameter panel: click a tab to show only that group.

XPU (`Render Device` = Embree+OptiX) is Path Tracer only. Even samples run on the GPU, odd
samples on the CPU, both full-frame. If OptiX cannot start, rendering stops with an error
(no Embree fallback). Non-PT integrators also stop on GPU/XPU.

Film settings are applied when the framebuffer is displayed or written to an LDR file; `.exr`
and `.hdr` outputs stay linear.
