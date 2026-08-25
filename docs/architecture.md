# Architecture

```
                    ┌──────────────┐   cook    ┌────────┐  flatten  ┌───────┐
 UI edits ────────▶ │  NodeGraph   │ ────────▶ │ Stage  │ ────────▶ │ Scene │
                    └──────────────┘           └────────┘           └───┬───┘
                                                                        │ build
                                       ┌────────────────────────────────┴─────────┐
                                       ▼                                          ▼
                              ┌──────────────────┐                     ┌────────────────────┐
                              │ Embree backend   │                     │ OptiX wavefront    │
                              │ (CPU, tiled)     │                     │ init/intersect/    │
                              │ integrator.h     │                     │ shade modules      │
                              │ full features    │                     │ pinhole, basic BSDF│
                              └────────┬─────────┘                     └─────────┬──────────┘
                                       └───────────► Framebuffer ◀───────────────┘
```

## Layers

| Directory | Depends on | Contents |
| --- | --- | --- |
| `src/core` | – | vectors, matrices, sampling, RNG, float images, 2D distributions, logging, thread pool |
| `src/scene` | core | POD scene description (`types.h`) plus the host side containers and primitive builders |
| `src/io` | scene, Qt Gui | Alembic import, Radiance/OpenEXR/LDR image loading and saving |
| `src/render` | scene | BSDF, light sampling, the integrator, the framebuffer, both backends and the render session |
| `src/nodes` | render, io, Qt Core | parameters, nodes, the stage, the graph and the built in node library |
| `src/ui` | nodes, Qt Widgets | the desktop application |
| `src/app` | nodes | default networks, `.scene` documents, headless rendering |

`src/core` and `src/scene/types.h` are written so that `nvcc` can compile them: every function
is marked with `SR_HD` (`__host__ __device__` under CUDA) and no STL container ever crosses
into device code.

## Cooking

A node network is evaluated in the same spirit as a Houdini Solaris LOP chain.

* A **Stage** is an ordered list of prims addressed by path (`/geo/sphere`, `/lights/key`)
  plus render settings and the active render camera.
* Each node receives the stages of its inputs. By default the graph pre-fills the output
  stage with a copy of input 0, so a node only has to *add to* or *edit* what flows through
  it. `merge` and `switch` opt out of that and assemble their output themselves.
* Results are cached per node. Editing a parameter marks that node and everything downstream
  dirty, so an edit deep in the network does not re-import the Alembic file at the top of it.
  The Alembic node additionally caches its file read, keyed by file, time and import options.
* `Stage::toScene()` flattens the prims into a `Scene`: meshes are de-duplicated by pointer,
  area lights get proxy geometry so rays can hit them, and the world bounds and camera are
  finalised. Without an authored camera the scene frames its own contents.

## Rendering

`RenderSession` owns a worker thread that asks the active `RenderDevice` for one sample per
pixel at a time and accumulates it into a `Framebuffer`. Cancellation is a flag polled by the
backends, which is what makes edits feel instant: the UI stops the session, re-cooks and
starts a fresh one.

The Embree backend implements `intersect` / `occluded` and hands them to `traceRadiance()` in
`src/render/integrator.h` (volumes, SSS, procedurals, BDPT, MNEE, polynomial optics). OptiX does
**not** compile that header. Like Cycles, GPU path tracing is a **wavefront** of small kernels:
`init_from_camera`, `intersect_closest`, `intersect_shadow`, `shade_surface`, `shade_background`,
`shade_shadow`. `optixTrace` lives only in the intersect modules; the BSDF lives only in
`shade_surface`. That is what keeps `cicc` from seeing one megakernel.

* **Camera rays (CPU)** use the physical camera: focal length and sensor width define FOV, a
  non-zero f-stop is a thin lens, `opticalModel == 1` is polynomial optics. **OptiX is pinhole
  only.**
* **Surfaces (CPU)** use the full principled BSDF plus procedurals / SSS. **OptiX** evaluates
  2D image maps and Lambert / GGX / dielectric transmission only.
* **Lights** are sampled analytically (rect and disk by area, sphere by cone, distant by its
  angular diameter, dome through the environment CDF) and combined with BSDF sampling using
  the power heuristic.
* **Termination** is Russian roulette after a configurable depth, with optional clamping of
  indirect contributions to suppress fireflies.

## The Embree backend

One `RTCScene` per mesh, instanced into a top level scene with
`RTC_GEOMETRY_TYPE_INSTANCE`. Geometry ids are forced to match instance indices so a hit maps
straight back to the scene description. Vertex and index buffers are allocated by Embree so
its padding requirements are satisfied. Each sample splits the image into tiles that are
distributed across a persistent thread pool.

## The OptiX backend

Each mesh becomes a compacted GAS; instances become an IAS with `instanceId` set to the
instance index. Materials, lights, instances and the environment CDF tables are uploaded once
per scene build and referenced by a `SceneView` inside the launch parameters.

Device programs are split the way Cycles splits GPU work:

* `optix_hit_miss.cu` — tiny closest-hit / miss payload writers.
* `optix_intersect_closest.cu` / `optix_intersect_shadow.cu` — the only TUs that call
  `optixTrace` (Cycles `__raygen__kernel_optix_integrator_intersect_*`).
* `optix_init_from_camera.cu` — pinhole raygen (no DoF / optics).
* `optix_shade_surface.cu` / `optix_shade_background.cu` / `optix_shade_shadow.cu` — shading.
  Lambert / GGX / glass + 2D maps. No `optixTrace`.

Volumes, SSS, MaterialX procedurals, BDPT, MNEE, wireframe, AO, and polynomial-optics cameras
stay on Embree. This engine does not ship SVM / MDL, so there is no on-the-fly shader JIT.

Each `.cu` is a separate `nvcc -ptx` job (ninja compiles them in parallel) and a separate OptiX
module in one pipeline. The host launches the raygen for the current wavefront stage, swapping
the SBT raygen record like Cycles. Two ray types: radiance (six payload registers) and shadow
rays that terminate on the first hit.
