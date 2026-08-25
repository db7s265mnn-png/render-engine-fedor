// Wavefront OptiX modules (Cycles-style split kernels + Iray tail):
//   optix_init_from_camera.cu
//   optix_intersect_closest.cu / optix_intersect_shadow.cu
//   optix_shade_surface.cu / optix_shade_background.cu / optix_shade_shadow.cu
//   optix_shade_volume.cu  — spectral hero-λ PT (GPU fog: Woodcock)
//   optix_path_tail.cu     — remaining live paths in one megakernel
//   optix_hit_miss.cu
