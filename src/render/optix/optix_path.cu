// Iray wavefront OptiX modules (4λ spectral PT):
//   init / intersect / shade — thin kernels, no payload (CH writes GpuHit)
//   path_tail — remainder of the state machine in a separate pipeline
//   hit_miss  — closest-hit / miss write wavefront state
