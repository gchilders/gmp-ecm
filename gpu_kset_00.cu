/* gpu_kset_00.cu: GPU ECM kernels for solo L 1..8, distributed L 1..8
   (see gpu_kset.cuh).  Copyright 2026 Greg Childers; GPL v3 or later. */
#define GPU_KSET 00
#define GPU_KSET_SOLO_LO 1
#define GPU_KSET_SOLO_HI 8
#define GPU_KSET_DIST_LO 1
#define GPU_KSET_DIST_HI 8
#include "gpu_kset.cuh"
