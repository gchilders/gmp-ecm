/* gpu_kset_16.cu: GPU ECM kernels for solo L 21..22 (even only), no
   distributed L (see gpu_kset.cuh).  Copyright 2026 Greg Childers; GPL v3
   or later. */
#define GPU_KSET 16
#define GPU_KSET_SOLO_LO 21
#define GPU_KSET_SOLO_HI 22
#define GPU_KSET_DIST_LO 1
#define GPU_KSET_DIST_HI 0
#include "gpu_kset.cuh"
