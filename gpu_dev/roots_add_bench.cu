/* roots_add_bench.cu: the arithmetic ceiling of the root generation
   (gpu_roots.cu) for one kernel shape: Montgomery products alone, and the
   Edwards addition rt_add, both with everything in registers, against
   the step kernel's time per addition (from GMPECM_ROOTS_PROF).

   nvcc -O2 -arch=sm_86 -I.. -DGPU_ROOTS_MAX_L=9 -o roots_add_bench \
     roots_add_bench.cu -lgmp
   ./roots_add_bench BITS TPI L INSTANCES ITERS */

#include "../gpu_roots.cu"
#include <gmp.h>

/* ITERS rounds of two independent product chains */
template<int L>
__global__ void
k_mulbench (uint32_t tpi, uint32_t *buf, const uint32_t *Ng, uint32_t np0,
            uint32_t iters)
{
  gm_group g;
  g.init (tpi);
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t off = g.lane * L;
  uint32_t N[L], N2[L], x[L], y[L];
  rt_modulus<L> (g, N, N2, Ng, off);
#pragma unroll
  for (int k = 0; k < L; k++)
    {
      x[k] = N[k] ^ (t * 0x9e3779b9u) >> 4;
      y[k] = N[k] ^ (t * 0x85ebca6bu) >> 4;
    }
  x[L - 1] = y[L - 1] = 0;
  for (uint32_t i = 0; i < iters; i++)
    {
      gm_mul<L, false> (g, x, x, y, N, np0);
      gm_mul<L, false> (g, y, y, x, N, np0);
    }
#pragma unroll
  for (int k = 0; k < L; k++)
    buf[(size_t) t * L + k] = x[k] ^ y[k];
}

/* ITERS additions p = p + q, points in registers */
template<int L>
__global__ void
k_addbench (uint32_t tpi, uint32_t *buf, const uint32_t *cst,
            const uint32_t *Ng, uint32_t np0, uint32_t iters)
{
  gm_group g;
  g.init (tpi);
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t n = g.tpi * L, off = g.lane * L;
  uint32_t N[L], N2[L];
  rt_modulus<L> (g, N, N2, Ng, off);
  rt_pt<L> p, q;
  rt_cload_pt<L> (p, cst, RC_PX, n, off);
  rt_copy_pt<L> (q, p);
  for (uint32_t i = 0; i < iters; i++)
    rt_add<L> (g, p, p, q, cst, n, off, N, N2, np0);
#pragma unroll
  for (int k = 0; k < L; k++)
    buf[(size_t) t * L + k] = p.X[k] ^ p.Y[k] ^ p.Z[k] ^ p.T[k];
}

int
main (int argc, char **argv)
{
  const int bits = atoi (argv[1]);
  const uint32_t tpi = atoi (argv[2]), L = atoi (argv[3]);
  const uint32_t inst = atoi (argv[4]), iters = atoi (argv[5]);
  assert (L == 5 || L == 9 || L == 12 || L == 17);
  mpz_t N;
  mpz_init (N);
  gmp_randstate_t rs;
  gmp_randinit_default (rs);
  mpz_urandomb (N, rs, bits);
  mpz_setbit (N, bits - 1);
  mpz_setbit (N, 0);
  const uint32_t n = tpi * L;
  std::vector<uint32_t> h (RC_COUNT * n);
  gpu_to_limbs (h.data (), n, N);
  uint32_t *dN = rt_upload (std::vector<uint32_t> (h.begin (), h.begin () + n));
  for (int i = 0; i < RC_COUNT; i++)
    {
      mpz_t r;
      mpz_init (r);
      mpz_urandomm (r, rs, N);
      gpu_to_limbs (&h[i * n], n, r);
      mpz_clear (r);
    }
  uint32_t *dc = rt_upload (h), *buf;
  const uint32_t threads = inst * tpi;
  CUDA_CHECK (cudaMalloc (&buf, (size_t) threads * L * 4));
  const uint32_t np0 = gpu_find_np0 (N);
  const unsigned blocks = (threads + GPU_TPB - 1) / GPU_TPB;
  cudaEvent_t a, b;
  cudaEventCreate (&a);
  cudaEventCreate (&b);
  float ms;
  for (int rep = 0; rep < 2; rep++)
    {
      cudaEventRecord (a);
      if (L == 9)
        k_mulbench<9><<<blocks, GPU_TPB>>> (tpi, buf, dN, np0, iters);
      else if (L == 17)
        k_mulbench<17><<<blocks, GPU_TPB>>> (tpi, buf, dN, np0, iters);
      else if (L == 12)
        k_mulbench<12><<<blocks, GPU_TPB>>> (tpi, buf, dN, np0, iters);
      else
        k_mulbench<5><<<blocks, GPU_TPB>>> (tpi, buf, dN, np0, iters);
      cudaEventRecord (b);
      cudaEventSynchronize (b);
      cudaEventElapsedTime (&ms, a, b);
    }
  CUDA_CHECK (cudaGetLastError ());
  printf ("tpi %u L %u, %u instances: product %.3f ns (whole GPU)\n", tpi, L,
          inst, ms * 1e6 / (2.0 * iters * inst));
  for (int rep = 0; rep < 2; rep++)
    {
      cudaEventRecord (a);
      if (L == 9)
        k_addbench<9><<<blocks, GPU_TPB>>> (tpi, buf, dc, dN, np0, iters);
      else if (L == 17)
        k_addbench<17><<<blocks, GPU_TPB>>> (tpi, buf, dc, dN, np0, iters);
      else if (L == 12)
        k_addbench<12><<<blocks, GPU_TPB>>> (tpi, buf, dc, dN, np0, iters);
      else
        k_addbench<5><<<blocks, GPU_TPB>>> (tpi, buf, dc, dN, np0, iters);
      cudaEventRecord (b);
      cudaEventSynchronize (b);
      cudaEventElapsedTime (&ms, a, b);
    }
  CUDA_CHECK (cudaGetLastError ());
  printf ("tpi %u L %u, %u instances: addition %.3f ns = %.3f ns per product "
          "(11 products)\n", tpi, L, inst, ms * 1e6 / ((double) iters * inst),
          ms * 1e6 / ((double) iters * inst) / 11);
  return 0;
}
