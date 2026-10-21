/* wide_mul_bench.cu: throughput of the wide-group Montgomery product
   (gpu_mont.cuh, 64 or 128 threads per curve) in its two forms ("block":
   lane 0 computes the q of K steps ahead; "lead": warp 0 runs ahead) with
   K = 2 or 4 steps per barrier, and a check that all give the same
   results.  The best form per L goes into gm_wide_lead_better.

   nvcc -O2 -arch=sm_86 -I.. -o wide_mul_bench wide_mul_bench.cu -lgmp
   ./wide_mul_bench [products per curve]
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <gmp.h>
#include "gpu_mont.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf (stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString (e_)); exit (1); } } while (0)

/* x <- x * b / R, ITERS times, per curve */
template<int L, int FORM, int K>
__global__ void __launch_bounds__ (128)
bench_kernel (uint32_t tpi, int iters, const uint32_t *Ng, uint32_t np0,
              uint32_t *data)
{
  __shared__ uint32_t smem[GM_WSHARED_WORDS (128, L)];
  gm_wgroup g;
  g.init (tpi, smem, L);
  const uint32_t inst = (blockIdx.x * blockDim.x + threadIdx.x) / tpi;
  const uint32_t n = tpi * L, off = g.lane * L;
  uint32_t x[L], b[L], N[L];
  for (int k = 0; k < L; k++)
    {
      x[k] = data[(size_t) inst * n + off + k];
      b[k] = x[k];
      N[k] = Ng[off + k];
    }
  for (int it = 0; it < iters; it++)
    if (FORM)
      gm_mul_wide_lead<L, K> (g, x, x, b, N, np0);
    else
      gm_mul_wide_block<L, K> (g, x, x, b, N, np0);
  for (int k = 0; k < L; k++)
    data[(size_t) inst * n + off + k] = x[k];
}

static int bad = 0;

template<int L, int FORM, int K>
static double
run (uint32_t tpi, int iters, uint32_t curves, const std::vector<uint32_t> &N,
     uint32_t np0, const std::vector<uint32_t> &init, std::vector<uint32_t> &out)
{
  const uint32_t n = tpi * L;
  uint32_t *gN, *gD;
  CK (cudaMalloc (&gN, n * 4));
  CK (cudaMalloc (&gD, (size_t) curves * n * 4));
  CK (cudaMemcpy (gN, N.data (), n * 4, cudaMemcpyHostToDevice));
  CK (cudaMemcpy (gD, init.data (), (size_t) curves * n * 4, cudaMemcpyHostToDevice));
  const uint32_t blocks = curves * tpi / 128;
  bench_kernel<L, FORM, K><<<blocks, 128>>> (tpi, 1, gN, np0, gD);   /* warm up */
  CK (cudaMemcpy (gD, init.data (), (size_t) curves * n * 4, cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1;
  CK (cudaEventCreate (&e0));
  CK (cudaEventCreate (&e1));
  CK (cudaEventRecord (e0));
  bench_kernel<L, FORM, K><<<blocks, 128>>> (tpi, iters, gN, np0, gD);
  CK (cudaEventRecord (e1));
  CK (cudaEventSynchronize (e1));
  CK (cudaGetLastError ());
  float ms;
  CK (cudaEventElapsedTime (&ms, e0, e1));
  out.resize ((size_t) curves * n);
  CK (cudaMemcpy (out.data (), gD, out.size () * 4, cudaMemcpyDeviceToHost));
  cudaFree (gN);
  cudaFree (gD);
  return (double) curves * iters / (ms / 1e3);
}

template<int L>
static void
shape (uint32_t tpi, int iters)
{
  const uint32_t n = tpi * L;
  gmp_randstate_t rs;
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 7 + n);
  mpz_t Nz, t;
  mpz_inits (Nz, t, NULL);
  mpz_urandomb (Nz, rs, 32 * n - 4);
  mpz_setbit (Nz, 32 * n - 5);
  mpz_setbit (Nz, 0);
  std::vector<uint32_t> N (n, 0);
  mpz_export (N.data (), NULL, -1, 4, 0, 0, Nz);
  mpz_set_ui (t, 0);
  mpz_setbit (t, 32);
  mpz_invert (t, Nz, t);
  const uint32_t np0 = - (uint32_t) mpz_get_ui (t);

  int dev, sms, blocks = 0;
  CK (cudaGetDevice (&dev));
  CK (cudaDeviceGetAttribute (&sms, cudaDevAttrMultiProcessorCount, dev));
  CK (cudaOccupancyMaxActiveBlocksPerMultiprocessor (&blocks, bench_kernel<L, 0, 2>, 128, 0));
  const uint32_t curves = blocks * sms * (128 / tpi);

  std::vector<uint32_t> init ((size_t) curves * n, 0);
  for (uint32_t c = 0; c < curves; c++)
    {
      mpz_urandomm (t, rs, Nz);
      mpz_export (&init[(size_t) c * n], NULL, -1, 4, 0, 0, t);
    }
  std::vector<uint32_t> o1, o2, o3, o4;
  double b2 = run<L, 0, 2> (tpi, iters, curves, N, np0, init, o1);
  double b4 = run<L, 0, 4> (tpi, iters, curves, N, np0, init, o2);
  double l2 = run<L, 1, 2> (tpi, iters, curves, N, np0, init, o3);
  double l4 = run<L, 1, 4> (tpi, iters, curves, N, np0, init, o4);
  const bool same = o1 == o2 && o1 == o3 && o1 == o4;
  bad += !same;
  const double best = std::max (std::max (b2, b4), std::max (l2, l4));
  printf ("TPI=%3u L=%2d (%5u bits) %4u curves  products/s: block K=2 %8.0f  "
          "K=4 %8.0f  lead K=2 %8.0f  K=4 %8.0f  best: %s  %s\n", tpi, L,
          32 * n, curves, b2, b4, l2, l4,
          best == b2 ? "block 2" : best == b4 ? "block 4" : best == l2 ? "lead 2" : "lead 4",
          same ? "same" : "DIFFERENT");
  fflush (stdout);
  mpz_clears (Nz, t, NULL);
  gmp_randclear (rs);
}

int
main (int argc, char **argv)
{
  int iters = argc > 1 ? atoi (argv[1]) : 2000;
  shape<8> (128, iters);
  shape<10> (128, iters);
  shape<12> (64, iters);
  shape<12> (128, iters);
  shape<13> (64, iters);
  shape<14> (64, iters);
  shape<14> (128, iters);
  shape<16> (64, iters);
  shape<16> (128, iters);
  shape<18> (64, iters);
  shape<18> (128, iters);
  shape<20> (64, iters);
  shape<20> (128, iters);
  shape<22> (64, iters);
  shape<22> (128, iters);
  shape<24> (64, iters);
  shape<24> (128, iters);
  printf ("%s\n", bad ? "MISMATCHES" : "all K agree");
  return bad != 0;
}
