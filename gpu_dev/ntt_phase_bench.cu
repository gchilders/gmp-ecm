/* ntt_phase_bench.cu: time the NTT stage 1 kernel and split its time
   into phases (see NTT_PROF in gpu_ntt.cuh).

   nvcc -O2 -arch=sm_86 -I.. -o ntt_phase_bench ntt_phase_bench.cu -lgmp
   ./ntt_phase_bench BITS [LADDER_BITS [CURVES]]

   Runs LADDER_BITS (default 64) ladder steps on CURVES curves (default:
   what fills the GPU) for a random BITS-bit N.  The values are random, so
   only the timing means anything.  Phase shares are of the summed per-block
   clock64 time; the multiply count is 8 Montgomery multiplications per
   ladder step. */

#define NTT_PROFILE 1
#include "../gpu_ntt.cu"
#include <gmp.h>

int
main (int argc, char **argv)
{
  if (argc < 2)
    {
      fprintf (stderr, "usage: %s BITS [LADDER_BITS [CURVES]]\n", argv[0]);
      return 1;
    }
  const size_t bits = strtoul (argv[1], NULL, 10);
  const uint64_t nbits = argc > 2 ? strtoull (argv[2], NULL, 10) : 64;
  const int lg = gpu_ntt_lg_for_bits (bits);
  if (lg < 0)
    {
      fprintf (stderr, "too large\n");
      return 1;
    }
  gmp_randstate_t rs;
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 1);
  mpz_t N;
  mpz_init (N);
  mpz_urandomb (N, rs, bits);
  mpz_setbit (N, bits - 1);
  mpz_setbit (N, 0);

  gpu_ntt_plan *P = gpu_ntt_create (N, lg);
  const uint32_t curves = argc > 3 ? atoi (argv[3]) : gpu_ntt_fill_curves (P);
  const int W = P->W;

  /* values below N: random limbs under N's top limb */
  std::vector<uint32_t> data ((size_t) curves * 4 * W);
  for (size_t i = 0; i < data.size (); i++)
    data[i] = (uint32_t) gmp_urandomb_ui (rs, 32);
  for (uint32_t k = 0; k < curves * 4; k++)
    data[(size_t) k * W + W - 1] = 0;
  std::vector<uint32_t> sb ((nbits + 31) / 32);
  for (size_t i = 0; i < sb.size (); i++)
    sb[i] = (uint32_t) gmp_urandomb_ui (rs, 32);

  uint32_t *g_data, *g_s;
  CUDA_CHECK (cudaMalloc (&g_data, data.size () * 4));
  CUDA_CHECK (cudaMalloc (&g_s, sb.size () * 4));
  CUDA_CHECK (cudaMemcpy (g_s, sb.data (), sb.size () * 4, cudaMemcpyHostToDevice));

  int dev;
  cudaDeviceProp prop;
  CUDA_CHECK (cudaGetDevice (&dev));
  CUDA_CHECK (cudaGetDeviceProperties (&prop, dev));

  float ms = 0;
  unsigned long long prof[NTT_NPROF];
  for (int rep = 0; rep < 2; rep++)   /* the first is a warm-up */
    {
      CUDA_CHECK (cudaMemcpy (g_data, data.data (), data.size () * 4,
                              cudaMemcpyHostToDevice));
      memset (prof, 0, sizeof prof);
      CUDA_CHECK (cudaMemcpyToSymbol (ntt_prof_g, prof, sizeof prof));
      cudaEvent_t e0, e1;
      CUDA_CHECK (cudaEventCreate (&e0));
      CUDA_CHECK (cudaEventCreate (&e1));
      CUDA_CHECK (cudaEventRecord (e0));
      gpu_ntt_launch (P, curves, g_s, nbits, 0, nbits, g_data, 17, NULL, NULL);
      CUDA_CHECK (cudaEventRecord (e1));
      CUDA_CHECK (cudaEventSynchronize (e1));
      CUDA_CHECK (cudaEventElapsedTime (&ms, e0, e1));
      CUDA_CHECK (cudaMemcpyFromSymbol (prof, ntt_prof_g, sizeof prof));
    }

  int blocks_per_sm = 0;
  CUDA_CHECK (cudaOccupancyMaxActiveBlocksPerMultiprocessor (
                  &blocks_per_sm, ntt_stage1_kernel, P->threads, P->smem));
  const double mulmods = (double) curves * nbits * 8;
  printf ("%s: %zu-bit N, L = 2^%d, %u curves x %d threads, %zu B dynamic smem,"
          " %d block(s)/SM\n", prop.name, bits, lg, curves, P->threads,
          P->smem, blocks_per_sm);
  printf ("%llu ladder bits: %.1f ms, %.0f mulmod/s, %.1f us per curve-mulmod\n",
          (unsigned long long) nbits, ms, mulmods / (ms / 1e3),
          ms * 1e3 / (nbits * 8) * 1.0);
  static const char *name[NTT_NPROF] = { "forward+load", "last fwd+ptwise+first inv",
                                         "inverse+store", "crt+carry", "final",
                                         "add/sub/mul_ui" };
  unsigned long long tot = 0;
  for (int i = 0; i < NTT_NPROF; i++)
    tot += prof[i];
  for (int i = 0; i < NTT_NPROF; i++)
    printf ("  %-26s %5.1f%%  %8.0f cycles per mulmod\n", name[i],
            100.0 * prof[i] / tot, (double) prof[i] / mulmods);
  printf ("  %-26s        %8.0f cycles per mulmod (per block)\n", "total",
          (double) tot / mulmods);
  gpu_ntt_destroy (P);
  return 0;
}
