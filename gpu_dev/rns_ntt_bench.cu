/* rns_ntt_bench.cu: time gpu_rns_ntt alone (forward + inverse) for
   ROWS = B P rows of length 2^LG, and report the effective bandwidth.

   nvcc -O2 -arch=sm_86 -I.. -o rns_ntt_bench rns_ntt_bench.cu -lgmp
   ./rns_ntt_bench BITS LG B */

#include "../gpu_rns.cu"
#include <gmp.h>

int
main (int argc, char **argv)
{
  const int bits = atoi (argv[1]), lg = atoi (argv[2]), B = atoi (argv[3]);
  mpz_t N;
  mpz_init (N);
  mpz_ui_pow_ui (N, 2, bits);
  mpz_sub_ui (N, N, 1);
  gpu_rns *R = gpu_rns_create (N, lg, 1);   /* as gpu_s2poly.cu */
  const int P = gpu_rns_primes (R), rows = B * P;
  const size_t len = (size_t) 1 << lg;
  uint32_t *d;
  CUDA_CHECK (cudaMalloc (&d, (size_t) rows * len * 4));
  CUDA_CHECK (cudaMemset (d, 0, (size_t) rows * len * 4));
  cudaEvent_t a, b;
  cudaEventCreate (&a);
  cudaEventCreate (&b);
  for (int rep = 0; rep < 3; rep++)
    {
      cudaEventRecord (a);
      for (int i = 0; i < 10; i++)
        {
          gpu_rns_ntt (R, d, len, rows, lg, 0, 0);
          gpu_rns_ntt (R, d, len, rows, lg, 1, 0);
        }
      cudaEventRecord (b);
      cudaEventSynchronize (b);
      float ms;
      cudaEventElapsedTime (&ms, a, b);
      ms /= 20;
      if (rep == 2)
        printf ("%d bits (P = %d), 2^%d x %d rows: %.3f ms per transform, "
                "%.0f GB/s (read + write)\n", bits, P, lg, rows, ms,
                2.0 * rows * len * 4 / ms / 1e6);
    }
  return 0;
}
