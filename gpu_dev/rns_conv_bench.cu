/* rns_conv_bench.cu: time the conversions gpu_rns_from_z and gpu_rns_to_z
   alone for B polynomials of C = 2^LG coefficients (rows of length 2^LG).

   nvcc -O2 -arch=sm_86 -I.. -o rns_conv_bench rns_conv_bench.cu -lgmp
   ./rns_conv_bench BITS LG B */

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
  const int P = gpu_rns_primes (R), nl = gpu_rns_limbs (R);
  const size_t len = (size_t) 1 << lg, C = len;
  uint32_t *d, *z;
  CUDA_CHECK (cudaMalloc (&d, (size_t) B * P * len * 4));
  CUDA_CHECK (cudaMalloc (&z, (size_t) B * C * nl * 4));
  /* pseudo-random limbs, values below 2N */
  std::vector<uint32_t> h ((size_t) B * C * nl);
  uint32_t s = 12345;
  for (size_t k = 0; k < h.size (); k++)
    {
      s = s * 1664525u + 1013904223u;
      h[k] = k % nl == (size_t) nl - 1 ? 0 : s;
    }
  CUDA_CHECK (cudaMemcpy (z, h.data (), h.size () * 4,
                          cudaMemcpyHostToDevice));
  cudaEvent_t a, b;
  cudaEventCreate (&a);
  cudaEventCreate (&b);
  const double n = (double) B * C;
  for (int what = 0; what < 2; what++)
    for (int rep = 0; rep < 3; rep++)
      {
        cudaEventRecord (a);
        for (int i = 0; i < 10; i++)
          if (what == 0)
            gpu_rns_from_z (R, d, len, z, C, C, B);
          else
            gpu_rns_to_z (R, z, C, d, len, C, B);
        cudaEventRecord (b);
        cudaEventSynchronize (b);
        float ms;
        cudaEventElapsedTime (&ms, a, b);
        ms /= 10;
        if (rep == 2)
          printf ("%d bits (P = %d, NL = %d), %s 2^%d x %d: %.3f ms, "
                  "%.2f ns per coefficient, %.0f GMAC/s\n", bits, P, nl,
                  what ? "to_z  " : "from_z", lg, B, ms, ms * 1e6 / n,
                  n * nl * P / ms / 1e6);
      }
  /* pointwise products (gpu_rns_mul), B's rows one per polynomial and
     shared */
  uint32_t *d2;
  CUDA_CHECK (cudaMalloc (&d2, (size_t) B * P * len * 4));
  CUDA_CHECK (cudaMemset (d2, 0, (size_t) B * P * len * 4));
  for (int bbc = 0; bbc < 2; bbc++)
    for (int rep = 0; rep < 3; rep++)
      {
        cudaEventRecord (a);
        for (int i = 0; i < 10; i++)
          gpu_rns_mul (R, d, d, d2, len, B * P, len, bbc);
        cudaEventRecord (b);
        cudaEventSynchronize (b);
        float ms;
        cudaEventElapsedTime (&ms, a, b);
        ms /= 10;
        if (rep == 2)
          printf ("  mul%s 2^%d x %d rows: %.3f ms, %.0f GB/s\n",
                  bbc ? " (B shared)" : "", lg, B * P, ms,
                  (bbc ? 2.0 : 3.0) * B * P * len * 4 / ms / 1e6);
      }
  return 0;
}
