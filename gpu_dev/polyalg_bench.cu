/* polyalg_bench.cu: time the polynomial part of one curve's stage 2
   (gpu_polyalg.cu) on random roots: F's tree, its inverse, K blocks of
   G = prod (x + g) and H <- (G - F) H mod F, then the transposed
   evaluation.

   nvcc -O2 -arch=sm_86 -I.. -o polyalg_bench polyalg_bench.cu -lgmp
   ./polyalg_bench BITS LOG2_DF BLOCKS */

#include "../gpu_rns.cu"
#include "../gpu_polyalg.cu"
#include <gmp.h>
#include <chrono>

static double
now ()
{
  CUDA_CHECK (cudaDeviceSynchronize ());
  return std::chrono::duration<double> (
             std::chrono::steady_clock::now ().time_since_epoch ()).count ();
}

int
main (int argc, char **argv)
{
  const int bits = argc > 1 ? atoi (argv[1]) : 1024;
  const int l = argc > 2 ? atoi (argv[2]) : 14;
  const int k = argc > 3 ? atoi (argv[3]) : 2;
  const size_t n = (size_t) 1 << l;
  gmp_randstate_t rs;
  gmp_randinit_default (rs);
  mpz_t N, x;
  mpz_inits (N, x, NULL);
  mpz_urandomb (N, rs, bits);
  mpz_setbit (N, bits - 1);
  mpz_setbit (N, 0);
  gpu_rns *R = gpu_rns_create (N, l + 1);
  const int nl = gpu_rns_limbs (R);
  gpu_polyalg *A = gpu_polyalg_create (R, n);
  std::vector<uint32_t> h (n * nl, 0);
  for (size_t i = 0; i < n; i++)
    {
      mpz_urandomm (x, rs, N);
      size_t cnt;
      mpz_export (&h[i * nl], &cnt, -1, 4, 0, 0, x);
    }
  uint32_t *roots, *F, *I, *G, *H, *V;
  std::vector<uint32_t *> tree (l);
  CUDA_CHECK (cudaMalloc (&roots, n * nl * 4));
  CUDA_CHECK (cudaMalloc (&F, n * nl * 4));
  CUDA_CHECK (cudaMalloc (&I, n * nl * 4));
  CUDA_CHECK (cudaMalloc (&G, n * nl * 4));
  CUDA_CHECK (cudaMalloc (&H, n * nl * 4));
  CUDA_CHECK (cudaMalloc (&V, n * nl * 4));
  for (auto &p : tree)
    CUDA_CHECK (cudaMalloc (&p, n * nl * 4));
  CUDA_CHECK (cudaMemcpy (roots, h.data (), h.size () * 4, cudaMemcpyHostToDevice));

  for (int rep = 0; rep < 2; rep++)
    {
      double t0 = now ();
      gpu_poly_from_roots (A, F, tree.data (), roots);
      double t1 = now ();
      gpu_poly_inverse (A, I, F);
      gpu_poly_prepare (A, F, I);
      double t2 = now ();
      double tg = 0, tm = 0;
      for (int j = 0; j < k; j++)
        {
          double a = now ();
          gpu_poly_from_roots (A, G, NULL, roots);
          double b = now ();
          gpu_poly_mulmod (A, H, G, j == 0);
          tg += b - a;
          tm += now () - b;
        }
      double t3 = now ();
      gpu_poly_tpeval (A, V, H, tree.data ());
      double t4 = now ();
      if (rep == 1)
        printf ("%d bits, dF = 2^%d, %d blocks (%d primes): tree %.1f ms, "
                "inverse+prep %.1f ms, G %.1f ms, G H mod F %.1f ms, "
                "tpeval %.1f ms; total %.1f ms\n", bits, l, k,
                gpu_rns_primes (R), (t1 - t0) * 1e3, (t2 - t1) * 1e3,
                tg * 1e3, tm * 1e3, (t4 - t3) * 1e3, (t4 - t0) * 1e3);
    }
  return 0;
}
