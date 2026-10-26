/* nmul_test.cu: check the transform products of gpu_nmul.cuh against GMP.

   nvcc -O2 -arch=sm_86 -I.. -o nmul_test nmul_test.cu -lgmp
   ./nmul_test [rounds]

   For every transform length 2^8 .. 2^12 and digit size (the smallest and
   largest allowed, and the one nm_shape picks for a few N sizes), a chain
   of rounds per instance, each value fed to the next round so that digit
   sizes are the ones products really produce:
     h = x - y + z            (a difference and a sum of three as operands)
     x' = x h / R             nm_mul (one hat)
     y' = x' y / R            nm_mulhh (two hats)
     z' = sum_k (x, y, h, x')_k (y, h, x', x)_k / R   nm_acc (four products)
     and x, y, z' to limbs (nm_to_limbs), and back (nm_from_limbs)
   Every value is checked mod N and for its range: products in
   (-0.6N, 2.6N), limbs in [0, 2N).
   Then sums of 16 products as the Z/N tree and evaluation kernels make
   them (NT_ZMAX terms, the most a transform-domain sum takes there):
     sum16  = REDC (sum a_k (b_k R)) = sum a_k b_k   (b_k R by nm_mul with R^2)
     sum16c = REDC (sum a_k b_k)                     (both canonical)
   with a_k, b_k in [0, 2N); instance 0 takes 2N - 1 for all of them, and
   the stress runs an N for which 2N - 1 has nearly every digit B - 1, so
   that the convolution coefficients come close to their bound at the
   largest digit size. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cudacommon.h"
#include "gpu_nmul.cuh"

#define NV 8      /* numbers per instance per round: x y z h x' y' z' back */

template<int LG>
__global__ void __launch_bounds__ ((1 << LG) / 8)
test_kernel (const NM_GRID_CONST nm_ctx c, int rounds, int32_t *V,
             uint32_t *H, uint32_t *limbs, uint32_t nl)
{
  constexpr int D = 1 << LG;
  const size_t inst = blockIdx.x;
  uint32_t *hh = H + inst * 4 * 4 * D;    /* hats of x, y, h, x' */
  for (int r = 0; r < rounds; r++)
    {
      int32_t *in = V + ((inst * (rounds + 1) + r) * NV) * D;
      int32_t *o = V + ((inst * (rounds + 1) + r + 1) * NV) * D;
      int32_t *x = in, *y = in + D, *z = in + 2 * D, *h = in + 3 * D;
      nm_sub<LG> (h, x, y);
      nm_add<LG> (h, h, z);
      nm_hat<LG> (c, hh, h);
      nm_mul<LG> (c, in + 4 * D, x, hh);             /* x' = x h */
      nm_hat<LG> (c, hh + 4 * D, in + 4 * D);
      nm_hat<LG> (c, hh + 8 * D, y);
      nm_mulhh<LG> (c, in + 5 * D, hh + 4 * D, hh + 8 * D);   /* y' = x' y */
      nm_hat<LG> (c, hh + 12 * D, x);
      uint32_t acc[2][2][8];
      nm_acc_clear<LG> (acc);
      nm_acc_add<LG> (c, acc, hh + 12 * D, hh + 8 * D);   /* x y */
      nm_acc_add<LG> (c, acc, hh + 8 * D, hh);            /* y h */
      nm_acc_add<LG> (c, acc, hh, hh + 4 * D);            /* h x' */
      nm_acc_add<LG> (c, acc, hh + 4 * D, hh + 12 * D);   /* x' x */
      nm_acc_redc<LG> (c, in + 6 * D, acc);
      int32_t v[8];
      nm_load<LG> (v, in + 6 * D);
      uint32_t *lb = limbs + (inst * rounds + r) * nl;
      nm_to_limbs<LG> (c, lb, nl, v);
      nm_from_limbs<LG> (c, in + 7 * D, lb, nl);
      /* next round: x', y', z' */
      nm_copy<LG> (o, in + 4 * D);
      nm_copy<LG> (o + D, in + 5 * D);
      nm_copy<LG> (o + 2 * D, in + 6 * D);
    }
}

#define NS 16     /* products per sum */

template<int LG>
__global__ void __launch_bounds__ ((1 << LG) / 8)
sum16_kernel (const NM_GRID_CONST nm_ctx c, const int32_t *X, uint32_t *H,
              int32_t *S, int32_t *Out)
{
  constexpr int D = 1 << LG;
  const size_t inst = blockIdx.x;
  const int32_t *a = X + inst * 2 * NS * D, *b = a + NS * D;
  uint32_t *ha = H + inst * 2 * NS * 4 * D, *hb = ha + NS * 4 * D;
  int32_t *s = S + inst * NS * D, *out = Out + inst * 2 * D;
  for (int k = 0; k < NS; k++)
    {
      nm_hat<LG> (c, ha + k * 4 * D, a + k * D);
      nm_mul<LG> (c, s + k * D, b + k * D, c.hR2);      /* b R */
      nm_hat<LG> (c, hb + k * 4 * D, s + k * D);
    }
  uint32_t acc[2][2][8];
  nm_acc_clear<LG> (acc);
  for (int k = 0; k < NS; k++)
    nm_acc_add<LG> (c, acc, ha + k * 4 * D, hb + k * 4 * D);
  nm_acc_redc<LG> (c, out, acc);
  for (int k = 0; k < NS; k++)
    nm_hat<LG> (c, hb + k * 4 * D, b + k * D);
  nm_acc_clear<LG> (acc);
  for (int k = 0; k < NS; k++)
    nm_acc_add<LG> (c, acc, ha + k * 4 * D, hb + k * 4 * D);
  nm_acc_redc<LG> (c, out + D, acc);
}

static int fails = 0;

static void
check (const char *what, const mpz_t got, const mpz_t want, const mpz_t N,
       double lo, double hi, int lg, int b, int inst, int r)
{
  mpz_t a, e, l, u;
  mpz_inits (a, e, l, u, NULL);
  mpz_mod (a, got, N);
  mpz_mod (e, want, N);
  /* range in units of N/1000 */
  mpz_mul_si (l, N, (long) (lo * 1000));
  mpz_mul_si (u, N, (long) (hi * 1000));
  mpz_t g1000;
  mpz_init (g1000);
  mpz_mul_ui (g1000, got, 1000);
  const bool val = mpz_cmp (a, e) == 0;
  const bool rng = mpz_cmp (g1000, l) >= 0 && mpz_cmp (g1000, u) < 0;
  if (!val || !rng)
    {
      if (fails < 10)
        printf ("  lg %d b %d inst %d round %d %s: %s\n", lg, b, inst, r, what,
                !val ? "WRONG VALUE" : "out of range");
      fails++;
    }
  mpz_clears (a, e, l, u, g1000, NULL);
}

/* STRESS: N = 2^BITS - c (c small and odd), so that 2N - 1 has all its
   digits B - 1 but the lowest and the top one */
template<int LG>
static void
run (int bits, int b, int rounds, gmp_randstate_t rs, bool stress = false)
{
  constexpr int D = 1 << LG;
  mpz_t N, M, t;
  mpz_inits (N, M, t, NULL);
  mpz_setbit (M, (mp_bitcnt_t) b * D);
  mpz_sub_ui (M, M, 1);
  unsigned long cs = 1;
  do
    {
      if (stress)
        {
          mpz_set_ui (N, 0);
          mpz_setbit (N, bits);
          mpz_sub_ui (N, N, cs);
          cs += 2;
        }
      else
        {
          mpz_urandomb (N, rs, bits);
          mpz_setbit (N, bits - 1);
          mpz_setbit (N, 0);
        }
      mpz_gcd (t, N, M);
    }
  while (mpz_cmp_ui (t, 1) != 0);
  nm_host h;
  if (!nm_ctx_create (h, N, LG, b))
    abort ();
  const int inst = 64;
  const uint32_t nl = (bits + 1 + 31) / 32 + 1;
  std::vector<int32_t> V ((size_t) inst * (rounds + 1) * NV * D, 0), dg;
  mpz_t x;
  mpz_init (x);
  for (int i = 0; i < inst; i++)
    for (int k = 0; k < 3; k++)
      {
        /* x, y, z in [0, 2N) (the first round), canonical digits */
        mpz_mul_2exp (t, N, 1);
        mpz_urandomm (x, rs, t);
        nm_digits (dg, x, D, b);
        std::copy (dg.begin (), dg.end (), V.begin () + ((size_t) i * (rounds + 1) * NV + k) * D);
      }
  int32_t *dV;
  uint32_t *dH, *dL;
  CUDA_CHECK (cudaMalloc (&dV, V.size () * 4));
  CUDA_CHECK (cudaMemcpy (dV, V.data (), V.size () * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK (cudaMalloc (&dH, (size_t) inst * 16 * D * 4));
  CUDA_CHECK (cudaMalloc (&dL, (size_t) inst * rounds * nl * 4));
  const size_t sm = nm_smem_bytes (LG);
  CUDA_CHECK (cudaFuncSetAttribute (test_kernel<LG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm));
  test_kernel<LG><<<inst, D / 8, sm>>> (h.c, rounds, dV, dH, dL, nl);
  CUDA_CHECK (cudaDeviceSynchronize ());
  CUDA_CHECK (cudaGetLastError ());
  CUDA_CHECK (cudaMemcpy (V.data (), dV, V.size () * 4, cudaMemcpyDeviceToHost));
  std::vector<uint32_t> L ((size_t) inst * rounds * nl);
  CUDA_CHECK (cudaMemcpy (L.data (), dL, L.size () * 4, cudaMemcpyDeviceToHost));

  mpz_t v[NV], Ri, e, s;
  for (int k = 0; k < NV; k++)
    mpz_init (v[k]);
  mpz_inits (Ri, e, s, NULL);
  mpz_invert (Ri, h.R, N);
  int maxd = 0;
  for (int i = 0; i < inst; i++)
    for (int r = 0; r < rounds; r++)
      {
        const int32_t *in = &V[((size_t) i * (rounds + 1) + r) * NV * D];
        for (int k = 0; k < NV; k++)
          nm_value (v[k], in + (size_t) k * D, D, b);
        for (int k = 0; k < 7 * D; k++)
          maxd = std::max (maxd, abs (in[k]));
        /* h = x - y + z */
        mpz_sub (e, v[0], v[1]);
        mpz_add (e, e, v[2]);
        check ("h", v[3], e, N, -8, 8, LG, b, i, r);
        mpz_mul (e, v[0], v[3]);
        mpz_mul (e, e, Ri);
        check ("mul", v[4], e, N, -0.6, 2.6, LG, b, i, r);
        mpz_mul (e, v[4], v[1]);
        mpz_mul (e, e, Ri);
        check ("mulhh", v[5], e, N, -0.6, 2.6, LG, b, i, r);
        mpz_mul (e, v[0], v[1]);
        mpz_mul (s, v[1], v[3]);
        mpz_add (e, e, s);
        mpz_mul (s, v[3], v[4]);
        mpz_add (e, e, s);
        mpz_mul (s, v[4], v[0]);
        mpz_add (e, e, s);
        mpz_mul (e, e, Ri);
        check ("acc", v[6], e, N, -0.6, 2.6, LG, b, i, r);
        mpz_import (s, nl, -1, 4, 0, 0, &L[((size_t) i * rounds + r) * nl]);
        check ("limbs", s, v[6], N, 0, 2, LG, b, i, r);
        check ("from limbs", v[7], s, N, 0, 2, LG, b, i, r);
        if (mpz_cmp (v[7], s) != 0)
          {
            if (fails < 10)
              printf ("  lg %d b %d from limbs: not the same number\n", LG, b);
            fails++;
          }
      }

  /* sums of NS products */
  std::vector<int32_t> X ((size_t) inst * 2 * NS * D);
  std::vector<mpz_t *> xv (inst);
  mpz_t two;
  mpz_init (two);
  mpz_mul_2exp (two, N, 1);
  for (int i = 0; i < inst; i++)
    {
      xv[i] = new mpz_t[2 * NS];
      for (int k = 0; k < 2 * NS; k++)
        {
          mpz_init (xv[i][k]);
          if (i == 0)
            mpz_sub_ui (xv[i][k], two, 1);
          else
            mpz_urandomm (xv[i][k], rs, two);
          nm_digits (dg, xv[i][k], D, b);
          std::copy (dg.begin (), dg.end (), X.begin () + ((size_t) i * 2 * NS + k) * D);
        }
    }
  int32_t *dX, *dS, *dO;
  CUDA_CHECK (cudaMalloc (&dX, X.size () * 4));
  CUDA_CHECK (cudaMemcpy (dX, X.data (), X.size () * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK (cudaFree (dH));
  CUDA_CHECK (cudaMalloc (&dH, (size_t) inst * 2 * NS * 4 * D * 4));
  CUDA_CHECK (cudaMalloc (&dS, (size_t) inst * NS * D * 4));
  CUDA_CHECK (cudaMalloc (&dO, (size_t) inst * 2 * D * 4));
  CUDA_CHECK (cudaFuncSetAttribute (sum16_kernel<LG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm));
  sum16_kernel<LG><<<inst, D / 8, sm>>> (h.c, dX, dH, dS, dO);
  CUDA_CHECK (cudaDeviceSynchronize ());
  CUDA_CHECK (cudaGetLastError ());
  std::vector<int32_t> O ((size_t) inst * 2 * D);
  CUDA_CHECK (cudaMemcpy (O.data (), dO, O.size () * 4, cudaMemcpyDeviceToHost));
  for (int i = 0; i < inst; i++)
    {
      mpz_set_ui (e, 0);
      for (int k = 0; k < NS; k++)
        mpz_addmul (e, xv[i][k], xv[i][NS + k]);
      nm_value (v[0], &O[(size_t) i * 2 * D], D, b);
      check ("sum16", v[0], e, N, -0.6, 2.6, LG, b, i, 0);
      mpz_mul (e, e, Ri);
      nm_value (v[1], &O[(size_t) i * 2 * D + D], D, b);
      check ("sum16c", v[1], e, N, -0.6, 2.6, LG, b, i, 0);
      for (int k = 0; k < 2 * NS; k++)
        mpz_clear (xv[i][k]);
      delete[] xv[i];
    }
  mpz_clear (two);
  cudaFree (dX);
  cudaFree (dS);
  cudaFree (dO);

  printf ("lg %2d b %2d (%6d-bit N%s): %d rounds x %d, max |digit| 2^%.2f, "
          "sums of %d  %s\n", LG, b, bits, stress ? ", stress" : "", rounds,
          inst, log2 ((double) maxd), NS, fails ? "FAILURES" : "ok");
  fflush (stdout);
  for (int k = 0; k < NV; k++)
    mpz_clear (v[k]);
  mpz_clears (N, M, t, x, Ri, e, s, NULL);
  cudaFree (dV);
  cudaFree (dH);
  cudaFree (dL);
  nm_ctx_free (h);
}

template<int LG>
static void
sizes (int rounds, gmp_randstate_t rs)
{
  const int D = 1 << LG;
  /* the smallest b (at a small N) and the largest (at the largest N) */
  run<LG> (D * 4, NM_BMIN (LG), rounds, rs);
  run<LG> (D * NM_BMAX (LG) - NM_SPARE_BITS, NM_BMAX (LG), rounds, rs);
  run<LG> (D * NM_BMAX (LG) - NM_SPARE_BITS, NM_BMAX (LG), rounds, rs, true);
  int lg, b;
  if (nm_shape ((size_t) D * 16, NM_MIN_LG, NM_MAX_LG, &lg, &b) && lg == LG)
    run<LG> (D * 16, b, rounds, rs);
}

int
main (int argc, char **argv)
{
  const int rounds = argc > 1 ? atoi (argv[1]) : 6;
  gmp_randstate_t rs;
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 5);
  sizes<8> (rounds, rs);
  sizes<9> (rounds, rs);
  sizes<10> (rounds, rs);
  sizes<11> (rounds, rs);
  sizes<12> (rounds, rs);
  printf ("%s\n", fails ? "FAILED" : "ALL OK");
  return fails != 0;
}
