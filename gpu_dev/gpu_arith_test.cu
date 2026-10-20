/* gpu_arith_test.cu: check the gpu_mont.cuh primitives against GMP.

   nvcc -O2 -arch=sm_89 -o gpu_arith_test gpu_arith_test.cu -lgmp
   ./gpu_arith_test [iterations]

   Every op is run for many (TPI, L) shapes on random and adversarial
   inputs (all-ones / all-zero limbs, N = 2^k +- 1, N close to the top of
   the allowed range), and checked for value mod N and for its output bound.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <gmp.h>
#include "gpu_mont.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf (stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString (e_)); exit (1); } } while (0)

enum { OP_MUL, OP_SQR, OP_ADD, OP_SUB, OP_MULUI, OP_DBL, OP_COUNT };
static const char *op_name[] = { "mul", "sqr", "add", "sub", "mul_ui", "dbl" };

template<int L, bool SOLO>
__global__ void
test_kernel (int op, uint32_t tpi, uint32_t count, const uint32_t *A,
             const uint32_t *B, const uint32_t *Ng, uint32_t np0,
             const uint32_t *D, uint32_t *out)
{
  gm_group g;
  g.init (SOLO ? 1 : tpi);
  uint32_t inst = (blockIdx.x * blockDim.x + threadIdx.x) / g.tpi;
  uint32_t n = g.tpi * L;
  uint32_t ci = inst < count ? inst : count - 1;  /* keep warps whole */
  uint32_t off = g.lane * L;
  uint32_t a[L], b[L], N[L], N2[L], r[L];
  for (int k = 0; k < L; k++)
    {
      a[k] = A[(size_t) ci * n + off + k];
      b[k] = B[(size_t) ci * n + off + k];
      N[k] = Ng[(size_t) ci * n + off + k];
    }
  gm_dbl<L, SOLO> (g, N2, N);
  switch (op)
    {
    case OP_MUL: gm_mul<L, SOLO> (g, r, a, b, N, np0 ? np0 : D[2 * ci + 1]); break;
    case OP_SQR: gm_sqr<L, SOLO> (g, r, a, N, np0 ? np0 : D[2 * ci + 1]); break;
    case OP_ADD: gm_add<L, SOLO> (g, r, a, b); break;
    case OP_SUB: gm_sub<L, SOLO> (g, r, a, b, N2); break;
    case OP_MULUI: gm_mul_ui<L, SOLO> (g, r, a, D[2 * ci], N, np0 ? np0 : D[2 * ci + 1]); break;
    default: gm_dbl<L, SOLO> (g, r, a); break;
    }
  if (inst < count)
    for (int k = 0; k < L; k++)
      out[(size_t) inst * n + off + k] = r[k];
}

static gmp_randstate_t rs;

static void
to_limbs (uint32_t *x, uint32_t n, const mpz_t s)
{
  size_t w = 0;
  if (mpz_sizeinbase (s, 2) > 32 * (size_t) n || mpz_sgn (s) < 0)
    { gmp_fprintf (stderr, "to_limbs overflow %Zx\n", s); exit (1); }
  mpz_export (x, &w, -1, 4, 0, 0, s);
  while (w < n) x[w++] = 0;
}

/* random x in [0, bound), biased towards nasty limb patterns */
static void
rand_below (mpz_t x, const mpz_t bound, int style)
{
  size_t bits = mpz_sizeinbase (bound, 2);
  switch (style % 5)
    {
    case 0: mpz_urandomm (x, rs, bound); return;
    case 1: mpz_sub_ui (x, bound, 1 + (gmp_urandomm_ui (rs, 3))); break;  /* top */
    case 2: mpz_set_ui (x, gmp_urandomm_ui (rs, 3)); return;               /* tiny */
    case 3: mpz_rrandomb (x, rs, bits); break;                             /* long runs */
    default:                                                               /* 2^k - 1 */
      mpz_set_ui (x, 0);
      mpz_setbit (x, gmp_urandomm_ui (rs, bits));
      mpz_sub_ui (x, x, 1);
      break;
    }
  if (mpz_sgn (x) < 0) mpz_set_ui (x, 0);
  mpz_mod (x, x, bound);
}

/* random odd modulus with 16N <= 2^(32n) */
static void
rand_mod (mpz_t N, uint32_t n, int style)
{
  size_t maxbits = 32 * (size_t) n - 4;
  size_t bits;
  do
    {
      switch (style % 6)
        {
        case 0: bits = maxbits; break;
        case 1: bits = maxbits > 40 ? maxbits - gmp_urandomm_ui (rs, 32) : maxbits; break;
        default: bits = 2 + gmp_urandomm_ui (rs, maxbits - 1); break;
        }
      switch (style % 4)
        {
        case 0: mpz_urandomb (N, rs, bits); mpz_setbit (N, bits - 1); break;
        case 1: mpz_set_ui (N, 0); mpz_setbit (N, bits); mpz_sub_ui (N, N, 1); break;  /* 2^k - 1 */
        case 2: mpz_set_ui (N, 0); mpz_setbit (N, bits - 1); mpz_add_ui (N, N, 1); break; /* 2^k + 1 */
        default: mpz_rrandomb (N, rs, bits); break;
        }
      mpz_setbit (N, 0);
    }
  while (mpz_sizeinbase (N, 2) > maxbits || mpz_cmp_ui (N, 3) < 0);
}

static int failures = 0;

template<int L, bool SOLO>
static void
run_shape (uint32_t tpi, int iters)
{
  const uint32_t n = tpi * L;
  const uint32_t count = 512;
  std::vector<uint32_t> A (count * n), B (count * n), Nn (count * n), D (2 * count), O (count * n);
  mpz_t *va = new mpz_t[count], *vb = new mpz_t[count], *vn = new mpz_t[count];
  mpz_t R, Rinv, t, u, bound;
  mpz_inits (R, Rinv, t, u, bound, NULL);
  mpz_set_ui (R, 0);
  mpz_setbit (R, 32 * (size_t) n);
  for (uint32_t i = 0; i < count; i++)
    mpz_inits (va[i], vb[i], vn[i], NULL);

  uint32_t *gA, *gB, *gN, *gD, *gO;
  CK (cudaMalloc (&gA, A.size () * 4)); CK (cudaMalloc (&gB, B.size () * 4));
  CK (cudaMalloc (&gN, Nn.size () * 4)); CK (cudaMalloc (&gD, D.size () * 4));
  CK (cudaMalloc (&gO, O.size () * 4));

  for (int it = 0; it < iters; it++)
    for (int op = 0; op < OP_COUNT; op++)
      {
        for (uint32_t i = 0; i < count; i++)
          {
            int st = it * 7 + i;
            rand_mod (vn[i], n, st + op);
            /* inputs at their maximal allowed ranges */
            switch (op)
              {
              case OP_MUL:   /* a < 4N, b < 4N */
              case OP_SQR:
                mpz_mul_ui (bound, vn[i], 4); rand_below (va[i], bound, st);
                rand_below (vb[i], bound, st / 5 + 1); break;
              case OP_ADD:   /* a < 4N, b < 2N (sum < 6N) */
                mpz_mul_ui (bound, vn[i], 4); rand_below (va[i], bound, st);
                mpz_mul_ui (bound, vn[i], 2); rand_below (vb[i], bound, st / 5 + 2); break;
              case OP_SUB:   /* a < 4N, b < 2N */
                mpz_mul_ui (bound, vn[i], 4); rand_below (va[i], bound, st);
                mpz_mul_ui (bound, vn[i], 2); rand_below (vb[i], bound, st / 5 + 3); break;
              case OP_MULUI: /* a < 2N */
                mpz_mul_ui (bound, vn[i], 2); rand_below (va[i], bound, st);
                mpz_set_ui (vb[i], 0); break;
              default:       /* a < 2^(32n-1) */
                mpz_fdiv_q_2exp (bound, R, 1); rand_below (va[i], bound, st);
                mpz_set_ui (vb[i], 0); break;
              }
            if (op == OP_SQR)
              mpz_set (vb[i], va[i]);
            to_limbs (&A[i * n], n, va[i]);
            to_limbs (&B[i * n], n, vb[i]);
            to_limbs (&Nn[i * n], n, vn[i]);
            uint32_t d = (uint32_t) gmp_urandomb_ui (rs, 32);
            if (i % 3 == 0) d = 0xffffffffu - (i % 2);
            D[2 * i] = d;
            mpz_set_ui (t, 0); mpz_setbit (t, 32);
            mpz_invert (t, vn[i], t);
            D[2 * i + 1] = - (uint32_t) mpz_get_ui (t);
          }
        CK (cudaMemcpy (gA, A.data (), A.size () * 4, cudaMemcpyHostToDevice));
        CK (cudaMemcpy (gB, B.data (), B.size () * 4, cudaMemcpyHostToDevice));
        CK (cudaMemcpy (gN, Nn.data (), Nn.size () * 4, cudaMemcpyHostToDevice));
        CK (cudaMemcpy (gD, D.data (), D.size () * 4, cudaMemcpyHostToDevice));
        uint32_t threads = count * tpi;
        test_kernel<L, SOLO><<<(threads + 127) / 128, 128>>> (op, tpi, count, gA, gB, gN, 0, gD, gO);
        CK (cudaGetLastError ());
        CK (cudaMemcpy (O.data (), gO, O.size () * 4, cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < count; i++)
          {
            mpz_t r;
            mpz_init (r);
            mpz_import (r, n, -1, 4, 0, 0, &O[i * n]);
            /* expected value (mod N) and bound */
            mpz_invert (Rinv, R, vn[i]);
            switch (op)
              {
              case OP_MUL: case OP_SQR:
                mpz_mul (t, va[i], vb[i]); mpz_mul (t, t, Rinv);
                mpz_mul_ui (bound, vn[i], 2); break;
              case OP_ADD:
                mpz_add (t, va[i], vb[i]); mpz_mul_ui (bound, vn[i], 6); break;
              case OP_SUB:
                mpz_sub (t, va[i], vb[i]); mpz_mul_ui (bound, vn[i], 4); break;
              case OP_MULUI:
                mpz_set_ui (u, 0); mpz_setbit (u, 32); mpz_invert (u, u, vn[i]);
                mpz_mul_ui (t, va[i], D[2 * i]); mpz_mul (t, t, u);
                mpz_mul_ui (bound, vn[i], 3); break;
              default:
                mpz_mul_2exp (t, va[i], 1); mpz_set (bound, R); break;
              }
            int bad;
            if (op == OP_ADD || op == OP_DBL)
              bad = mpz_cmp (r, t) != 0;   /* exact */
            else
              {
                mpz_sub (u, r, t);
                bad = !mpz_divisible_p (u, vn[i]) || mpz_cmp (r, bound) >= 0;
              }
            if (op == OP_SUB && !bad)
              {
                /* must be a - b, or a - b + 2N when that is negative */
                mpz_sub (u, va[i], vb[i]);
                if (mpz_sgn (u) < 0) mpz_addmul_ui (u, vn[i], 2);
                bad = mpz_cmp (r, u) != 0;
              }
            if (bad)
              {
                if (failures++ < 20)
                  gmp_printf ("FAIL %s TPI=%u L=%d i=%u\n  N=%Zx\n  a=%Zx\n  b=%Zx\n  d=%08x\n  got=%Zx\n",
                              op_name[op], tpi, L, i, vn[i], va[i], vb[i], D[2 * i], r);
              }
            mpz_clear (r);
          }
      }

  CK (cudaFree (gA)); CK (cudaFree (gB)); CK (cudaFree (gN)); CK (cudaFree (gD)); CK (cudaFree (gO));
  for (uint32_t i = 0; i < count; i++)
    mpz_clears (va[i], vb[i], vn[i], NULL);
  delete[] va; delete[] vb; delete[] vn;
  mpz_clears (R, Rinv, t, u, bound, NULL);
  printf ("TPI=%2u L=%2d (%5u bits): %s\n", tpi, L, 32 * n, failures ? "FAILURES so far" : "ok");
  fflush (stdout);
}

template<int L>
static void
run_dist (int iters)
{
  for (uint32_t tpi = 2; tpi <= 32; tpi *= 2)
    run_shape<L, false> (tpi, iters);
}

int
main (int argc, char **argv)
{
  int iters = argc > 1 ? atoi (argv[1]) : 4;
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 12345);

  run_shape<1, true> (1, iters);
  run_shape<2, true> (1, iters);
  run_shape<3, true> (1, iters);
  run_shape<4, true> (1, iters);
  run_shape<5, true> (1, iters);
  run_shape<8, true> (1, iters);
  run_shape<13, true> (1, iters);
  run_shape<16, true> (1, iters);

  run_dist<1> (iters);
  run_dist<2> (iters);
  run_dist<3> (iters);
  run_dist<4> (iters);
  run_dist<7> (iters);
  run_dist<8> (iters);
  run_dist<16> (iters);
  run_dist<24> (iters);

  printf ("%s (%d failures)\n", failures ? "FAILED" : "ALL OK", failures);
  return failures != 0;
}
