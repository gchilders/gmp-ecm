/* ntt_mul_test.cu: check the NTT Montgomery multiplication and the
   multiword operations of gpu_ntt.cuh against GMP.

   nvcc -O2 -arch=sm_86 -I.. -I. -o ntt_mul_test ntt_mul_test.cu -lgmp
   ./ntt_mul_test [cases]

   For every transform length 2^10..2^14, random and adversarial N (odd,
   up to the size limit, 2^k-1 shapes, sparse) and inputs at the edges of
   their allowed ranges. */

#include "../gpu_ntt.cu"
#include <gmp.h>

enum { OP_MUL, OP_SQR, OP_ADD, OP_SUB, OP_DBL, OP_MULUI, OP_N };
static const char *opname[] = { "mul", "sqr", "add", "sub", "dbl", "mul_ui" };

__global__ void __launch_bounds__ (NTT_MAX_THREADS)
test_kernel (int op, const ntt_ctx c, uint32_t *scratch, size_t pc,
             const uint32_t *A, const uint32_t *B, const uint32_t *D,
             uint32_t *out)
{
  extern __shared__ uint32_t sm[];
  __shared__ ntt_shared S;
  const int W = c.W, L = c.L, t = blockIdx.x;
  uint32_t *base = scratch + (size_t) t * pc;
  ntt_scratch X;
  X.ahat = base;
  for (int i = 0; i < NTT_NPRIMES; i++)
    X.res[i] = base + (size_t) L * (1 + i);
  X.T = base + (size_t) L * (1 + NTT_NPRIMES);
  X.Q = X.T + L;
  X.U = X.Q + L;
  X.mw = X.U + L;
  uint32_t *a = base + ntt_scratch_words (L), *b = a + W, *r = b + W;
  ntt_copy (c, a, A + (size_t) t * W);
  ntt_copy (c, b, B + (size_t) t * W);
  switch (op)
    {
    case OP_MUL: ntt_mont_mul (sm, S, c, X, r, a, b); break;
    case OP_SQR: ntt_mont_mul (sm, S, c, X, r, a, a); break;
    case OP_ADD: ntt_mw_add (S, c, r, a, b); break;
    case OP_SUB: ntt_mw_sub (S, c, r, a, b); break;
    case OP_DBL: ntt_mw_dbl (c, r, a); break;
    default: ntt_mw_mul_ui (S, c, X, r, a, D[t]); break;
    }
  ntt_copy (c, out + (size_t) t * W, r);
}

static gmp_randstate_t rs;

static void
rand_below (mpz_t x, const mpz_t bound, int style)
{
  size_t bits = mpz_sizeinbase (bound, 2);
  switch (style % 5)
    {
    case 0: mpz_urandomm (x, rs, bound); return;
    case 1: mpz_sub_ui (x, bound, 1 + gmp_urandomm_ui (rs, 3)); break;
    case 2: mpz_set_ui (x, gmp_urandomm_ui (rs, 3)); return;
    case 3: mpz_rrandomb (x, rs, bits); break;
    default: mpz_set_ui (x, 0); mpz_setbit (x, gmp_urandomm_ui (rs, bits));
             mpz_sub_ui (x, x, 1); break;
    }
  if (mpz_sgn (x) < 0) mpz_set_ui (x, 0);
  mpz_mod (x, x, bound);
}

int
main (int argc, char **argv)
{
  int cases = argc > 1 ? atoi (argv[1]) : 64;
  int failures = 0;
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 4242);
  for (int lg = NTT_MIN_LG; lg <= NTT_MAX_LG; lg++)
    for (int shape = 0; shape < 4; shape++)
      {
        const int L = 1 << lg, W = L / 2;
        /* N: largest allowed, a 2^k-1 at the limit, mid-size, sparse */
        size_t maxbits = 32 * (size_t) W - 4;
        mpz_t N, R, Rinv, x, y, z, bound;
        mpz_inits (N, R, Rinv, x, y, z, bound, NULL);
        switch (shape)
          {
          case 0: mpz_urandomb (N, rs, maxbits); mpz_setbit (N, maxbits - 1); break;
          case 1: mpz_setbit (N, maxbits); mpz_sub_ui (N, N, 1); break;
          case 2: mpz_urandomb (N, rs, maxbits - 700); mpz_setbit (N, maxbits - 701); break;
          default: mpz_rrandomb (N, rs, maxbits); break;
          }
        mpz_setbit (N, 0);
        mpz_setbit (R, 32 * (size_t) W);
        mpz_invert (Rinv, R, N);
        gpu_ntt_plan *P = gpu_ntt_create (N, lg);
        const size_t pc = per_curve_words (L) + 3 * (size_t) W;
        uint32_t *g_s, *g_a, *g_b, *g_d, *g_o;
        CUDA_CHECK (cudaMalloc (&g_s, (size_t) cases * pc * 4));
        CUDA_CHECK (cudaMalloc (&g_a, (size_t) cases * W * 4));
        CUDA_CHECK (cudaMalloc (&g_b, (size_t) cases * W * 4));
        CUDA_CHECK (cudaMalloc (&g_d, (size_t) cases * 4));
        CUDA_CHECK (cudaMalloc (&g_o, (size_t) cases * W * 4));
        CUDA_CHECK (cudaFuncSetAttribute (test_kernel,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int) P->smem));
        std::vector<uint32_t> ha ((size_t) cases * W), hb ((size_t) cases * W),
                              hd (cases), ho ((size_t) cases * W);
        mpz_t *va = new mpz_t[cases], *vb = new mpz_t[cases];
        for (int i = 0; i < cases; i++) { mpz_init (va[i]); mpz_init (vb[i]); }
        for (int op = 0; op < OP_N; op++)
          {
            for (int i = 0; i < cases; i++)
              {
                int st = i * 7 + op;
                switch (op)
                  {
                  case OP_MUL: case OP_SQR:   /* a, b < 4N: product < 16N^2 */
                    mpz_mul_ui (bound, N, 4); rand_below (va[i], bound, st);
                    rand_below (vb[i], bound, st / 5 + 1); break;
                  case OP_ADD:                /* < 4N + 2N */
                    mpz_mul_ui (bound, N, 4); rand_below (va[i], bound, st);
                    mpz_mul_ui (bound, N, 2); rand_below (vb[i], bound, st / 5 + 2); break;
                  case OP_SUB:
                    mpz_mul_ui (bound, N, 4); rand_below (va[i], bound, st);
                    mpz_mul_ui (bound, N, 2); rand_below (vb[i], bound, st / 5 + 3); break;
                  case OP_DBL:
                    mpz_set (bound, R); mpz_fdiv_q_2exp (bound, bound, 1);
                    rand_below (va[i], bound, st); mpz_set_ui (vb[i], 0); break;
                  default:
                    mpz_mul_ui (bound, N, 2); rand_below (va[i], bound, st);
                    mpz_set_ui (vb[i], 0); break;
                  }
                if (op == OP_SQR) mpz_set (vb[i], va[i]);
                std::vector<uint32_t> w;
                to_words (w, W, va[i]); memcpy (&ha[(size_t) i * W], w.data (), W * 4);
                to_words (w, W, vb[i]); memcpy (&hb[(size_t) i * W], w.data (), W * 4);
                hd[i] = (i % 3 == 0) ? 0xffffffffu - (i % 2)
                                     : (uint32_t) gmp_urandomb_ui (rs, 32);
              }
            CUDA_CHECK (cudaMemcpy (g_a, ha.data (), ha.size () * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK (cudaMemcpy (g_b, hb.data (), hb.size () * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK (cudaMemcpy (g_d, hd.data (), hd.size () * 4, cudaMemcpyHostToDevice));
            test_kernel<<<cases, P->threads, P->smem>>> (op, P->ctx, g_s, pc, g_a, g_b, g_d, g_o);
            CUDA_CHECK (cudaGetLastError ());
            CUDA_CHECK (cudaMemcpy (ho.data (), g_o, ho.size () * 4, cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < cases; i++)
              {
                mpz_import (z, W, -1, 4, 0, 0, &ho[(size_t) i * W]);
                int ok;
                switch (op)
                  {
                  case OP_MUL: case OP_SQR:
                    mpz_mul (x, va[i], vb[i]); mpz_mul (x, x, Rinv);
                    mpz_sub (y, z, x); mpz_mul_ui (bound, N, 2);
                    ok = mpz_divisible_p (y, N) && mpz_cmp (z, bound) < 0; break;
                  case OP_ADD:
                    mpz_add (x, va[i], vb[i]); ok = mpz_cmp (x, z) == 0; break;
                  case OP_SUB:
                    mpz_sub (x, va[i], vb[i]);
                    if (mpz_sgn (x) < 0) mpz_addmul_ui (x, N, 2);
                    ok = mpz_cmp (x, z) == 0; break;
                  case OP_DBL:
                    mpz_mul_2exp (x, va[i], 1); ok = mpz_cmp (x, z) == 0; break;
                  default:
                    mpz_set_ui (y, 0); mpz_setbit (y, 32); mpz_invert (y, y, N);
                    mpz_mul_ui (x, va[i], hd[i]); mpz_mul (x, x, y);
                    mpz_sub (y, z, x); mpz_mul_ui (bound, N, 3);
                    ok = mpz_divisible_p (y, N) && mpz_cmp (z, bound) < 0; break;
                  }
                if (!ok)
                  {
                    bad++;
                    if (failures + bad <= 5)
                      gmp_printf ("FAIL %s lg=%d shape=%d case=%d\n  N=%Zx\n  a=%Zx\n  b=%Zx\n  got=%Zx\n",
                                  opname[op], lg, shape, i, N, va[i], vb[i], z);
                  }
              }
            failures += bad;
            printf ("lg=%2d shape=%d %-6s %s (%d/%d)\n", lg, shape, opname[op],
                    bad ? "FAIL" : "ok", cases - bad, cases);
            fflush (stdout);
          }
        for (int i = 0; i < cases; i++) { mpz_clear (va[i]); mpz_clear (vb[i]); }
        delete[] va; delete[] vb;
        cudaFree (g_s); cudaFree (g_a); cudaFree (g_b); cudaFree (g_d); cudaFree (g_o);
        gpu_ntt_destroy (P);
        mpz_clears (N, R, Rinv, x, y, z, bound, NULL);
      }
  printf ("%s (%d failures)\n", failures ? "FAILED" : "ALL OK", failures);
  return failures != 0;
}
