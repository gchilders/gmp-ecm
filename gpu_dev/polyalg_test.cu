/* polyalg_test.cu: check the polynomial algorithms (gpu_polyalg.cu) against
   GMP.

   nvcc -O2 -arch=sm_86 -I.. -o polyalg_test polyalg_test.cu -lgmp
   ./polyalg_test

   For moduli of several sizes and n = 2^l roots: F = prod (x + f_i) and
   its product tree, I = 1/rev(F) mod x^n, A mod F for a random A of degree
   2n-2, the transposed evaluation of a polynomial at the -f_i (against
   Horner), and the whole stage 2 chain: H = prod_j G_j mod F over k blocks
   of roots g, evaluated at the -f_i, against prod_(j,m) (g_jm - f_i). */

#include "../gpu_rns.cu"
#include "../gpu_polyalg.cu"
#include <gmp.h>

static gmp_randstate_t rs;
static int failures = 0;

struct zvec {
  std::vector<__mpz_struct> v;
  zvec (size_t n = 0) : v (n) { for (auto &z : v) mpz_init (&z); }
  ~zvec () { for (auto &z : v) mpz_clear (&z); }
  void resize (size_t n) { size_t o = v.size (); v.resize (n);
                           for (size_t i = o; i < n; i++) mpz_init (&v[i]); }
  __mpz_struct *operator[] (size_t i) { return &v[i]; }
  size_t size () const { return v.size (); }
};

static int NL;

static uint32_t *
to_dev (zvec &z)
{
  std::vector<uint32_t> h (z.size () * NL, 0);
  for (size_t i = 0; i < z.size (); i++)
    {
      size_t cnt;
      mpz_export (&h[i * NL], &cnt, -1, 4, 0, 0, z[i]);
    }
  uint32_t *d;
  CUDA_CHECK (cudaMalloc (&d, h.size () * 4));
  CUDA_CHECK (cudaMemcpy (d, h.data (), h.size () * 4, cudaMemcpyHostToDevice));
  return d;
}

static void
from_dev (zvec &z, const uint32_t *d, size_t n, const mpz_t N)
{
  std::vector<uint32_t> h (n * NL);
  CUDA_CHECK (cudaMemcpy (h.data (), d, h.size () * 4, cudaMemcpyDeviceToHost));
  z.resize (n);
  for (size_t i = 0; i < n; i++)
    {
      mpz_import (z[i], NL, -1, 4, 0, 0, &h[i * NL]);
      mpz_mod (z[i], z[i], N);
    }
}

static void
check (const char *what, int bits, size_t n, int bad, size_t total)
{
  if (bad)
    {
      failures++;
      printf ("FAIL %s (%d bits, n = %zu): %d of %zu wrong\n", what, bits, n,
              bad, total);
    }
}

static void
test (int bits, int l, int kblocks)
{
  const size_t n = (size_t) 1 << l;
  mpz_t N, t, u, x;
  mpz_inits (N, t, u, x, NULL);
  mpz_urandomb (N, rs, bits);
  mpz_setbit (N, bits - 1);
  mpz_setbit (N, 0);
  gpu_rns *R = gpu_rns_create (N, std::max (l + 1, 4));
  NL = gpu_rns_limbs (R);
  gpu_polyalg *A = gpu_polyalg_create (R, n);

  zvec f (n);
  for (size_t i = 0; i < n; i++)
    mpz_urandomm (f[i], rs, N);
  uint32_t *df = to_dev (f);
  std::vector<uint32_t *> tree (l > 0 ? l : 1);
  for (auto &p : tree)
    CUDA_CHECK (cudaMalloc (&p, n * NL * 4));
  uint32_t *dF, *dI, *dH, *dG, *dV, *dA2;
  CUDA_CHECK (cudaMalloc (&dF, n * NL * 4));
  CUDA_CHECK (cudaMalloc (&dI, n * NL * 4));
  CUDA_CHECK (cudaMalloc (&dH, n * NL * 4));
  CUDA_CHECK (cudaMalloc (&dG, n * NL * 4));
  CUDA_CHECK (cudaMalloc (&dV, n * NL * 4));
  CUDA_CHECK (cudaMalloc (&dA2, 2 * n * NL * 4));

  /* F and its tree */
  gpu_poly_from_roots (A, dF, tree.data (), df);
  zvec F, Fc (n + 1);
  from_dev (F, dF, n, N);
  mpz_set_ui (Fc[0], 1);              /* Fc[c]: coefficient of x^c */
  for (size_t i = 0; i < n; i++)
    {
      /* times (x + f_i), Fc of degree i */
      for (size_t j = i + 1; j > 0; j--)
        {
          mpz_mul (t, Fc[j], f[i]);
          mpz_add (Fc[j], t, Fc[j - 1]);
          mpz_mod (Fc[j], Fc[j], N);
        }
      mpz_mul (Fc[0], Fc[0], f[i]);
      mpz_mod (Fc[0], Fc[0], N);
    }
  int bad = 0;
  for (size_t c = 0; c < n; c++)
    bad += mpz_cmp (F[c], Fc[c]) != 0;
  check ("F", bits, n, bad, n);

  /* inverse: rev(F) I = 1 mod x^n */
  gpu_poly_inverse (A, dI, dF);
  zvec I;
  from_dev (I, dI, n, N);
  bad = 0;
  for (size_t c = 0; c < n; c++)
    {
      /* coefficient c of rev(F) I: revF[0] = 1, revF[s] = F[n - s] */
      mpz_set (x, I[c]);
      for (size_t s = 1; s <= c; s++)
        mpz_addmul (x, F[n - s], I[c - s]);
      mpz_mod (x, x, N);
      bad += mpz_cmp_ui (x, c == 0 ? 1 : 0) != 0;
    }
  check ("inverse", bits, n, bad, n);

  gpu_poly_prepare (A, dF, dI);

  /* reduction of a random A2 of degree 2n-2 */
  if (n > 1 && n <= 512)
    {
      zvec A2 (2 * n - 1);
      for (size_t i = 0; i < 2 * n - 1; i++)
        mpz_urandomm (A2[i], rs, N);
      uint32_t *d = to_dev (A2);
      gpu_poly_reduce (A, dA2, d);
      zvec Rg;
      from_dev (Rg, dA2, n, N);
      /* CPU: long division by the monic F */
      zvec r (2 * n - 1);
      for (size_t i = 0; i < 2 * n - 1; i++)
        mpz_set (r[i], A2[i]);
      for (size_t d2 = 2 * n - 2; d2 >= n; d2--)
        {
          /* subtract r[d2] x^(d2-n) F */
          mpz_set (u, r[d2]);
          for (size_t c = 0; c < n; c++)
            {
              mpz_submul (r[d2 - n + c], u, F[c]);
              mpz_mod (r[d2 - n + c], r[d2 - n + c], N);
            }
          mpz_set_ui (r[d2], 0);
        }
      bad = 0;
      for (size_t c = 0; c < n; c++)
        bad += mpz_cmp (Rg[c], r[c]) != 0;
      check ("reduce", bits, n, bad, n);
      cudaFree (d);
    }

  /* the chain: H = prod_j G_j mod F, values at -f_i */
  std::vector<zvec> g (kblocks);
  zvec Hc (n);                         /* CPU H, for n <= 256 */
  for (int j = 0; j < kblocks; j++)
    {
      g[j].resize (n);
      for (size_t i = 0; i < n; i++)
        mpz_urandomm (g[j][i], rs, N);
      uint32_t *dg = to_dev (g[j]);
      gpu_poly_from_roots (A, dG, NULL, dg);
      gpu_poly_mulmod (A, dH, dG, j == 0);
      cudaFree (dg);
      if (n > 256)
        continue;
      /* CPU: G_j, then H <- G_j H mod F */
      zvec Gc (n + 1), Gg, Hg;
      mpz_set_ui (Gc[0], 1);
      for (size_t i = 0; i < n; i++)
        {
          for (size_t q = i + 1; q > 0; q--)
            {
              mpz_mul (t, Gc[q], g[j][i]);
              mpz_add (Gc[q], t, Gc[q - 1]);
              mpz_mod (Gc[q], Gc[q], N);
            }
          mpz_mul (Gc[0], Gc[0], g[j][i]);
          mpz_mod (Gc[0], Gc[0], N);
        }
      from_dev (Gg, dG, n, N);
      int bg = 0;
      for (size_t c = 0; c < n; c++)
        bg += mpz_cmp (Gg[c], Gc[c]) != 0;
      check ("G (no tree)", bits, n, bg, n);
      zvec prod (2 * n + 1);
      if (j == 0)
        for (size_t c = 0; c < n; c++)
          {
            mpz_sub (Hc[c], Gc[c], F[c]);
            mpz_mod (Hc[c], Hc[c], N);
          }
      else
        {
          for (size_t a = 0; a <= n; a++)
            for (size_t b = 0; b < n; b++)
              mpz_addmul (prod[a + b], Gc[a], Hc[b]);
          for (size_t d2 = 2 * n - 1; d2 >= n; d2--)
            {
              mpz_mod (u, prod[d2], N);
              for (size_t c = 0; c < n; c++)
                mpz_submul (prod[d2 - n + c], u, F[c]);
              mpz_set_ui (prod[d2], 0);
            }
          for (size_t c = 0; c < n; c++)
            mpz_mod (Hc[c], prod[c], N);
        }
      from_dev (Hg, dH, n, N);
      int bh = 0;
      for (size_t c = 0; c < n; c++)
        bh += mpz_cmp (Hg[c], Hc[c]) != 0;
      if (bh)
        printf ("  block %d: H differs in %d of %zu coefficients\n", j, bh, n);
      check ("H after block", bits, n, bh, n);
    }
  gpu_poly_tpeval (A, dV, dH, tree.data ());
  zvec H, V;
  from_dev (H, dH, n, N);
  from_dev (V, dV, n, N);
  int bad_h = 0, bad_v = 0;
  for (size_t i = 0; i < n; i++)
    {
      /* Horner: H(-f_i) */
      mpz_set_ui (x, 0);
      mpz_neg (u, f[i]);
      for (size_t c = n; c-- > 0;)
        {
          mpz_mul (x, x, u);
          mpz_add (x, x, H[c]);
          mpz_mod (x, x, N);
        }
      bad_h += mpz_cmp (x, V[i]) != 0;
      /* prod (g - f_i) */
      mpz_set_ui (t, 1);
      for (int j = 0; j < kblocks; j++)
        for (size_t m = 0; m < n; m++)
          {
            mpz_sub (u, g[j][m], f[i]);
            mpz_mul (t, t, u);
            mpz_mod (t, t, N);
          }
      bad_v += mpz_cmp (t, V[i]) != 0;
    }
  check ("tpeval vs Horner", bits, n, bad_h, n);
  check ("chain vs prod (g - f)", bits, n, bad_v, n);
  printf ("%5d bits, n = %5zu, %d blocks (%d primes, K = %d): %s\n", bits, n,
          kblocks, gpu_rns_primes (R), gpu_rns_lg (R),
          failures ? "failures so far" : "ok");
  fflush (stdout);

  for (auto &p : tree)
    cudaFree (p);
  cudaFree (df); cudaFree (dF); cudaFree (dI); cudaFree (dH); cudaFree (dG);
  cudaFree (dV); cudaFree (dA2);
  gpu_polyalg_destroy (A);
  gpu_rns_destroy (R);
  mpz_clears (N, t, u, x, NULL);
}

int
main ()
{
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 77);
  for (int bits : { 256, 1024, 4096 })
    for (int l : { 0, 1, 2, 4, 8, 10 })
      test (bits, l, l == 0 ? 2 : 3);
  printf ("%s (%d failures)\n", failures ? "FAILED" : "ALL OK", failures);
  return failures != 0;
}
