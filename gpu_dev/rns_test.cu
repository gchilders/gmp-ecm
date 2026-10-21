/* rns_test.cu: check the RNS polynomial engine (gpu_rns.cu) against GMP.

   nvcc -O2 -arch=sm_86 -I.. -o rns_test rns_test.cu -lgmp
   ./rns_test

   For moduli of 512 to 32768 bits: the round trip Z/N -> RNS -> Z/N, forward
   then inverse transforms at lengths below, at and above the in-block
   length, and products of random polynomials mod N (all coefficients for
   small degrees, sampled ones for the longest transforms). */

#include "../gpu_rns.cu"
#include <gmp.h>

static gmp_randstate_t rs;
static int failures = 0;

static void
fail (const char *what, int bits, const char *detail)
{
  if (failures++ < 20)
    printf ("FAIL %s (%d bits): %s\n", what, bits, detail);
}

struct mpz_vec {
  std::vector<__mpz_struct> v;
  mpz_vec (size_t n) : v (n) { for (auto &z : v) mpz_init (&z); }
  ~mpz_vec () { for (auto &z : v) mpz_clear (&z); }
  __mpz_struct *operator[] (size_t i) { return &v[i]; }
};

static void
fill (std::vector<uint32_t> &h, mpz_vec &z, size_t C, int nl, const mpz_t N,
      int style)
{
  h.assign (C * nl, 0);
  for (size_t c = 0; c < C; c++)
    {
      mpz_t x;
      mpz_init (x);
      switch ((c + style) % 7)
        {
        case 0: mpz_set_ui (x, 0); mpz_setbit (x, 32 * (size_t) nl);
                mpz_sub_ui (x, x, 1); break;              /* 2^(32nl) - 1 */
        case 1: mpz_sub_ui (x, N, 1); break;
        case 2: mpz_set_ui (x, c % 3); break;
        case 3: mpz_mul_2exp (x, N, 1); mpz_sub_ui (x, x, 1); break;
        default: mpz_urandomm (x, rs, N); break;
        }
      mpz_set (z[c], x);
      size_t cnt;
      mpz_export (&h[c * nl], &cnt, -1, 4, 0, 0, x);
      mpz_clear (x);
    }
}

static void
get (mpz_t x, const std::vector<uint32_t> &h, size_t c, int nl)
{
  mpz_import (x, nl, -1, 4, 0, 0, &h[c * nl]);
}

static void
test_size (int bits, int lg_want)
{
  mpz_t N, x, y, t;
  mpz_inits (N, x, y, t, NULL);
  mpz_urandomb (N, rs, bits);
  mpz_setbit (N, bits - 1);
  mpz_setbit (N, 0);
  gpu_rns *R = gpu_rns_create (N, lg_want);
  const int nl = gpu_rns_limbs (R), P = gpu_rns_primes (R), K = gpu_rns_lg (R);
  printf ("%6d bits: %4d limbs, %4d primes, K = %d\n", bits, nl, P, K);
  fflush (stdout);
  char msg[256];

  /* round trip */
  {
    const size_t C = 300;
    std::vector<uint32_t> h, o (C * nl);
    mpz_vec z (C);
    fill (h, z, C, nl, N, 0);
    uint32_t *da, *dr, *db;
    CUDA_CHECK (cudaMalloc (&da, h.size () * 4));
    CUDA_CHECK (cudaMalloc (&db, h.size () * 4));
    CUDA_CHECK (cudaMalloc (&dr, (size_t) P * 512 * 4));
    CUDA_CHECK (cudaMemcpy (da, h.data (), h.size () * 4, cudaMemcpyHostToDevice));
    gpu_rns_from_z (R, dr, 512, da, C, C, 1);
    gpu_rns_to_z (R, db, C, dr, 512, C, 1);
    CUDA_CHECK (cudaMemcpy (o.data (), db, o.size () * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    mpz_mul_2exp (t, N, 1);
    for (size_t c = 0; c < C; c++)
      {
        get (y, o, c, nl);
        mpz_sub (x, y, z[c]);
        if (!mpz_divisible_p (x, N) || mpz_cmp (y, t) >= 0)
          bad++;
      }
    if (bad)
      {
        snprintf (msg, sizeof msg, "%d of %zu wrong", bad, C);
        fail ("round trip", bits, msg);
      }
    cudaFree (da); cudaFree (db); cudaFree (dr);
  }

  /* transforms: forward then inverse is the identity */
  for (int lg : { 6, std::min (K, 12), K })
    {
      const size_t len = (size_t) 1 << lg;
      std::vector<uint32_t> h ((size_t) P * len), o (h.size ());
      std::vector<uint32_t> ps (P);
      CUDA_CHECK (cudaMemcpy (ps.data (), R->d_p, P * 4, cudaMemcpyDeviceToHost));
      for (int i = 0; i < P; i++)
        for (size_t e = 0; e < len; e++)
          h[(size_t) i * len + e] = (uint32_t) (gmp_urandomb_ui (rs, 32) % ps[i]);
      uint32_t *dr;
      CUDA_CHECK (cudaMalloc (&dr, h.size () * 4));
      CUDA_CHECK (cudaMemcpy (dr, h.data (), h.size () * 4, cudaMemcpyHostToDevice));
      gpu_rns_ntt (R, dr, len, P, lg, 0, 0);
      gpu_rns_ntt (R, dr, len, P, lg, 1, 0);
      CUDA_CHECK (cudaMemcpy (o.data (), dr, o.size () * 4, cudaMemcpyDeviceToHost));
      if (o != h)
        {
          snprintf (msg, sizeof msg, "length 2^%d", lg);
          fail ("ntt round trip", bits, msg);
        }
      cudaFree (dr);
    }

  /* products */
  struct { size_t na, nb; bool all; } cases[] = {
    { 1, 1, true }, { 37, 59, true }, { 300, 211, true },
    { ((size_t) 1 << (K - 1)) - 3, ((size_t) 1 << (K - 1)) + 1, false },
    { ((size_t) 1 << K) - 100, 101, false } };
  for (auto &cs : cases)
    {
      const size_t na = cs.na, nb = cs.nb, nc = na + nb - 1;
      if (nc > ((size_t) 1 << K))
        continue;
      std::vector<uint32_t> ha, hb, hc (nc * nl);
      mpz_vec za (na), zb (nb);
      fill (ha, za, na, nl, N, 1);
      fill (hb, zb, nb, nl, N, 3);
      uint32_t *da, *db, *dc;
      CUDA_CHECK (cudaMalloc (&da, ha.size () * 4));
      CUDA_CHECK (cudaMalloc (&db, hb.size () * 4));
      CUDA_CHECK (cudaMalloc (&dc, hc.size () * 4));
      CUDA_CHECK (cudaMemcpy (da, ha.data (), ha.size () * 4, cudaMemcpyHostToDevice));
      CUDA_CHECK (cudaMemcpy (db, hb.data (), hb.size () * 4, cudaMemcpyHostToDevice));
      gpu_rns_polymul (R, dc, da, na, db, nb, 1);
      CUDA_CHECK (cudaMemcpy (hc.data (), dc, hc.size () * 4, cudaMemcpyDeviceToHost));
      int bad = 0, checked = 0;
      mpz_mul_2exp (t, N, 1);
      for (size_t k = 0; k < nc; k++)
        {
          if (!cs.all && k > 8 && k + 8 < nc && gmp_urandomm_ui (rs, nc / 24 + 1))
            continue;
          mpz_set_ui (x, 0);
          const size_t i0 = k >= nb - 1 ? k - (nb - 1) : 0;
          const size_t i1 = k < na - 1 ? k : na - 1;
          for (size_t i = i0; i <= i1; i++)
            mpz_addmul (x, za[i], zb[k - i]);
          get (y, hc, k, nl);
          mpz_sub (x, y, x);
          checked++;
          if (!mpz_divisible_p (x, N) || mpz_cmp (y, t) >= 0)
            bad++;
        }
      if (bad)
        {
          snprintf (msg, sizeof msg, "%zu x %zu: %d of %d checked wrong", na, nb,
                    bad, checked);
          fail ("polymul", bits, msg);
        }
      else
        printf ("   polymul %zu x %zu ok (%d coefficients checked)\n", na, nb,
                checked);
      fflush (stdout);
      cudaFree (da); cudaFree (db); cudaFree (dc);
    }
  gpu_rns_destroy (R);
  mpz_clears (N, x, y, t, NULL);
}

int
main (int argc, char **argv)
{
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 2026);
  const int lg = argc > 1 ? atoi (argv[1]) : 16;
  for (int bits : { 512, 1024, 2048, 4096, 8192, 16384, 32768 })
    test_size (bits, lg);
  printf ("%s (%d failures)\n", failures ? "FAILED" : "ALL OK", failures);
  return failures != 0;
}
