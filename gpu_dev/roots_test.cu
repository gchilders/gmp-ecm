/* roots_test.cu: check the GPU roots of F and G (gpu_roots.cu) against
   GMP-ECM's own (ecm_rootsF, ecm_rootsG, through roots_ref.c), root for
   root, for several moduli, root parameters, kernel shapes and numbers of
   progressions.

   gcc -O2 -I<build dir> -c roots_ref.c
   nvcc -O2 -arch=sm_86 -I.. -DGPU_ROOTS_MAX_L=4 \
     -o roots_test roots_test.cu roots_ref.o <build dir>/.libs/libecm.a -lgmp
   ./roots_test */

#include "../gpu_roots.cu"
#include <gmp.h>
#include <chrono>

extern "C" int roots_ref (mpz_t *F, mpz_t *G, const mpz_t N, const mpz_t A,
                          const mpz_t x, int S, unsigned long d1,
                          unsigned long d2, const mpz_t i0, unsigned long dF,
                          unsigned long blocks);

static gmp_randstate_t rs;
static int failures = 0;

struct tcase {
  int bits, S;
  unsigned long d1, d2;
  const char *i0;
  unsigned long dF, blocks;
  uint32_t C, tpi, L;
};

static double
now ()
{
  CUDA_CHECK (cudaDeviceSynchronize ());
  return std::chrono::duration<double> (
             std::chrono::steady_clock::now ().time_since_epoch ()).count ();
}

/* compare the C dF device roots in D with the reference REF[c][off + i] */
static int
compare (const char *what, const uint32_t *d, uint32_t C, size_t dF, int nl,
         std::vector<mpz_t *> &ref, size_t off, const mpz_t N)
{
  std::vector<uint32_t> h ((size_t) C * dF * nl);
  CUDA_CHECK (cudaMemcpy (h.data (), d, h.size () * 4,
                          cudaMemcpyDeviceToHost));
  mpz_t v;
  mpz_init (v);
  int bad = 0;
  for (uint32_t c = 0; c < C; c++)
    for (size_t i = 0; i < dF; i++)
      {
        mpz_import (v, nl, -1, 4, 0, 0, &h[((size_t) c * dF + i) * nl]);
        /* below 2N, as promised, and the same mod N */
        int big = 0;
        if (mpz_cmp (v, N) >= 0)
          {
            mpz_sub (v, v, N);
            big = mpz_cmp (v, N) >= 0;
          }
        if (big || mpz_cmp (v, ref[c][off + i]) != 0)
          {
            if (bad < 3)
              gmp_printf ("  %s curve %u root %zu: GPU %Zx\n    GMP-ECM %Zx\n",
                          what, c, i, v, ref[c][off + i]);
            bad++;
          }
      }
  mpz_clear (v);
  return bad;
}

static void
test (const tcase &t, const char *nr)
{
  if (nr != NULL)
    setenv ("GMPECM_ROOTS_NR", nr, 1);
  else
    unsetenv ("GMPECM_ROOTS_NR");
  const int nl = (t.bits + 4 + 31) / 32;
  mpz_t N, i0, f;
  mpz_inits (N, i0, f, NULL);
  /* a product of two primes, so that no factor turns up */
  do
    {
      mpz_urandomb (N, rs, t.bits / 2);
      mpz_setbit (N, t.bits / 2 - 1);
      mpz_nextprime (N, N);
      mpz_urandomb (f, rs, t.bits - t.bits / 2);
      mpz_setbit (f, t.bits - t.bits / 2 - 1);
      mpz_nextprime (f, f);
      mpz_mul (N, N, f);
    }
  while ((int) mpz_sizeinbase (N, 2) > t.bits);
  mpz_set_str (i0, t.i0, 10);
  mpz_t *Av = new mpz_t[t.C], *xv = new mpz_t[t.C];
  std::vector<mpz_t *> Fr (t.C), Gr (t.C);
  for (uint32_t c = 0; c < t.C; c++)
    {
      mpz_init (Av[c]);
      mpz_init (xv[c]);
      mpz_urandomm (Av[c], rs, N);
      mpz_urandomm (xv[c], rs, N);
      Fr[c] = new mpz_t[t.dF];
      Gr[c] = new mpz_t[t.dF * t.blocks];
      for (size_t i = 0; i < t.dF; i++)
        mpz_init (Fr[c][i]);
      for (size_t i = 0; i < t.dF * t.blocks; i++)
        mpz_init (Gr[c][i]);
    }

  double t0 = now ();
  gpu_roots *G = gpu_roots_create (N, t.tpi, t.L, t.S, t.d1, t.d2, i0, t.dF,
                                   t.blocks, nl, t.C, 1);
  assert (G != NULL);
  uint32_t *d_out;
  CUDA_CHECK (cudaMalloc (&d_out, (size_t) t.C * t.dF * nl * 4));
  int curve = -2;
  int fr = gpu_roots_curves (G, Av, xv, f, &curve);
  double t1 = now ();
  int bad = 0;
  double tF = 0, tG = 0;
  if (fr)
    gmp_printf ("  factor %Zd at setup, curve %d\n", f, curve);
  else
    {
      double a = now ();
      fr = gpu_roots_F (G, d_out, f, &curve);
      tF = now () - a;
      for (uint32_t c = 0; c < t.C; c++)
        {
          int r = roots_ref (Fr[c], Gr[c], N, Av[c], xv[c], t.S, t.d1, t.d2,
                             i0, t.dF, t.blocks);
          if (r)
            printf ("  GMP-ECM found a factor (%d) for curve %u\n", r, c);
        }
      if (fr)
        gmp_printf ("  factor %Zd in F, curve %d\n", f, curve);
      else
        bad += compare ("F", d_out, t.C, t.dF, nl, Fr, 0, N);
      for (unsigned long b = 0; b < t.blocks && !fr; b++)
        {
          a = now ();
          fr = gpu_roots_G (G, d_out, f, &curve);
          tG += now () - a;
          if (fr)
            gmp_printf ("  factor %Zd in G block %lu, curve %d\n", f, b,
                        curve);
          else
            bad += compare ("G", d_out, t.C, t.dF, nl, Gr, b * t.dF, N);
        }
    }
  if (bad)
    failures++;
  printf ("%5d bits S=%d d1=%lu d2=%lu i0=%s dF=%lu x%lu, %u curves, tpi %u "
          "L %u, nr %s: %s (setup %.1f ms, F %.1f ms, G %.1f ms)\n",
          t.bits, t.S, t.d1, t.d2, t.i0, t.dF, t.blocks, t.C, t.tpi, t.L,
          nr ? nr : "auto", bad ? "FAIL" : "ok", (t1 - t0) * 1e3, tF * 1e3,
          tG * 1e3);
  fflush (stdout);

  CUDA_CHECK (cudaFree (d_out));
  gpu_roots_destroy (G);
  for (uint32_t c = 0; c < t.C; c++)
    {
      for (size_t i = 0; i < t.dF; i++)
        mpz_clear (Fr[c][i]);
      for (size_t i = 0; i < t.dF * t.blocks; i++)
        mpz_clear (Gr[c][i]);
      delete[] Fr[c];
      delete[] Gr[c];
      mpz_clear (Av[c]);
      mpz_clear (xv[c]);
    }
  delete[] Av;
  delete[] xv;
  mpz_clears (N, i0, f, NULL);
}

int
main (int argc, char **argv)
{
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 4242);
  static const tcase cases[] = {
    { 250, 6, 2310, 13, "3", 240, 2, 4, 4, 4 },
    { 500, -12, 30030, 17, "100", 2880, 3, 8, 4, 4 },
    { 1000, -30, 210, 11, "12345678901", 24, 2, 3, 8, 4 },
    { 58, 1, 60, 1, "1", 8, 4, 2, 2, 1 },
    { 1000, -6, 2310, 1, "50000", 240, 3, 5, 32, 1 },
    { 300, 2, 420, 13, "7", 48, 5, 3, 4, 3 },
    /* negative i0 (GMP-ECM's choice for small B1), odd S: negative
       coefficients */
    { 400, -12, 2310, 13, "-4", 256, 3, 4, 4, 4 },
    { 300, 3, 210, 11, "-20", 24, 4, 3, 4, 3 },
  };
  for (const tcase &t : cases)
    for (const char *nr : { (const char *) NULL, "1", "7" })
      test (t, nr);
  /* a bigger one for timing: 1018 bits, Dickson 12, dF = 2880, 32 curves,
     the shape chosen by gpu_roots */
  if (argc > 1)
    {
      tcase big = { 1018, -12, 30030, 17, "300", 2880, 4, 32, 0, 0 };
      test (big, NULL);
    }
  printf ("%s (%d failures)\n", failures ? "FAILED" : "ALL OK", failures);
  return failures != 0;
}
