/* s2poly_bench.cu: time the polynomial GPU stage 2 (gpu_s2poly.cu) on
   random curves, with its phases (GMPECM_S2POLY_PROF).  Links libecm for
   outputf.

   nvcc -O2 -arch=sm_86 -I.. -I<build dir> -DGPU_ROOTS_MAX_L=8 -o s2poly_bench \
     s2poly_bench.cu <build dir>/.libs/libecm.a -lgmp -lm
   ./s2poly_bench BITS CURVES S d1 d2 i0 dF k
   (GMP-ECM's parameters: "-v" prints dF, k, d = d1, d2, i0; S from choose_S)
   With V=3 in the environment, each curve's "Product of G(f_i)". */

#include "../gpu_rns.cu"
#include "../gpu_polyalg.cu"
#include "../gpu_roots.cu"
#include "../gpu_s2poly.cu"

int
main (int argc, char **argv)
{
  if (argc < 9)
    {
      fprintf (stderr, "usage: %s BITS CURVES S d1 d2 i0 dF k\n", argv[0]);
      return 1;
    }
  const int bits = atoi (argv[1]);
  const uint32_t C = atoi (argv[2]);
  const int S = atoi (argv[3]);
  const unsigned long d1 = atol (argv[4]), d2 = atol (argv[5]);
  const unsigned long dF = atol (argv[7]), k = atol (argv[8]);
  setenv ("GMPECM_S2POLY_PROF", "1", 0);
  /* V=3: print each curve's "Product of G(f_i)" (to compare batch sizes) */
  ECM_STDOUT = stdout;
  ECM_STDERR = stderr;
  set_verbose (getenv ("V") != NULL ? atoi (getenv ("V")) : 0);
  gmp_randstate_t rs;
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 17);
  mpz_t N, i0;
  mpz_inits (N, i0, NULL);
  mpz_set_str (i0, argv[6], 10);
  mpz_urandomb (N, rs, bits);
  mpz_setbit (N, bits - 1);
  /* no factor to find: a prime, or above 4096 bits (nextprime is slow
     there) an odd number without prime factors below 10^5 */
  if (bits <= 4096)
    mpz_nextprime (N, N);
  else
    {
      mpz_setbit (N, 0);
      for (unsigned long p = 3; p < 100000; p += 2)
        if (mpz_divisible_ui_p (N, p))
          {
            mpz_add_ui (N, N, 2);
            p = 1;
          }
    }
  mpz_t *xs = new mpz_t[C], *As = new mpz_t[C];
  std::vector<int> found (C, ECM_NO_FACTOR_FOUND);
  for (uint32_t c = 0; c < C; c++)
    {
      mpz_init (xs[c]);
      mpz_init (As[c]);
      mpz_urandomm (xs[c], rs, N);
      mpz_urandomm (As[c], rs, N);
    }
  float ms;
  int r = gpu_ecm_stage2_poly (xs, found.data (), As, N, S, d1, d2, i0, dF,
                               k, C, &ms, 0);
  printf ("%d bits, %u curves, S = %d, dF = %lu, k = %lu: %.1f ms, %.2f ms "
          "per curve (result %d)\n", bits, C, S, dF, k, ms, ms / C, r);
  return 0;
}
