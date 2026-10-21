/* roots_ref.c: GMP-ECM's own roots of F and G (ecm_rootsF, ecm_rootsG),
   the reference for roots_test.cu.  Built against libecm:

   gcc -O2 -I<build dir> -c roots_ref.c */

#include "ecm-impl.h"

/* For the Montgomery curve of A with the point of x (plain residues mod
   N), as GMP-ECM's stage 2 sees it after montgomery_to_weierstrass: the dF
   roots of F into F, and BLOCKS blocks of dF roots of G into G.  Returns
   0, or nonzero if a factor was found on the way. */
int
roots_ref (mpz_t *F, mpz_t *G, const mpz_t N, const mpz_t A, const mpz_t x,
           int S, unsigned long d1, unsigned long d2, const mpz_t i0,
           unsigned long dF, unsigned long blocks)
{
  mpmod_t mod;
  curve X;
  root_params_t rp;
  ecm_roots_state_t *st = NULL;
  listz_t l;
  mpz_t f;
  unsigned long i, b;
  int r = 0;

  mpmod_init (mod, N, ECM_MOD_DEFAULT);
  mpres_init (X.x, mod);
  mpres_init (X.y, mod);
  mpres_init (X.A, mod);
  mpres_set_z (X.x, x, mod);
  mpres_set_z (X.A, A, mod);
  mpz_init (f);
  mpz_init_set (rp.i0, i0);
  rp.d1 = d1;
  rp.d2 = d2;
  rp.S = S;
  l = init_list (dF);

  if (montgomery_to_weierstrass (f, X.x, X.y, X.A, mod))
    r = 1;
  if (r == 0 && ecm_rootsF (f, l, &rp, dF, &X, mod))
    r = 2;
  for (i = 0; r == 0 && i < dF; i++)
    mpz_set (F[i], l[i]);
  if (r == 0 && (st = ecm_rootsG_init (f, &X, &rp, dF, blocks, mod)) == NULL)
    r = 3;
  for (b = 0; r == 0 && b < blocks; b++)
    {
      if (ecm_rootsG (f, l, dF, st, mod))
        r = 4;
      for (i = 0; i < dF; i++)
        mpz_set (G[b * dF + i], l[i]);
    }

  if (st != NULL)
    ecm_rootsG_clear (st, mod);
  clear_list (l, dF);
  mpz_clear (rp.i0);
  mpz_clear (f);
  mpres_clear (X.x, mod);
  mpres_clear (X.y, mod);
  mpres_clear (X.A, mod);
  mpmod_clear (mod);
  return r;
}
