/* roots_zn_test.cu: check the Z/N product tree and transposed evaluation
   levels (gpu_roots_tree, gpu_roots_mp) against GMP, with the schoolbook
   kernels and with the transform products (GMPECM_ROOTS_NTT=1).

   nvcc -O2 -arch=sm_86 -I.. -o roots_zn_test roots_zn_test.cu -lgmp
   (all the schoolbook kernels: with -DGPU_ROOTS_MAX_L=6 there are none
   above about 24500 bits, and the 33000- and 65536-bit cases fail)
   ./roots_zn_test

   tree: pairs of monic nodes x^d + A, x^d + B (d low coefficients each)
   give x^2d + C, C_k = sum_(i+j=k) a_i b_j + (a_(k-d) + b_(k-d), k >= d);
   mp: out[i] = sum_j R_j T[i+j] + T[half+i], out[half+i] the same with L
   (see rt_mp_body).  Inputs below 2N; outputs must be below 2N and equal
   mod N. */

#include "../gpu_roots.cu"
#include <gmp.h>

static gmp_randstate_t rs;
static int failures = 0;

static void
get (mpz_t r, const std::vector<uint32_t> &h, size_t i, uint32_t nl)
{
  mpz_import (r, nl, -1, 4, 0, 0, &h[i * nl]);
}

static bool
check (const char *what, const mpz_t got, mpz_t want, const mpz_t N,
       const mpz_t N2)
{
  mpz_mod (want, want, N);
  mpz_t g;
  mpz_init (g);
  mpz_mod (g, got, N);
  const bool ok = mpz_cmp (g, want) == 0 && mpz_sgn (got) >= 0
                  && mpz_cmp (got, N2) < 0;
  mpz_clear (g);
  if (!ok && failures < 10)
    printf ("  %s wrong\n", what);
  return ok;
}

static void
test (int bits, bool ntt)
{
  if (ntt)
    setenv ("GMPECM_ROOTS_NTT", "1", 1);
  else
    setenv ("GMPECM_ROOTS_NTT", "0", 1);
  const uint32_t nl = (bits + 4 + 31) / 32;
  mpz_t N, N2, i0, a, b, e;
  mpz_inits (N, N2, i0, a, b, e, NULL);
  for (;;)
    {
      mpz_urandomb (N, rs, bits);
      mpz_setbit (N, bits - 1);
      mpz_setbit (N, 0);
      bool ok = true;
      for (unsigned long p = 3; p < 10000 && ok; p += 2)
        ok = !mpz_divisible_ui_p (N, p);
      if (ok)
        break;
    }
  mpz_mul_2exp (N2, N, 1);
  mpz_set_ui (i0, 7);
  gpu_roots *G = gpu_roots_create (N, 0, 0, 2, 2310, 13, i0, 16, 1, nl, 2, 0);
  assert (G != NULL);
  int bad = 0;
  for (uint32_t d = 1; d <= 32; d *= 2)
    {
      /* tree: 6 pairs */
      const size_t pairs = 6, count = pairs * 2 * d;
      std::vector<uint32_t> in (count * nl), out (count * nl);
      std::vector<__mpz_struct> v (count);
      for (size_t i = 0; i < count; i++)
        {
          mpz_init (&v[i]);
          mpz_urandomm (&v[i], rs, N2);
          mpz_export (&in[i * nl], NULL, -1, 4, 0, 0, &v[i]);
        }
      uint32_t *din, *dout;
      CUDA_CHECK (cudaMalloc (&din, in.size () * 4));
      CUDA_CHECK (cudaMalloc (&dout, out.size () * 4));
      CUDA_CHECK (cudaMemcpy (din, in.data (), in.size () * 4, cudaMemcpyHostToDevice));
      gpu_roots_tree (G, dout, din, count, d);
      CUDA_CHECK (cudaDeviceSynchronize ());
      CUDA_CHECK (cudaMemcpy (out.data (), dout, out.size () * 4, cudaMemcpyDeviceToHost));
      for (size_t p = 0; p < pairs; p++)
        for (size_t k = 0; k < 2 * d; k++)
          {
            mpz_set_ui (e, 0);
            for (size_t i = 0; i < d; i++)
              if (k >= i && k - i < d)
                mpz_addmul (e, &v[p * 2 * d + i], &v[p * 2 * d + d + k - i]);
            if (k >= d)
              {
                mpz_add (e, e, &v[p * 2 * d + k - d]);
                mpz_add (e, e, &v[p * 2 * d + d + k - d]);
              }
            get (a, out, p * 2 * d + k, nl);
            bad += !check ("tree", a, e, N, N2);
          }
      /* mp: 5 nodes of 2 half = 2 d values, children in TREE */
      const uint32_t half = d;
      const size_t nodes = 5, cnt = nodes * 2 * half;
      std::vector<uint32_t> tv (cnt * nl), tr (cnt * nl), mo (cnt * nl);
      std::vector<__mpz_struct> T (cnt), C (cnt);
      for (size_t i = 0; i < cnt; i++)
        {
          mpz_init (&T[i]);
          mpz_init (&C[i]);
          mpz_urandomm (&T[i], rs, N2);
          mpz_urandomm (&C[i], rs, N2);
          mpz_export (&tv[i * nl], NULL, -1, 4, 0, 0, &T[i]);
          mpz_export (&tr[i * nl], NULL, -1, 4, 0, 0, &C[i]);
        }
      uint32_t *dt, *dc, *dm;
      CUDA_CHECK (cudaMalloc (&dt, tv.size () * 4));
      CUDA_CHECK (cudaMalloc (&dc, tr.size () * 4));
      CUDA_CHECK (cudaMalloc (&dm, mo.size () * 4));
      CUDA_CHECK (cudaMemcpy (dt, tv.data (), tv.size () * 4, cudaMemcpyHostToDevice));
      CUDA_CHECK (cudaMemcpy (dc, tr.data (), tr.size () * 4, cudaMemcpyHostToDevice));
      gpu_roots_mp (G, dm, dt, dc, cnt, half);
      CUDA_CHECK (cudaDeviceSynchronize ());
      CUDA_CHECK (cudaMemcpy (mo.data (), dm, mo.size () * 4, cudaMemcpyDeviceToHost));
      for (size_t nd = 0; nd < nodes; nd++)
        for (uint32_t r = 0; r < 2 * half; r++)
          {
            const uint32_t i = r % half;
            /* r < half: child R (the second half), else L */
            const size_t ch = nd * 2 * half + (r < half ? half : 0);
            mpz_set_ui (e, 0);
            for (uint32_t j = 0; j < half; j++)
              mpz_addmul (e, &C[ch + j], &T[nd * 2 * half + i + j]);
            mpz_add (e, e, &T[nd * 2 * half + half + i]);
            get (a, mo, nd * 2 * half + r, nl);
            bad += !check ("mp", a, e, N, N2);
          }
      for (auto &z : v)
        mpz_clear (&z);
      for (size_t i = 0; i < cnt; i++)
        {
          mpz_clear (&T[i]);
          mpz_clear (&C[i]);
        }
      cudaFree (din);
      cudaFree (dout);
      cudaFree (dt);
      cudaFree (dc);
      cudaFree (dm);
    }
  printf ("%6d bits, %s: %s\n", bits, G->nt ? "NTT" : "schoolbook",
          bad ? "FAIL" : "ok");
  fflush (stdout);
  failures += bad != 0;
  gpu_roots_destroy (G);
  mpz_clears (N, N2, i0, a, b, e, NULL);
}

int
main ()
{
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 99);
  for (int bits : { 1000, 4096, 9000, 24000, 33000, 65536 })
    for (bool ntt : { false, true })
      test (bits, ntt);
  printf ("%s\n", failures ? "FAILED" : "ALL OK");
  return failures != 0;
}
