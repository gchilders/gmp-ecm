/* gpu_polyalg.cu: polynomial algorithms of the stage 2 FFT continuation on
   the GPU (see gpu_polyalg.h).

  Copyright 2026 Greg Childers

  This program is free software; you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by the
  Free Software Foundation; either version 3 of the License, or (at your
  option) any later version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
  more details.

  You should have received a copy of the GNU General Public License along
  with this program; see the file COPYING.  If not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02111-1307, USA.
*/

/* The algorithms are those of GMP-ECM's stage 2 (stage2.c, listz.c,
 * polyeval.c, ecm_ntt.c), with every product a batched cyclic convolution
 * in RNS form (gpu_rns_conv) and every result back in Z/N:
 *
 * - product tree: level by level, the nodes of degree 2d are the products
 *   (x^d + A)(x^d + B) = x^2d + x^d (A + B) + AB of the pairs of nodes of
 *   degree d: AB is a cyclic convolution of length 2d (no wrap), then the
 *   x^d (A + B) part is added;
 * - I = 1/rev(F) mod x^n by Newton: I_2k = I_k - x^k (I_k E mod x^k) with
 *   E the coefficients k .. 2k-1 of rev(F) I_k;
 * - A mod F for deg A <= 2n-2: rev(Q) = rev(high A) I mod x^(n-1), then
 *   R = low(A) - low(Q F), F's leading term not contributing below x^n;
 * - transposed multipoint evaluation (polyeval_tellegen): T = rev of the
 *   coefficients n-1 .. 2n-2 of invF H, invF = rev(I); then down the tree,
 *   a node's values b (k of them, children L and R of degree k/2) become
 *   MP(R, b) + b[k/2 ..] for L's half and MP(L, b) + b[k/2 ..] for R's,
 *   MP(a, b)[i] = sum_j a_j b_(i+j), the middle product: entries
 *   k/2 - 1 .. k - 2 of the cyclic convolution of length k of rev(a) and b.
 *   At the leaves the values are H(-r_i). */

#include "gpu_polyalg.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include "cudacommon.h"

#define PA_NTMP 6

struct gpu_polyalg {
  gpu_rns *R;
  size_t n;
  int B;                            /* polynomials side by side */
  int l, nl;
  const uint32_t *d_N;
  uint32_t *d_N2, *d_one;
  uint32_t *t[PA_NTMP];             /* scratch, 2n B numbers each */
  const uint32_t *F;                /* from gpu_poly_prepare */
  uint32_t *Iprep, *Fprep, *invFprep;
  uint32_t *Tprep;                  /* B P n words: an operand's transforms
                                       (gpu_poly_inverse, gpu_poly_tpeval) */
  /* product tree levels up to degree lowmax by lowmul (gpu_polyalg_lowmul) */
  void (*lowmul) (void *, uint32_t *, const uint32_t *, size_t, size_t);
  void *lowctx;
  size_t lowmax;
  /* transposed evaluation levels up to children of degree mpmax by lowmp */
  void (*lowmp) (void *, uint32_t *, const uint32_t *, const uint32_t *,
                 size_t, size_t);
  size_t mpmax;
  std::vector<void *> dev;
  /* kept transforms of F's tree (gpu_polyalg_treecache): kx[t], ky[t] the
     left and right children of the nodes of level t of the transposed
     evaluation (children of degree n / 2^(t+1)), as the tree build made
     them; NULL where not kept */
  std::vector<uint32_t *> kx, ky;
};

/* ---- numbers ---------------------------------------------------------- */

/* Every stored number is below 2N (the conversions give that); the
   operations below keep it so, which is what makes a - b + 2N >= 0. */

/* x <- x - 2N while x >= 2N (at most ITERS times) */
static __device__ void
zn_below2n (uint32_t *x, const uint32_t *N2, int nl, int iters)
{
  for (int it = 0; it < iters; it++)
    {
      int ge = 1;
      for (int w = nl - 1; w >= 0; w--)
        if (x[w] != N2[w])
          {
            ge = x[w] > N2[w];
            break;
          }
      if (!ge)
        return;
      int64_t c = 0;
      for (int w = 0; w < nl; w++)
        {
          c += (int64_t) x[w] - N2[w];
          x[w] = (uint32_t) c;
          c >>= 32;
        }
    }
}

enum { ZN_ADD, ZN_SUB, ZN_NEG };

/* r = a + b, a + 2N - b, or 2N - a, reduced below 2N, for COUNT numbers of
   each of BATCH items (item j at r + j RS numbers, etc.) */
__global__ void
k_zn_lin (uint32_t *r, size_t rs, const uint32_t *a, size_t as,
          const uint32_t *b, size_t bs, size_t count, size_t batch, int mode,
          const uint32_t *N2, int nl)
{
  for (size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
       x < count * batch; x += (size_t) gridDim.x * blockDim.x)
    {
      const size_t j = x / count, e = x % count;
      uint32_t *rr = r + (j * rs + e) * nl;
      const uint32_t *aa = a + (j * as + e) * nl;
      const uint32_t *bb = b + (j * bs + e) * nl;
      if (mode == ZN_ADD)
        {
          uint64_t c = 0;
          for (int w = 0; w < nl; w++)
            {
              c += (uint64_t) aa[w] + bb[w];
              rr[w] = (uint32_t) c;
              c >>= 32;
            }
        }
      else
        {
          /* x - y + 2N, x = a (SUB) or 0 (NEG), y = b (SUB) or a (NEG) */
          const uint32_t *y = mode == ZN_SUB ? bb : aa;
          int64_t c = 0;
          for (int w = 0; w < nl; w++)
            {
              c += (int64_t) (mode == ZN_SUB ? aa[w] : 0) - y[w] + N2[w];
              rr[w] = (uint32_t) c;
              c >>= 32;
            }
        }
      zn_below2n (rr, N2, nl, 2);
    }
}

/* The same through shared memory, tiles of TT = blockDim numbers (with
   2 TT (NL + 1) words), so that loads and stores are coalesced: a tile's
   words are moved by consecutive threads, then each thread does one
   number. */
__global__ void
k_zn_lin_t (uint32_t *r, size_t rs, const uint32_t *a, size_t as,
            const uint32_t *b, size_t bs, size_t count, size_t batch,
            int mode, const uint32_t *N2, int nl)
{
  extern __shared__ uint32_t sm[];
  const int TT = blockDim.x, st = nl + 1, tid = threadIdx.x;
  uint32_t *ta = sm, *tb = sm + TT * st;
  const size_t total = count * batch;
  for (size_t x0 = (size_t) blockIdx.x * TT; x0 < total;
       x0 += (size_t) gridDim.x * TT)
    {
      for (int k = tid; k < TT * nl; k += TT)
        {
          const int num = k / nl, w = k % nl;
          const size_t x = x0 + num;
          if (x < total)
            {
              const size_t j = x / count, e = x % count;
              ta[num * st + w] = a[(j * as + e) * nl + w];
              if (mode != ZN_NEG)
                tb[num * st + w] = b[(j * bs + e) * nl + w];
            }
        }
      __syncthreads ();
      if (x0 + tid < total)
        {
          uint32_t *aa = ta + tid * st;
          const uint32_t *bb = tb + tid * st;
          if (mode == ZN_ADD)
            {
              uint64_t c = 0;
              for (int w = 0; w < nl; w++)
                {
                  c += (uint64_t) aa[w] + bb[w];
                  aa[w] = (uint32_t) c;
                  c >>= 32;
                }
            }
          else
            {
              int64_t c = 0;
              for (int w = 0; w < nl; w++)
                {
                  c += mode == ZN_SUB ? (int64_t) aa[w] - bb[w] + N2[w]
                                      : (int64_t) N2[w] - aa[w];
                  aa[w] = (uint32_t) c;
                  c >>= 32;
                }
            }
          zn_below2n (aa, N2, nl, 2);
        }
      __syncthreads ();
      for (int k = tid; k < TT * nl; k += TT)
        {
          const int num = k / nl, w = k % nl;
          const size_t x = x0 + num;
          if (x < total)
            {
              const size_t j = x / count, e = x % count;
              r[(j * rs + e) * nl + w] = ta[num * st + w];
            }
        }
      __syncthreads ();
    }
}

/* r[i] = a[seg - 1 - i], i < seg, for BATCH segments (r != a), segment j
   of A into segment j ^ SWAP of R; one thread per limb */
__global__ void
k_zn_rev (uint32_t *r, size_t rs, const uint32_t *a, size_t as, size_t seg,
          size_t batch, int nl, int swap)
{
  for (size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
       x < seg * batch * nl; x += (size_t) gridDim.x * blockDim.x)
    {
      const size_t w = x % nl, e = x / nl, j = e / seg, i = e % seg;
      r[((j ^ swap) * rs + i) * nl + w] = a[(j * as + seg - 1 - i) * nl + w];
    }
}

/* r[j RS] = x, j < BATCH */
__global__ void
k_zn_bcast (uint32_t *r, size_t rs, const uint32_t *x, size_t batch, int nl)
{
  for (size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
       e < batch * nl; e += (size_t) gridDim.x * blockDim.x)
    r[(e / nl) * rs * nl + e % nl] = x[e % nl];
}

/* x <- x mod N for x < 16 N */
__global__ void
k_zn_reduce (uint32_t *xs, size_t count, const uint32_t *N, int nl)
{
  for (size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x; e < count;
       e += (size_t) gridDim.x * blockDim.x)
    {
      uint32_t *x = xs + e * nl;
      for (int it = 0; it < 16; it++)
        {
          int ge = 1;
          for (int w = nl - 1; w >= 0; w--)
            if (x[w] != N[w])
              {
                ge = x[w] > N[w];
                break;
              }
          if (!ge)
            break;
          int64_t c = 0;
          for (int w = 0; w < nl; w++)
            {
              c += (int64_t) x[w] - N[w];
              x[w] = (uint32_t) c;
              c >>= 32;
            }
        }
    }
}

static unsigned
blocks_for (size_t items)
{
  return (unsigned) std::max ((size_t) 1, std::min ((size_t) 4096,
                                                    (items + 255) / 256));
}

static void
zn_lin (gpu_polyalg *A, uint32_t *r, size_t rs, const uint32_t *a, size_t as,
        const uint32_t *b, size_t bs, size_t count, size_t batch, int mode)
{
  gpu_rns_prof_begin ();
  int tt = 128;
  while (tt > 32 && (size_t) 2 * tt * (A->nl + 1) * 4 > 48 * 1024)
    tt /= 2;
  const size_t smem = (size_t) 2 * tt * (A->nl + 1) * 4;
  if (smem <= 48 * 1024)
    {
      const size_t tiles = (count * batch + tt - 1) / tt;
      k_zn_lin_t<<<(unsigned) std::min (tiles, (size_t) 65536), tt, smem>>> (
          r, rs, a, as, b, bs, count, batch, mode, A->d_N2, A->nl);
    }
  else
    k_zn_lin<<<blocks_for (count * batch), 256>>> (r, rs, a, as, b, bs, count,
                                                   batch, mode, A->d_N2,
                                                   A->nl);
  CUDA_CHECK (cudaGetLastError ());
  gpu_rns_prof_end ();
}

static void
zn_rev (gpu_polyalg *A, uint32_t *r, size_t rs, const uint32_t *a, size_t as,
        size_t seg, size_t batch, int swap = 0)
{
  gpu_rns_prof_begin ();
  k_zn_rev<<<blocks_for (seg * batch * A->nl), 256>>> (r, rs, a, as, seg,
                                                       batch, A->nl, swap);
  CUDA_CHECK (cudaGetLastError ());
  gpu_rns_prof_end ();
}

static void
zn_copy (gpu_polyalg *A, uint32_t *r, const uint32_t *a, size_t count)
{
  gpu_rns_prof_begin ();
  CUDA_CHECK (cudaMemcpy (r, a, count * A->nl * sizeof (uint32_t),
                          cudaMemcpyDeviceToDevice));
  gpu_rns_prof_end ();
}

void
gpu_zn_reduce (gpu_polyalg *A, uint32_t *x, size_t count)
{
  k_zn_reduce<<<blocks_for (count), 256>>> (x, count, A->d_N, A->nl);
  CUDA_CHECK (cudaGetLastError ());
}

static void
zn_bcast (gpu_polyalg *A, uint32_t *r, size_t rs, const uint32_t *x)
{
  k_zn_bcast<<<blocks_for (A->B * A->nl), 256>>> (r, rs, x, A->B, A->nl);
  CUDA_CHECK (cudaGetLastError ());
}

static int
lg2 (size_t m)
{
  int l = 0;
  while (((size_t) 1 << l) < m)
    l++;
  return l;
}

/* ---- context ---------------------------------------------------------- */

gpu_polyalg *
gpu_polyalg_create (gpu_rns *R, size_t n, int B)
{
  gpu_polyalg *A = new gpu_polyalg;
  A->R = R;
  A->n = n;
  A->B = B;
  A->l = lg2 (n);
  assert (((size_t) 1 << A->l) == n && B >= 1);
  A->nl = gpu_rns_limbs (R);
  A->d_N = gpu_rns_dev_N (R);
  const int nl = A->nl;
  std::vector<uint32_t> N (nl), N2 (nl), one (nl, 0);
  CUDA_CHECK (cudaMemcpy (N.data (), A->d_N, nl * 4, cudaMemcpyDeviceToHost));
  uint32_t c = 0;
  for (int w = 0; w < nl; w++)
    {
      N2[w] = (N[w] << 1) | c;
      c = N[w] >> 31;
    }
  one[0] = 1;
  CUDA_CHECK (cudaMalloc ((void **) &A->d_N2, nl * 4));
  CUDA_CHECK (cudaMalloc ((void **) &A->d_one, nl * 4));
  CUDA_CHECK (cudaMemcpy (A->d_N2, N2.data (), nl * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK (cudaMemcpy (A->d_one, one.data (), nl * 4, cudaMemcpyHostToDevice));
  A->dev.push_back (A->d_N2);
  A->dev.push_back (A->d_one);
  for (int i = 0; i < PA_NTMP; i++)
    {
      CUDA_CHECK (cudaMalloc ((void **) &A->t[i], 2 * n * B * nl * 4));
      A->dev.push_back (A->t[i]);
    }
  A->F = NULL;
  A->Iprep = A->Fprep = A->invFprep = NULL;
  if (A->l + 1 <= gpu_rns_lg (R))
    {
      /* gpu_poly_prepare's transforms, now so that gpu_polyalg_treecache
         budgets with them */
      const size_t words = (size_t) B * gpu_rns_primes (R) << (A->l + 1);
      CUDA_CHECK (cudaMalloc ((void **) &A->Iprep, 3 * words * 4));
      A->dev.push_back (A->Iprep);
      A->Fprep = A->Iprep + words;
      A->invFprep = A->Fprep + words;
    }
  CUDA_CHECK (cudaMalloc ((void **) &A->Tprep,
                          (size_t) gpu_rns_primes (R) * n * B * 4));
  A->dev.push_back (A->Tprep);
  A->lowmul = NULL;
  A->lowctx = NULL;
  A->lowmax = 0;
  A->lowmp = NULL;
  A->kx.assign (A->l > 0 ? A->l : 1, NULL);
  A->ky.assign (A->l > 0 ? A->l : 1, NULL);
  A->mpmax = 0;
  return A;
}

void
gpu_polyalg_lowmul (gpu_polyalg *A,
                    void (*fn) (void *, uint32_t *, const uint32_t *, size_t,
                                size_t),
                    void *ctx, size_t dmax)
{
  A->lowmul = fn;
  A->lowctx = ctx;
  A->lowmax = fn != NULL ? dmax : 0;
}

void
gpu_polyalg_lowmp (gpu_polyalg *A,
                   void (*fn) (void *, uint32_t *, const uint32_t *,
                               const uint32_t *, size_t, size_t),
                   void *ctx, size_t hmax)
{
  A->lowmp = fn;
  A->lowctx = ctx;
  A->mpmax = fn != NULL ? hmax : 0;
}

void
gpu_polyalg_treecache (gpu_polyalg *A, size_t bytes)
{
  /* each level: the transforms of its nodes' two children, P rows of 2d
     for each pair, from the top down to the levels done in Z/N */
  const size_t n = A->n, nb = n * A->B;
  const size_t level = (size_t) 2 * gpu_rns_primes (A->R) * nb * 4;
  for (size_t d = n / 2; d > A->lowmax && d > A->mpmax; d /= 2)
    {
      const int t = A->l - 1 - lg2 (d);
      if (A->kx[t] != NULL || bytes < level)
        continue;
      CUDA_CHECK (cudaMalloc ((void **) &A->kx[t], level / 2));
      CUDA_CHECK (cudaMalloc ((void **) &A->ky[t], level / 2));
      A->dev.push_back (A->kx[t]);
      A->dev.push_back (A->ky[t]);
      bytes -= level;
    }
}

int
gpu_polyalg_kept (const gpu_polyalg *A)
{
  int c = 0;
  for (const uint32_t *p : A->kx)
    c += p != NULL;
  return c;
}

void
gpu_polyalg_destroy (gpu_polyalg *A)
{
  for (void *d : A->dev)
    CUDA_CHECK (cudaFree (d));
  delete A;
}

/* ---- algorithms ------------------------------------------------------- */

/* Every function works on the B polynomials side by side (polynomial b's n
   numbers at b n); product trees are then B trees side by side, level by
   level, since the nodes of a level never straddle two polynomials. */

void
gpu_poly_from_roots (gpu_polyalg *A, uint32_t *F, uint32_t **tree,
                     const uint32_t *roots)
{
  const size_t n = A->n, nb = n * A->B;
  const int l = A->l, nl = A->nl;
  if (n == 1)
    {
      zn_copy (A, F, roots, nb);
      if (tree != NULL)
        zn_copy (A, tree[0], roots, nb);
      return;
    }
  const uint32_t *cur = roots;
  if (tree != NULL)
    {
      zn_copy (A, tree[l - 1], roots, nb);
      cur = tree[l - 1];
    }
  int ping = 0;
  for (size_t d = 1; d < n; d *= 2)
    {
      const int lgd = lg2 (2 * d);
      const int t = l - 1 - lg2 (d);
      uint32_t *dst = 2 * d == n ? F
                      : tree != NULL ? tree[l - 1 - lgd] : A->t[ping ^= 1];
      if (d <= A->lowmax)
        {
          gpu_rns_prof_begin ();
          A->lowmul (A->lowctx, dst, cur, nb, d);
          gpu_rns_prof_end ();
        }
      else
        gpu_rns_conv_monic (A->R, dst, 2 * d, cur, 2 * d, cur + d * nl,
                            2 * d, d, (int) (nb / (2 * d)),
                            tree != NULL ? A->kx[t] : NULL,
                            tree != NULL ? A->ky[t] : NULL);
      cur = dst;
    }
}

void
gpu_poly_inverse (gpu_polyalg *A, uint32_t *I, const uint32_t *F)
{
  const size_t n = A->n;
  const int nl = A->nl, B = A->B;
  uint32_t *revF = A->t[0], *E = A->t[1], *D = A->t[2];
  zn_bcast (A, revF, n, A->d_one);
  if (n > 1)
    zn_rev (A, revF + nl, n, F + nl, n, n - 1, B);  /* revF[t] = F[n - t] */
  zn_bcast (A, I, n, A->d_one);
  for (size_t k = 1; k < n; k *= 2)
    {
      const int lg = lg2 (2 * k);
      /* I_k's transforms, for both products */
      gpu_rns_prep (A->R, A->Tprep, I, n, k, lg, B);
      /* E = coefficients k .. 2k-1 of rev(F) I_k */
      gpu_rns_conv_prep (A->R, E, k, k, k, revF, n, 2 * k, A->Tprep, 1, lg,
                         B);
      /* I[k .. 2k-1] = -(I_k E mod x^k) */
      gpu_rns_conv_prep (A->R, D, k, 0, k, E, k, k, A->Tprep, 1, lg, B);
      zn_lin (A, I + k * nl, n, D, k, D, k, k, B, ZN_NEG);
    }
}

void
gpu_poly_prepare (gpu_polyalg *A, const uint32_t *F, const uint32_t *I)
{
  const size_t n = A->n;
  const int lg = A->l + 1, P = gpu_rns_primes (A->R), B = A->B;
  assert (lg <= gpu_rns_lg (A->R));
  const size_t words = (size_t) B * P << lg;
  if (A->Iprep == NULL)
    {
      CUDA_CHECK (cudaMalloc ((void **) &A->Iprep, 3 * words * 4));
      A->dev.push_back (A->Iprep);
      A->Fprep = A->Iprep + words;
      A->invFprep = A->Fprep + words;
    }
  A->F = F;
  gpu_rns_prep (A->R, A->Iprep, I, n, n - 1 > 0 ? n - 1 : 1, lg, B);
  gpu_rns_prep (A->R, A->Fprep, F, n, n, lg, B);
  zn_rev (A, A->t[0], n, I, n, n, B);             /* invF = rev (I) */
  gpu_rns_prep (A->R, A->invFprep, A->t[0], n, n, lg, B);
}

void
gpu_poly_reduce (gpu_polyalg *A, uint32_t *Rm, const uint32_t *A2)
{
  const size_t n = A->n;
  const int lg = A->l + 1, nl = A->nl, B = A->B;
  uint32_t *ra = A->t[0], *rq = A->t[1], *Q = A->t[2], *QF = A->t[3];
  if (n == 1)
    {
      /* each A2 is a constant */
      zn_rev (A, Rm, 1, A2, 2, 1, B);
      return;
    }
  /* each A2 has 2n numbers (2n - 1 used), Rm n */
  zn_rev (A, ra, n, A2 + n * nl, 2 * n, n - 1, B);   /* rev (high A) */
  gpu_rns_conv (A->R, rq, n, 0, n - 1, ra, n, n - 1, NULL, 1, 0, A->Iprep,
                lg, B);
  zn_rev (A, Q, n, rq, n, n - 1, B);
  gpu_rns_conv (A->R, QF, n, 0, n, Q, n, n - 1, NULL, 1, 0, A->Fprep, lg, B);
  zn_lin (A, Rm, n, A2, 2 * n, QF, n, n, B, ZN_SUB);
}

void
gpu_poly_mulmod (gpu_polyalg *A, uint32_t *H, const uint32_t *G, int first)
{
  const size_t n = A->n;
  const int B = A->B;
  if (first)
    {
      zn_lin (A, H, n, G, n, A->F, n, n, B, ZN_SUB);
      return;
    }
  uint32_t *Gp = A->t[4], *prod = A->t[5];
  zn_lin (A, Gp, n, G, n, A->F, n, n, B, ZN_SUB);
  gpu_rns_conv (A->R, prod, 2 * n, 0, 2 * n - 1, Gp, n, n, H, n, n, NULL,
                A->l + 1, B);
  gpu_poly_reduce (A, H, prod);
}

void
gpu_poly_prodvals (gpu_polyalg *A, uint32_t *out, uint32_t *vals)
{
  const int nl = A->nl, B = A->B;
  /* products of pairs, as cyclic convolutions of length 1, level by level
     (a level's pairs never straddle two polynomials) */
  size_t cnt = A->n;
  uint32_t *cur = vals;
  int ping = 0;
  while (cnt > 1)
    {
      uint32_t *dst = cnt == 2 ? out : A->t[ping ^= 1];
      gpu_rns_conv (A->R, dst, 1, 0, 1, cur, 2, 1, cur + nl, 2, 1, NULL, 0,
                    (int) (B * cnt / 2));
      cur = dst;
      cnt /= 2;
    }
  if (A->n == 1)
    zn_copy (A, out, vals, B);
}

void
gpu_poly_tpeval (gpu_polyalg *A, uint32_t *vals, const uint32_t *H,
                 uint32_t *const *tree)
{
  const size_t n = A->n, nb = n * A->B;
  const int l = A->l, nl = A->nl, B = A->B;
  uint32_t *U = A->t[0], *T = A->t[1], *revT = A->t[2], *T2 = A->t[3];
  if (n == 1)
    {
      zn_copy (A, vals, H, nb);
      return;
    }
  /* T = rev of the coefficients n-1 .. 2n-2 of invF H */
  gpu_rns_conv (A->R, U, n, n - 1, n, H, n, n, NULL, 1, 0, A->invFprep,
                l + 1, B);
  zn_rev (A, T, n, U, n, n, B);
  for (int t = 0; t < l; t++)
    {
      const size_t k = n >> t, half = k / 2, nodes = nb / k;
      const int lgk = lg2 (k);
      if (half <= A->mpmax)
        {
          gpu_rns_prof_begin ();
          A->lowmp (A->lowctx, T2, T, tree[t], nb, half);
          gpu_rns_prof_end ();
          std::swap (T, T2);
          continue;
        }
      /* the children reversed, each node's two swapped, so that item 2j
         (j's right child) gives left half j and item 2j + 1 right half j:
         left halves MP (right child, b), right halves MP (left child, b),
         with b's transforms shared by the two */
      gpu_rns_prep (A->R, A->Tprep, T, k, k, lgk, (int) nodes);
      if (A->kx[t] != NULL)
        /* the children's transforms kept from the tree build */
        gpu_rns_corr_kept (A->R, T2, half, half, A->Tprep, A->kx[t],
                           A->ky[t], half, (int) nodes);
      else
        {
          zn_rev (A, revT, half, tree[t], half, half, nb / half, 1);
          gpu_rns_conv_prep (A->R, T2, half, half - 1, half, revT, half,
                             half, A->Tprep, 2, lgk, (int) (2 * nodes));
        }
      zn_lin (A, T2, k, T2, k, T + half * nl, k, half, nodes, ZN_ADD);
      zn_lin (A, T2 + half * nl, k, T2 + half * nl, k, T + half * nl, k, half,
              nodes, ZN_ADD);
      std::swap (T, T2);
    }
  zn_copy (A, vals, T, nb);
}
