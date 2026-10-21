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
  int l, nl;
  const uint32_t *d_N;
  uint32_t *d_N2, *d_one;
  uint32_t *t[PA_NTMP];             /* scratch, 2n numbers each */
  const uint32_t *F;                /* from gpu_poly_prepare */
  uint32_t *Iprep, *Fprep, *invFprep;
  std::vector<void *> dev;
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

/* r[i] = a[seg - 1 - i], i < seg, for BATCH segments (r != a) */
__global__ void
k_zn_rev (uint32_t *r, size_t rs, const uint32_t *a, size_t as, size_t seg,
          size_t batch, int nl)
{
  for (size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
       x < seg * batch; x += (size_t) gridDim.x * blockDim.x)
    {
      const size_t j = x / seg, i = x % seg;
      uint32_t *rr = r + (j * rs + i) * nl;
      const uint32_t *aa = a + (j * as + seg - 1 - i) * nl;
      for (int w = 0; w < nl; w++)
        rr[w] = aa[w];
    }
}

/* level of degree 2d: node j, coefficients d .. 2d-1 += A + B, the nodes
   A, B of degree d below it (reduced below 2N) */
__global__ void
k_monic_fix (uint32_t *L2, const uint32_t *L, size_t d, size_t nodes,
             const uint32_t *N2, int nl)
{
  for (size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
       x < d * nodes; x += (size_t) gridDim.x * blockDim.x)
    {
      const size_t j = x / d, i = x % d;
      uint32_t *r = L2 + (j * 2 * d + d + i) * nl;
      const uint32_t *a = L + (j * 2 * d + i) * nl;
      const uint32_t *b = L + (j * 2 * d + d + i) * nl;
      uint64_t c = 0;
      for (int w = 0; w < nl; w++)
        {
          c += (uint64_t) r[w] + a[w] + b[w];
          r[w] = (uint32_t) c;
          c >>= 32;
        }
      zn_below2n (r, N2, nl, 2);
    }
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
  k_zn_lin<<<blocks_for (count * batch), 256>>> (r, rs, a, as, b, bs, count,
                                                 batch, mode, A->d_N2, A->nl);
  CUDA_CHECK (cudaGetLastError ());
}

static void
zn_rev (gpu_polyalg *A, uint32_t *r, size_t rs, const uint32_t *a, size_t as,
        size_t seg, size_t batch)
{
  k_zn_rev<<<blocks_for (seg * batch), 256>>> (r, rs, a, as, seg, batch,
                                               A->nl);
  CUDA_CHECK (cudaGetLastError ());
}

static void
zn_copy (gpu_polyalg *A, uint32_t *r, const uint32_t *a, size_t count)
{
  CUDA_CHECK (cudaMemcpy (r, a, count * A->nl * sizeof (uint32_t),
                          cudaMemcpyDeviceToDevice));
}

void
gpu_zn_reduce (gpu_polyalg *A, uint32_t *x, size_t count)
{
  k_zn_reduce<<<blocks_for (count), 256>>> (x, count, A->d_N, A->nl);
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
gpu_polyalg_create (gpu_rns *R, size_t n)
{
  gpu_polyalg *A = new gpu_polyalg;
  A->R = R;
  A->n = n;
  A->l = lg2 (n);
  assert (((size_t) 1 << A->l) == n);
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
      CUDA_CHECK (cudaMalloc ((void **) &A->t[i], 2 * n * nl * 4));
      A->dev.push_back (A->t[i]);
    }
  A->F = NULL;
  A->Iprep = A->Fprep = A->invFprep = NULL;
  return A;
}

void
gpu_polyalg_destroy (gpu_polyalg *A)
{
  for (void *d : A->dev)
    CUDA_CHECK (cudaFree (d));
  delete A;
}

/* ---- algorithms ------------------------------------------------------- */

void
gpu_poly_from_roots (gpu_polyalg *A, uint32_t *F, uint32_t **tree,
                     const uint32_t *roots)
{
  const size_t n = A->n;
  const int l = A->l, nl = A->nl;
  if (n == 1)
    {
      zn_copy (A, F, roots, 1);
      if (tree != NULL)
        zn_copy (A, tree[0], roots, 1);
      return;
    }
  const uint32_t *cur = roots;
  if (tree != NULL)
    {
      zn_copy (A, tree[l - 1], roots, n);
      cur = tree[l - 1];
    }
  int ping = 0;
  for (size_t d = 1; d < n; d *= 2)
    {
      const int lgd = lg2 (2 * d);
      uint32_t *dst = 2 * d == n ? F
                      : tree != NULL ? tree[l - 1 - lgd] : A->t[ping ^= 1];
      gpu_rns_conv (A->R, dst, 2 * d, 0, 2 * d, cur, 2 * d, d,
                    cur + d * nl, 2 * d, d, NULL, lgd, (int) (n / (2 * d)));
      k_monic_fix<<<blocks_for (n / 2), 256>>> (dst, cur, d, n / (2 * d),
                                                A->d_N2, nl);
      CUDA_CHECK (cudaGetLastError ());
      cur = dst;
    }
}

void
gpu_poly_inverse (gpu_polyalg *A, uint32_t *I, const uint32_t *F)
{
  const size_t n = A->n;
  const int nl = A->nl;
  uint32_t *revF = A->t[0], *E = A->t[1], *D = A->t[2];
  zn_copy (A, revF, A->d_one, 1);
  if (n > 1)
    zn_rev (A, revF + nl, 0, F + nl, 0, n - 1, 1);   /* revF[t] = F[n - t] */
  zn_copy (A, I, A->d_one, 1);
  for (size_t k = 1; k < n; k *= 2)
    {
      const int lg = lg2 (2 * k);
      /* E = coefficients k .. 2k-1 of rev(F) I_k */
      gpu_rns_conv (A->R, E, k, k, k, revF, 2 * k, 2 * k, I, k, k, NULL, lg, 1);
      /* I[k .. 2k-1] = -(I_k E mod x^k) */
      gpu_rns_conv (A->R, D, k, 0, k, I, k, k, E, k, k, NULL, lg, 1);
      zn_lin (A, I + k * nl, 0, D, 0, D, 0, k, 1, ZN_NEG);
    }
}

void
gpu_poly_prepare (gpu_polyalg *A, const uint32_t *F, const uint32_t *I)
{
  const size_t n = A->n;
  const int lg = A->l + 1, P = gpu_rns_primes (A->R);
  assert (lg <= gpu_rns_lg (A->R));
  const size_t words = (size_t) P << lg;
  if (A->Iprep == NULL)
    {
      CUDA_CHECK (cudaMalloc ((void **) &A->Iprep, 3 * words * 4));
      A->dev.push_back (A->Iprep);
      A->Fprep = A->Iprep + words;
      A->invFprep = A->Fprep + words;
    }
  A->F = F;
  gpu_rns_prep (A->R, A->Iprep, I, n - 1 > 0 ? n - 1 : 1, lg);
  gpu_rns_prep (A->R, A->Fprep, F, n, lg);
  zn_rev (A, A->t[0], 0, I, 0, n, 1);             /* invF = rev (I) */
  gpu_rns_prep (A->R, A->invFprep, A->t[0], n, lg);
}

void
gpu_poly_reduce (gpu_polyalg *A, uint32_t *Rm, const uint32_t *A2)
{
  const size_t n = A->n;
  const int lg = A->l + 1, nl = A->nl;
  uint32_t *ra = A->t[0], *rq = A->t[1], *Q = A->t[2], *QF = A->t[3];
  if (n == 1)
    {
      /* A2 is a constant */
      if (Rm != A2)
        zn_copy (A, Rm, A2, 1);
      return;
    }
  zn_rev (A, ra, 0, A2 + n * nl, 0, n - 1, 1);    /* rev (high A) */
  gpu_rns_conv (A->R, rq, 0, 0, n - 1, ra, 0, n - 1, NULL, 0, 0, A->Iprep,
                lg, 1);
  zn_rev (A, Q, 0, rq, 0, n - 1, 1);
  gpu_rns_conv (A->R, QF, 0, 0, n, Q, 0, n - 1, NULL, 0, 0, A->Fprep, lg, 1);
  zn_lin (A, Rm, 0, A2, 0, QF, 0, n, 1, ZN_SUB);
}

void
gpu_poly_mulmod (gpu_polyalg *A, uint32_t *H, const uint32_t *G, int first)
{
  const size_t n = A->n;
  if (first)
    {
      zn_lin (A, H, 0, G, 0, A->F, 0, n, 1, ZN_SUB);
      return;
    }
  uint32_t *Gp = A->t[4], *prod = A->t[5];
  zn_lin (A, Gp, 0, G, 0, A->F, 0, n, 1, ZN_SUB);
  gpu_rns_conv (A->R, prod, 0, 0, 2 * n - 1, Gp, 0, n, H, 0, n, NULL,
                A->l + 1, 1);
  gpu_poly_reduce (A, H, prod);
}

void
gpu_poly_tpeval (gpu_polyalg *A, uint32_t *vals, const uint32_t *H,
                 uint32_t *const *tree)
{
  const size_t n = A->n;
  const int l = A->l, nl = A->nl;
  uint32_t *U = A->t[0], *T = A->t[1], *revT = A->t[2], *T2 = A->t[3];
  if (n == 1)
    {
      zn_copy (A, vals, H, 1);
      return;
    }
  /* T = rev of the coefficients n-1 .. 2n-2 of invF H */
  gpu_rns_conv (A->R, U, 0, n - 1, n, H, 0, n, NULL, 0, 0, A->invFprep,
                l + 1, 1);
  zn_rev (A, T, 0, U, 0, n, 1);
  for (int t = 0; t < l; t++)
    {
      const size_t k = n >> t, half = k / 2, nodes = n / k;
      const int lgk = lg2 (k);
      zn_rev (A, revT, half, tree[t], half, half, n / half);
      /* left halves: MP (right child, b); right halves: MP (left child, b) */
      gpu_rns_conv (A->R, T2, k, half - 1, half, revT + half * nl, k, half,
                    T, k, k, NULL, lgk, (int) nodes);
      gpu_rns_conv (A->R, T2 + half * nl, k, half - 1, half, revT, k, half,
                    T, k, k, NULL, lgk, (int) nodes);
      zn_lin (A, T2, k, T2, k, T + half * nl, k, half, nodes, ZN_ADD);
      zn_lin (A, T2 + half * nl, k, T2 + half * nl, k, T + half * nl, k, half,
              nodes, ZN_ADD);
      std::swap (T, T2);
    }
  zn_copy (A, vals, T, n);
}
