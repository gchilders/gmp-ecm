/* gpu_mont.cuh: multi-precision Montgomery arithmetic for GPU ECM stage 1.

Copyright 2026 Greg Childers

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/* An n-limb number (32-bit limbs) is spread over a group of TPI consecutive
 * threads, L limbs per thread, n = TPI * L.  Thread t holds limbs
 * [t*L, t*L + L).  L is a compile-time constant; TPI is chosen at run time
 * (any power of two up to 32: lanes of one warp, gm_group), except that
 * TPI == 1 ("solo", one curve per thread) is a separate compile-time
 * specialisation with no shuffles.  Wide groups of TPI = 64 or 128 threads
 * (gm_wgroup) span warps of one block and exchange what crosses a warp
 * through shared memory; the functions below take either kind of group.
 *
 * All arithmetic is lazy: with R = 2^(32n) and 16N <= R, Montgomery products
 * of inputs whose product is below 16N^2 come out below 2N, so no
 * comparisons or conditional subtractions against N are ever needed.  See
 * the bounds in gpu_stage1.cu.
 */

#ifndef _GPU_MONT_CUH
#define _GPU_MONT_CUH 1

#include <stdint.h>

#define GM_DEV __device__ __forceinline__
#define GM_FULL 0xffffffffu

/* ---- PTX carry-chain primitives ------------------------------------- */

GM_DEV uint32_t add_cc (uint32_t a, uint32_t b)
{ uint32_t r; asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
GM_DEV uint32_t addc_cc (uint32_t a, uint32_t b)
{ uint32_t r; asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
GM_DEV uint32_t addc (uint32_t a, uint32_t b)
{ uint32_t r; asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
GM_DEV uint32_t sub_cc (uint32_t a, uint32_t b)
{ uint32_t r; asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
GM_DEV uint32_t subc_cc (uint32_t a, uint32_t b)
{ uint32_t r; asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
GM_DEV uint32_t subc (uint32_t a, uint32_t b)
{ uint32_t r; asm volatile ("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
GM_DEV uint32_t madlo_cc (uint32_t a, uint32_t b, uint32_t c)
{ uint32_t r; asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
GM_DEV uint32_t madclo_cc (uint32_t a, uint32_t b, uint32_t c)
{ uint32_t r; asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
GM_DEV uint32_t madhi_cc (uint32_t a, uint32_t b, uint32_t c)
{ uint32_t r; asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
GM_DEV uint32_t madchi_cc (uint32_t a, uint32_t b, uint32_t c)
{ uint32_t r; asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
GM_DEV uint32_t madhi (uint32_t a, uint32_t b, uint32_t c)
{ uint32_t r; asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
GM_DEV uint32_t madchi (uint32_t a, uint32_t b, uint32_t c)
{ uint32_t r; asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }

/* ---- lane group ------------------------------------------------------ */

struct gm_group {
  static constexpr bool wide = false;
  uint32_t tpi;   /* threads per instance (power of 2, <= 32) */
  uint32_t lane;  /* index of this thread inside its group */
  uint32_t base;  /* lane id (in the warp) of the group's lane 0 */
  uint32_t mask;  /* tpi low bits set */

  GM_DEV void init (uint32_t t)
  {
    uint32_t wl = threadIdx.x & 31;
    tpi = t;
    lane = wl & (t - 1);
    base = wl & ~(t - 1);
    mask = (t == 32) ? GM_FULL : ((1u << t) - 1);
  }

  /* This group's bits of a warp ballot. */
  GM_DEV uint32_t bits (uint32_t ballot) const { return (ballot >> base) & mask; }

  /* Given per-lane carry-out bits GEN (to lane+1) and per-lane PROP bits
   * (the lane would pass an incoming carry straight through), return the
   * carry into this lane and, in *top, the carry out of the whole group.
   * The same identity serves borrows (prop = "slice is all zeros"). */
  GM_DEV uint32_t resolve (uint32_t gen, uint32_t prop, uint32_t *top) const
  {
    uint32_t G = bits (__ballot_sync (GM_FULL, gen));
    uint32_t P = bits (__ballot_sync (GM_FULL, prop));
    uint64_t x = ((uint64_t) G << 1) + P;
    uint32_t C = (uint32_t) x ^ P;
    if (top)
      *top = (uint32_t) (x >> tpi) & 1;
    return (C >> lane) & 1;
  }

  /* V of the lane below (0 in lane 0) */
  GM_DEV uint32_t from_below (uint32_t v) const
  {
    uint32_t r = __shfl_up_sync (GM_FULL, v, 1, tpi);
    return lane == 0 ? 0 : r;
  }

  /* V of the lane above (the top lane gets lane 0's; callers pass zero
     there) */
  GM_DEV uint32_t from_above (uint32_t v) const
  {
    return __shfl_sync (GM_FULL, v, (lane + 1) & (tpi - 1), tpi);
  }

  /* V of lane SRC */
  GM_DEV uint32_t bcast (uint32_t v, uint32_t src) const
  {
    return __shfl_sync (GM_FULL, v, src, tpi);
  }
};

/* A wide group: TPI = 64 or 128 threads (whole warps) of one block, one
 * named barrier per group.  Shared memory per group (see gm_wshared_words):
 * room for one operand, then two phases of GM_WSLOTS slot words.  Each
 * synchronising operation writes the slots of the current phase, waits at
 * the barrier, reads, and flips the phase; so a slot is only rewritten
 * after a barrier that all of its readers have passed. */
#define GM_WIDE_MAXK 4  /* product steps per barrier, at most */
#define GM_WS_Q   0                       /* the q of 2K rows (warp 0) */
#define GM_WS_IN  (2 * GM_WIDE_MAXK)      /* 2K per warp: words crossing to
                                             the warp below */
#define GM_WS_RES (GM_WS_IN + 3 * 2 * GM_WIDE_MAXK)  /* 2 per warp: ballots */
#define GM_WS_ONE (GM_WS_RES + 8)         /* 1 per warp */
#define GM_WSLOTS (GM_WS_ONE + 4)

struct gm_wgroup {
  static constexpr bool wide = true;
  uint32_t tpi;    /* 64 or 128 */
  uint32_t lane;   /* 0 .. tpi-1 */
  uint32_t wl;     /* lane in the warp */
  uint32_t warp;   /* warp in the group */
  uint32_t nw;     /* warps in the group */
  uint32_t bar;    /* named barrier */
  uint32_t *sa;    /* operand area, tpi * L words */
  uint32_t *ss;    /* slots */
  mutable uint32_t phase;

  GM_DEV void init (uint32_t t, uint32_t *smem, uint32_t L)
  {
    const uint32_t gi = threadIdx.x / t;
    tpi = t;
    lane = threadIdx.x & (t - 1);
    wl = threadIdx.x & 31;
    warp = lane >> 5;
    nw = t >> 5;
    bar = 1 + gi;
    sa = smem + gi * t * L;
    ss = smem + blockDim.x * L + gi * 2 * GM_WSLOTS;
    phase = 0;
  }

  GM_DEV void sync () const
  {
    asm volatile ("bar.sync %0, %1;" :: "r"(bar), "r"(tpi) : "memory");
  }

  GM_DEV uint32_t *slots () const { return ss + phase * GM_WSLOTS; }

  /* barrier, then flip the phase */
  GM_DEV void exchange () const
  {
    sync ();
    phase ^= 1;
  }

  GM_DEV uint32_t resolve (uint32_t gen, uint32_t prop, uint32_t *top) const
  {
    uint32_t G = __ballot_sync (GM_FULL, gen);
    uint32_t P = __ballot_sync (GM_FULL, prop);
    uint32_t *s = slots ();
    if (wl == 0)
      {
        s[GM_WS_RES + 2 * warp] = G;
        s[GM_WS_RES + 2 * warp + 1] = P;
      }
    exchange ();
    /* C = ((G << 1) + P) ^ P over the nw-word numbers */
    uint32_t cy = 0, prev = 0, cin = 0;
    for (uint32_t w = 0; w < nw; w++)
      {
        const uint32_t gw = s[GM_WS_RES + 2 * w], pw = s[GM_WS_RES + 2 * w + 1];
        const uint64_t x = (uint64_t) ((gw << 1) | (prev >> 31)) + pw + cy;
        if (w == warp)
          cin = (((uint32_t) x ^ pw) >> wl) & 1;
        cy = (uint32_t) (x >> 32);
        prev = gw;
      }
    if (top)
      *top = ((prev >> 31) + cy) & 1;
    return cin;
  }

  GM_DEV uint32_t from_below (uint32_t v) const
  {
    uint32_t r = __shfl_up_sync (GM_FULL, v, 1);
    uint32_t *s = slots ();
    if (wl == 31)
      s[GM_WS_ONE + warp] = v;
    exchange ();
    if (wl == 0)
      r = warp > 0 ? s[GM_WS_ONE + warp - 1] : 0;
    return r;
  }

  /* (0 in the top thread) */
  GM_DEV uint32_t from_above (uint32_t v) const
  {
    uint32_t r = __shfl_down_sync (GM_FULL, v, 1);
    uint32_t *s = slots ();
    if (wl == 0)
      s[GM_WS_ONE + warp] = v;
    exchange ();
    if (wl == 31)
      r = warp + 1 < nw ? s[GM_WS_ONE + warp + 1] : 0;
    return r;
  }

  GM_DEV uint32_t bcast (uint32_t v, uint32_t src) const
  {
    uint32_t *s = slots ();
    if (lane == src)
      s[GM_WS_ONE] = v;
    exchange ();
    return s[GM_WS_ONE];
  }
};

/* Which form of the wide product is faster for L (see gm_mul_dist): on an
   A10 the lead form with 4 steps per barrier (which needs L >= 16) by
   4-19%; below, the block form with 2 */
GM_DEV constexpr bool gm_wide_lead_better (int L)
{
  return L >= 16;
}

/* Shared memory of the wide groups of a block of B threads. */
#define GM_WSHARED_WORDS(B, L) ((B) * (L) + 2 * ((B) / 64) * GM_WSLOTS)

template<class G>
GM_DEV uint32_t gm_resolve (const G &g, uint32_t gen, uint32_t prop, uint32_t *top)
{
  return g.resolve (gen, prop, top);
}

/* r (a slice with carry-out bit GEN into the next lane) := fully
 * propagated sum.  The carry out of the top lane is dropped. */
template<int L, bool SOLO, class G>
GM_DEV void gm_prop_carry (const G &g, uint32_t (&r)[L], uint32_t gen)
{
  if (SOLO)
    return;
  uint32_t all = r[0];
#pragma unroll
  for (int k = 1; k < L; k++)
    all &= r[k];
  uint32_t cin = gm_resolve (g, gen, all == GM_FULL, NULL);
  if (L == 1)
    r[0] += cin;
  else
    {
      r[0] = add_cc (r[0], cin);
#pragma unroll
      for (int k = 1; k < L - 1; k++)
        r[k] = addc_cc (r[k], 0);
      r[L-1] = addc (r[L-1], 0);
    }
}

/* r = a + b.  Caller guarantees no overflow out of n limbs. */
template<int L, bool SOLO, class G>
GM_DEV void gm_add (const G &g, uint32_t (&r)[L],
                    const uint32_t (&a)[L], const uint32_t (&b)[L])
{
  r[0] = add_cc (a[0], b[0]);
#pragma unroll
  for (int k = 1; k < L; k++)
    r[k] = addc_cc (a[k], b[k]);
  uint32_t gen = addc (0, 0);
  gm_prop_carry<L, SOLO> (g, r, gen);
}

/* r = a - b, plus m2 (= 2N) if a - b is negative.
 * Caller guarantees -m2 <= a - b < 2^(32n). */
template<int L, bool SOLO, class G>
GM_DEV void gm_sub (const G &g, uint32_t (&r)[L],
                    const uint32_t (&a)[L], const uint32_t (&b)[L],
                    const uint32_t (&m2)[L])
{
  r[0] = sub_cc (a[0], b[0]);
#pragma unroll
  for (int k = 1; k < L; k++)
    r[k] = subc_cc (a[k], b[k]);
  uint32_t neg = subc (0, 0) & 1;

  if (!SOLO)
    {
      uint32_t any = r[0];
#pragma unroll
      for (int k = 1; k < L; k++)
        any |= r[k];
      uint32_t bin = gm_resolve (g, neg, any == 0, &neg);
      if (L == 1)
        r[0] -= bin;
      else
        {
          r[0] = sub_cc (r[0], bin);
#pragma unroll
          for (int k = 1; k < L - 1; k++)
            r[k] = subc_cc (r[k], 0);
          r[L-1] = subc (r[L-1], 0);
        }
    }

  uint32_t msk = 0 - neg;
  r[0] = add_cc (r[0], m2[0] & msk);
#pragma unroll
  for (int k = 1; k < L; k++)
    r[k] = addc_cc (r[k], m2[k] & msk);
  uint32_t gen = addc (0, 0);
  gm_prop_carry<L, SOLO> (g, r, gen);
}

/* r = 2a (no overflow out of n limbs). */
template<int L, bool SOLO, class G>
GM_DEV void gm_dbl (const G &g, uint32_t (&r)[L], const uint32_t (&a)[L])
{
  uint32_t in = 0;
  if (!SOLO)
    in = g.from_below (a[L-1] >> 31);
#pragma unroll
  for (int k = L - 1; k > 0; k--)
    r[k] = __funnelshift_l (a[k-1], a[k], 1);
  r[0] = (a[0] << 1) | in;
}

/* After the accumulation loop of a Montgomery product each lane holds a
 * slice ACC plus a small word C (<= 2) belonging just above the slice.
 * Fold the C's upwards and propagate. */
template<int L, bool SOLO, class G>
GM_DEV void gm_fold (const G &g, uint32_t (&acc)[L], uint32_t c)
{
  if (SOLO)
    return;  /* C carries weight R, so it is 0 by the bounds. */
  uint32_t cin = g.from_below (c);
  uint32_t gen;
  if (L == 1)
    {
      acc[0] = add_cc (acc[0], cin);
      gen = addc (0, 0);
    }
  else
    {
      acc[0] = add_cc (acc[0], cin);
#pragma unroll
      for (int k = 1; k < L; k++)
        acc[k] = addc_cc (acc[k], 0);
      gen = addc (0, 0);
    }
  gm_prop_carry<L, SOLO> (g, acc, gen);
}

/* Accumulator of a Montgomery product: the lane's slice positions
 * 0..L-1 live in acc[], stored rotated by J limbs (position p in
 * acc[(p + J) % L]), and positions L and L+1 in h0 and h1. */
template<int L, int J>
GM_DEV uint32_t &gm_slot (uint32_t (&acc)[L], uint32_t &h0, uint32_t &h1, int p)
{
  return p < L ? acc[(p + J) % L] : (p == L ? h0 : h1);
}

/* (acc, h0, h1) += m * x.  The products are added as two carry chains,
 * even limbs of x then odd limbs, so that within a chain the low and high
 * words of each product are adjacent; ptxas fuses each such mad.lo/madc.hi
 * pair into one IMAD.WIDE.U32.X.  (A single interleaved chain needs a
 * separate IMAD, IMAD.HI and two IADD3s per product.)  The sum must fit
 * below position L + 2. */
template<int L, int J>
GM_DEV void gm_muladd (uint32_t (&acc)[L], uint32_t &h0, uint32_t &h1,
                       uint32_t m, const uint32_t (&x)[L])
{
#pragma unroll
  for (int k = 0; k < L; k += 2)
    {
      uint32_t &lo = gm_slot<L, J> (acc, h0, h1, k);
      uint32_t &hi = gm_slot<L, J> (acc, h0, h1, k + 1);
      lo = k == 0 ? madlo_cc (m, x[k], lo) : madclo_cc (m, x[k], lo);
      hi = madchi_cc (m, x[k], hi);
    }
  if (L % 2 == 0)
    {
      h0 = addc_cc (h0, 0);
      h1 = addc (h1, 0);
    }
  else
    h1 = addc (h1, 0);

  if (L > 1)
    {
#pragma unroll
      for (int k = 1; k < L; k += 2)
        {
          uint32_t &lo = gm_slot<L, J> (acc, h0, h1, k);
          uint32_t &hi = gm_slot<L, J> (acc, h0, h1, k + 1);
          lo = k == 1 ? madlo_cc (m, x[k], lo) : madclo_cc (m, x[k], lo);
          hi = madchi_cc (m, x[k], hi);
        }
      if (L % 2 == 0)
        h1 = addc (h1, 0);
      else
        {
          h0 = addc_cc (h0, 0);
          h1 = addc (h1, 0);
        }
    }
}

/* One Montgomery row: ACC += ai * B + q * M, then shift the accumulator
 * (as a distributed number) down one limb, which turns rotation J into
 * J + 1.  Between rows h1 is 0 and h0 is a small carry (see gm_fold). */
template<int L, bool SOLO, int J, class G>
GM_DEV void gm_row (const G &g, uint32_t (&acc)[L], uint32_t &h0,
                    uint32_t ai, const uint32_t (&b)[L],
                    const uint32_t (&m)[L], uint32_t mp0)
{
  uint32_t h1 = 0;
  gm_muladd<L, J> (acc, h0, h1, ai, b);

  uint32_t q = acc[J % L] * mp0;
  if (!SOLO)
    q = g.bcast (q, 0);
  gm_muladd<L, J> (acc, h0, h1, q, m);

  /* The old position 0 (zero in lane 0) moves to the lane below; the
   * top lane receives lane 0's zero.  The new position L-1 is h0 plus
   * the limb from the lane above. */
  uint32_t in = 0;
  if (!SOLO)
    in = g.from_above (acc[J % L]);
  acc[J % L] = add_cc (h0, in);
  h0 = addc (h1, 0);
}

/* L rows, with ai taken from limb j of lane S's copy of A. */
template<int L, bool SOLO, int J>
struct gm_rows {
  template<class G>
  GM_DEV static void run (const G &g, uint32_t (&acc)[L], uint32_t &c,
                          uint32_t s, const uint32_t (&a)[L],
                          const uint32_t (&b)[L], const uint32_t (&m)[L],
                          uint32_t mp0)
  {
    uint32_t ai = SOLO ? a[J] : g.bcast (a[J], s);
    gm_row<L, SOLO, J> (g, acc, c, ai, b, m, mp0);
    gm_rows<L, SOLO, J + 1>::run (g, acc, c, s, a, b, m, mp0);
  }
};
template<int L, bool SOLO>
struct gm_rows<L, SOLO, L> {
  template<class G>
  GM_DEV static void run (const G &, uint32_t (&)[L], uint32_t &,
                          uint32_t, const uint32_t (&)[L],
                          const uint32_t (&)[L], const uint32_t (&)[L],
                          uint32_t) {}
};

/* ---- distributed product, two rows per step ------------------------ */

/* Fused product pairs: (lo, hi) += a * b (+ carry in), see gm_muladd. */
#define GM_P0(lo, hi, a, b) do { lo = madlo_cc (a, b, lo); hi = madchi_cc (a, b, hi); } while (0)
#define GM_P(lo, hi, a, b) do { lo = madclo_cc (a, b, lo); hi = madchi_cc (a, b, hi); } while (0)
/* same, reading (slo, shi) and writing (dlo, dhi) */
#define GM_PS(dlo, dhi, a, b, slo, shi) do { dlo = madclo_cc (a, b, slo); dhi = madchi_cc (a, b, shi); } while (0)

/* Ripple the carry flag through x[from..to] (inclusive); drop it after. */
template<int N>
GM_DEV void gm_ripple (uint32_t (&x)[N], int from, int to)
{
#pragma unroll
  for (int p = 0; p < N; p++)
    if (p >= from && p <= to)
      x[p] = p < to ? addc_cc (x[p], 0) : addc (x[p], 0);
}

/* The accumulator is held as two numbers so that every fused product pair
 * lands on an aligned register pair (IMAD.WIDE needs one):
 *   A[k]  at position k     (pairs start at even positions), k = 0..L+2
 *   U[k]  at position k+1   (pairs start at odd positions),  k = 0..L+1
 * plus C, a word at position 0.  A step adds two rows, t0 X + q0 M at
 * position 0 and (t1 X + q1 M) W at position 1, then shifts down two limbs;
 * the shift is folded into the last two chains (they write their sums two
 * limbs lower).  Between steps only A[0..L], U[0..L-1] and C are live,
 * and positions >= L hold a small carry.
 *
 * WIDE (gm_wgroup): with LEAD, warp 0 computes q0, q1 as usual and
 * returns them in q0w, q1w, and the other warps take them from there;
 * without, every warp takes them from q0w, q1w.  The two words for the
 * lane below go out in w0o, w1o; within a warp they are also delivered
 * here, but not across warps. */
template<int L, bool WIDE, class G, bool LEAD = false>
GM_DEV void gm_step2 (const G &g, uint32_t (&A)[L + 3],
                      uint32_t (&U)[L + 2], uint32_t &C,
                      uint32_t t0, uint32_t t1, const uint32_t (&x)[L],
                      const uint32_t (&m)[L], uint32_t mp0,
                      uint32_t &q0w, uint32_t &q1w, uint32_t &w0o, uint32_t &w1o)
{
  /* row 0: C into position 0, its carry into U[0]; t0 * x_odd -> U */
  A[0] = add_cc (A[0], C);
#pragma unroll
  for (int k = 1; k < L; k += 2)
    GM_P (U[k-1], U[k], t0, x[k]);
  gm_ripple<L + 2> (U, L - 1 - (L & 1) + 1, L + 1);   /* after last pair */

  /* t0 * x_even -> A */
  GM_P0 (A[0], A[1], t0, x[0]);
#pragma unroll
  for (int k = 2; k < L; k += 2)
    GM_P (A[k], A[k+1], t0, x[k]);
  gm_ripple<L + 3> (A, (L - 1) - ((L - 1) & 1) + 2, L + 2);

  uint32_t q0;
  if constexpr (WIDE && LEAD)
    {
      /* a select, not a branch: no carry flag is live here, but keep the
         shape of the q1 case below */
      const uint32_t s0 = __shfl_sync (GM_FULL, A[0] * mp0, 0);
      q0 = g.warp == 0 ? s0 : q0w;
      q0w = q0;
    }
  else if constexpr (WIDE)
    q0 = q0w;
  else
    q0 = __shfl_sync (GM_FULL, A[0] * mp0, 0, g.tpi);
  GM_P0 (A[0], A[1], q0, m[0]);
#pragma unroll
  for (int k = 2; k < L; k += 2)
    GM_P (A[k], A[k+1], q0, m[k]);
  gm_ripple<L + 3> (A, (L - 1) - ((L - 1) & 1) + 2, L + 2);
  GM_P0 (U[0], U[1], q0, m[1]);
#pragma unroll
  for (int k = 3; k < L; k += 2)
    GM_P (U[k-1], U[k], q0, m[k]);
  gm_ripple<L + 2> (U, L - 1 - (L & 1) + 1, L + 1);

  /* row 1 (position 1): t1 * x_even -> U, t1 * x_odd -> A */
  GM_P0 (U[0], U[1], t1, x[0]);
#pragma unroll
  for (int k = 2; k < L; k += 2)
    GM_P (U[k], U[k+1], t1, x[k]);
  gm_ripple<L + 2> (U, (L - 1) - ((L - 1) & 1) + 2, L + 1);
  GM_P0 (A[2], A[3], t1, x[1]);
#pragma unroll
  for (int k = 3; k < L; k += 2)
    GM_P (A[k+1], A[k+2], t1, x[k]);
  gm_ripple<L + 3> (A, L - 1 - (L & 1) + 3, L + 2);

  /* position 1 = A[1] + U[0]; its carry goes to position 2, which is where
     the next chain (q1 * m_odd -> A) starts */
  const uint32_t w0 = A[0];
  U[0] = add_cc (U[0], A[1]);
  uint32_t q1;
  if constexpr (WIDE && LEAD)
    {
      /* the carry of U[0] += A[1] is live until the chain below takes it:
         select, don't branch */
      const uint32_t s1 = __shfl_sync (GM_FULL, U[0] * mp0, 0);
      q1 = g.warp == 0 ? s1 : q1w;
      q1w = q1;
    }
  else if constexpr (WIDE)
    q1 = q1w;
  else
    q1 = __shfl_sync (GM_FULL, U[0] * mp0, 0, g.tpi);

  /* q1 * m_odd -> A, shifted down two limbs */
#pragma unroll
  for (int k = 1; k < L; k += 2)
    GM_PS (A[k-1], A[k], q1, m[k], A[k+1], A[k+2]);
  {
    /* move (and carry into) the A words above the last pair */
    const int first = L - 1 - (L & 1) + 3;  /* first position not written */
#pragma unroll
    for (int p = 2; p <= L + 2; p++)
      if (p >= first)
        A[p-2] = p < L + 2 ? addc_cc (A[p], 0) : addc (A[p], 0);
    A[L+1] = 0;
    A[L+2] = 0;
  }

  /* q1 * m_even -> U, shifted down two limbs; position 1 goes to the lane
     below, position 2 becomes the new C */
  uint32_t w1;
  w1 = madlo_cc (q1, m[0], U[0]);
  C = madchi_cc (q1, m[0], U[1]);
#pragma unroll
  for (int k = 2; k < L; k += 2)
    GM_PS (U[k-2], U[k-1], q1, m[k], U[k], U[k+1]);
  {
    const int first = (L - 1) - ((L - 1) & 1) + 2;  /* first U index not read */
#pragma unroll
    for (int p = 2; p <= L + 1; p++)
      if (p >= first)
        U[p-2] = p < L + 1 ? addc_cc (U[p], 0) : addc (U[p], 0);
    U[L] = 0;
    U[L+1] = 0;
  }

  /* positions 0 and 1 of the lane above become our L-2 and L-1 (the top
     lane gets lane 0's, which are zero; in a wide group the top lane of
     each warp gets its words from the next warp later, see gm_mul_dist) */
  uint32_t in0, in1;
  if constexpr (WIDE)
    {
      w0o = w0;
      w1o = w1;
      in0 = __shfl_down_sync (GM_FULL, w0, 1);
      in1 = __shfl_down_sync (GM_FULL, w1, 1);
      if (g.wl == 31)
        in0 = in1 = 0;
    }
  else
    {
      const uint32_t up = (g.lane + 1) & (g.tpi - 1);
      in0 = __shfl_sync (GM_FULL, w0, up, g.tpi);
      in1 = __shfl_sync (GM_FULL, w1, up, g.tpi);
    }
  A[L-2] = add_cc (A[L-2], in0);
  A[L-1] = addc_cc (A[L-1], in1);
  A[L] = addc (A[L], 0);
}

/* Distributed Montgomery product (TPI >= 2, L >= 2). */
template<int L>
GM_DEV void gm_mul_dist (const gm_group &g, uint32_t (&r)[L],
                         const uint32_t (&a)[L], const uint32_t (&b)[L],
                         const uint32_t (&m)[L], uint32_t mp0)
{
  uint32_t A[L + 3], U[L + 2], C = 0;
#pragma unroll
  for (int k = 0; k < L + 3; k++)
    A[k] = 0;
#pragma unroll
  for (int k = 0; k < L + 2; k++)
    U[k] = 0;

  if (L % 2 == 0)
    {
#pragma unroll 1
      for (uint32_t s = 0; s < g.tpi; s++)
        {
#pragma unroll
          for (int j = 0; j < L; j += 2)
            {
              uint32_t t0 = __shfl_sync (GM_FULL, a[j], s, g.tpi);
              uint32_t t1 = __shfl_sync (GM_FULL, a[j + 1 < L ? j + 1 : 0], s, g.tpi);
              gm_step2<L, false> (g, A, U, C, t0, t1, b, m, mp0, t0, t1, t0, t1);
            }
        }
    }
  else
    {
      /* odd L: take the limbs of two source lanes per iteration */
#pragma unroll 1
      for (uint32_t s = 0; s < g.tpi; s += 2)
        {
#pragma unroll
          for (int e = 0; e < 2 * L; e += 2)
            {
              uint32_t t0 = __shfl_sync (GM_FULL, a[e % L], s + e / L, g.tpi);
              uint32_t t1 = __shfl_sync (GM_FULL, a[(e + 1) % L], s + (e + 1) / L, g.tpi);
              gm_step2<L, false> (g, A, U, C, t0, t1, b, m, mp0, t0, t1, t0, t1);
            }
        }
    }

  /* r = A + U W + C, with the lane's overflow (position L) in h */
  r[0] = add_cc (A[0], C);
#pragma unroll
  for (int k = 1; k < L; k++)
    r[k] = addc_cc (A[k], U[k-1]);
  uint32_t h = addc (A[L], U[L-1]);
  gm_fold<L, false> (g, r, h);
}

/* Distributed Montgomery product over a wide group, "lead" form: the steps
 * of gm_mul_dist, A's limbs read from shared memory, in blocks of KK steps
 * per barrier.  Warp 0 (which holds lane 0, where the q come from) runs a
 * block ahead and computes the q as gm_step2 does; at each barrier it
 * publishes them for the other warps, which run the block it did last.
 * The bottom lane of each warp publishes the words it shifted down to the
 * warp below during its block; the top lane of that warp adds them a
 * barrier later where they would have got to by then: R = 2 KK positions
 * lower per block it has done since, so at L - 2R for warp 0 (L - R at the
 * very end) and at L - R for the others.  They have not left the lane yet
 * as long as 4 KK <= L, which sets KK. */
template<int L, int K>
GM_DEV void gm_mul_wide_lead (const gm_wgroup &g, uint32_t (&r)[L],
                              const uint32_t (&a)[L], const uint32_t (&b)[L],
                              const uint32_t (&m)[L], uint32_t mp0)
{
  static_assert (K == 1 || K == 2 || K == GM_WIDE_MAXK,
                 "steps per barrier: 1, 2 or 4 (slot layout, tpi L / 2K)");
  constexpr int KK = 4 * K <= L ? K : (L >= 8 ? 2 : 1);
  constexpr int R = 2 * KK;
  static_assert (L >= 4, "wide product needs L >= 4");
#pragma unroll
  for (int k = 0; k < L; k++)
    g.sa[g.lane * L + k] = a[k];
  g.sync ();

  uint32_t A[L + 3], U[L + 2], C = 0;
#pragma unroll
  for (int k = 0; k < L + 3; k++)
    A[k] = 0;
#pragma unroll
  for (int k = 0; k < L + 2; k++)
    U[k] = 0;

  const bool lead = g.warp == 0;
  const bool late = g.wl == 31 && g.warp + 1 < g.nw;
  const uint32_t nb = g.tpi * L / R;   /* tpi * L is a multiple of 64 */
  const uint32_t *sp = g.slots ();
#pragma unroll 1
  for (uint32_t it = 0; ; it++)
    {
      /* the words from the warp above, published at the last barrier */
      if (it >= 2)
        {
          uint32_t in[R];
#pragma unroll
          for (int i = 0; i < R; i++)
            in[i] = late ? sp[GM_WS_IN + R * g.warp + i] : 0;
          if (lead && it <= nb)
            {
              A[L-2*R] = add_cc (A[L-2*R], in[0]);
#pragma unroll
              for (int i = 1; i < R; i++)
                A[L-2*R+i] = addc_cc (A[L-2*R+i], in[i]);
#pragma unroll
              for (int p = L - R; p < L; p++)
                A[p] = addc_cc (A[p], 0);
            }
          else
            {
              A[L-R] = add_cc (A[L-R], in[0]);
#pragma unroll
              for (int i = 1; i < R; i++)
                A[L-R+i] = addc_cc (A[L-R+i], in[i]);
            }
          A[L] = addc (A[L], 0);
        }
      if (it == nb + 1)
        break;

      /* warp 0: block it; the others: block it - 1, with warp 0's q */
      uint32_t *s = g.slots ();
      uint32_t q[R], wo[R];
#pragma unroll
      for (int i = 0; i < R; i++)
        {
          q[i] = 0;
          wo[i] = 0;
        }
      const int bk = lead ? (int) it : (int) it - 1;
      if (bk >= 0 && bk < (int) nb)
        {
          if (!lead)
#pragma unroll
            for (int i = 0; i < R; i++)
              q[i] = sp[GM_WS_Q + i];
#pragma unroll
          for (int j = 0; j < KK; j++)
            gm_step2<L, true, gm_wgroup, true> (g, A, U, C, g.sa[R * bk + 2*j],
                               g.sa[R * bk + 2*j + 1], b, m, mp0,
                               q[2*j], q[2*j+1], wo[2*j], wo[2*j+1]);
        }
      if (lead && g.wl == 0)
#pragma unroll
        for (int i = 0; i < R; i++)
          s[GM_WS_Q + i] = q[i];
      if (g.wl == 0 && g.warp > 0)
#pragma unroll
        for (int i = 0; i < R; i++)
          s[GM_WS_IN + R * (g.warp - 1) + i] = wo[i];
      sp = s;
      g.exchange ();
    }

  /* r = A + U W + C, with the lane's overflow (position L) in h */
  r[0] = add_cc (A[0], C);
#pragma unroll
  for (int k = 1; k < L; k++)
    r[k] = addc_cc (A[k], U[k-1]);
  uint32_t h = addc (A[L], U[L-1]);
  gm_fold<L, false> (g, r, h);
}

/* Lane 0 of a wide group: the q of the next R rows (R/2 steps, R <= L),
 * from the low R positions of the accumulator (A[0..] + U[0..] W + C) and
 * the rows t: the unique q < W^R with acc + t x + q m = 0 mod W^R, which
 * gm_step2 would find one word at a time.  Nothing else reaches those
 * positions in the meantime: the words from lane 1 come in at L and up. */
template<int L, int R>
GM_DEV void gm_wide_qblock (const uint32_t (&A)[L + 3],
                            const uint32_t (&U)[L + 2], uint32_t C,
                            const uint32_t (&t)[R], const uint32_t (&x)[L],
                            const uint32_t (&m)[L], uint32_t mp0,
                            uint32_t (&q)[R])
{
  uint32_t w[R];
  uint64_t c = (uint64_t) A[0] + C;
  w[0] = (uint32_t) c;
  c >>= 32;
#pragma unroll
  for (int i = 1; i < R; i++)
    {
      c += (uint64_t) A[i] + U[i-1];
      w[i] = (uint32_t) c;
      c >>= 32;
    }
#pragma unroll
  for (int r = 0; r < R; r++)
    {
      c = 0;
#pragma unroll
      for (int i = 0; r + i < R; i++)
        {
          c += (uint64_t) t[r] * x[i] + w[r + i];
          w[r + i] = (uint32_t) c;
          c >>= 32;
        }
      q[r] = w[r] * mp0;
      c = 0;
#pragma unroll
      for (int i = 0; r + i < R; i++)
        {
          c += (uint64_t) q[r] * m[i] + w[r + i];
          w[r + i] = (uint32_t) c;
          c >>= 32;
        }
    }
}

/* The "block" form: all warps in step, KK steps (2 KK <= L rows) per
 * barrier.  At each barrier lane 0 publishes the q of the next block
 * (gm_wide_qblock), and the bottom lane of each warp the words it shifted
 * down to the warp below during the last block; the top lane of that warp
 * adds them where they would have got to, at positions L - 2KK .. L - 1,
 * which they would not yet have left. */
template<int L, int K>
GM_DEV void gm_mul_wide_block (const gm_wgroup &g, uint32_t (&r)[L],
                               const uint32_t (&a)[L], const uint32_t (&b)[L],
                               const uint32_t (&m)[L], uint32_t mp0)
{
  static_assert (K == 1 || K == 2 || K == GM_WIDE_MAXK,
                 "steps per barrier: 1, 2 or 4 (slot layout, tpi L / 2K)");
  constexpr int KK = 2 * K <= L ? K : (L >= 8 ? 4 : (L >= 4 ? 2 : 1));
  constexpr int R = 2 * KK;
#pragma unroll
  for (int k = 0; k < L; k++)
    g.sa[g.lane * L + k] = a[k];
  g.sync ();

  uint32_t A[L + 3], U[L + 2], C = 0;
#pragma unroll
  for (int k = 0; k < L + 3; k++)
    A[k] = 0;
#pragma unroll
  for (int k = 0; k < L + 2; k++)
    U[k] = 0;

  uint32_t wo[R];
#pragma unroll
  for (int i = 0; i < R; i++)
    wo[i] = 0;
  const uint32_t blocks = g.tpi * L / R;   /* tpi * L is a multiple of 64 */
#pragma unroll 1
  for (uint32_t bk = 0; bk <= blocks; bk++)
    {
      uint32_t t[R];
      uint32_t *s = g.slots ();
      if (bk < blocks)
        {
#pragma unroll
          for (int i = 0; i < R; i++)
            t[i] = g.sa[R * bk + i];
          if (g.lane == 0)
            {
              uint32_t q[R];
              gm_wide_qblock<L, R> (A, U, C, t, b, m, mp0, q);
#pragma unroll
              for (int i = 0; i < R; i++)
                s[GM_WS_Q + i] = q[i];
            }
        }
      if (g.wl == 0 && g.warp > 0)
#pragma unroll
        for (int i = 0; i < R; i++)
          s[GM_WS_IN + R * (g.warp - 1) + i] = wo[i];
      g.exchange ();
      uint32_t in[R];
      const bool late = g.wl == 31 && g.warp + 1 < g.nw;
#pragma unroll
      for (int i = 0; i < R; i++)
        in[i] = late ? s[GM_WS_IN + R * g.warp + i] : 0;
      A[L-R] = add_cc (A[L-R], in[0]);
#pragma unroll
      for (int i = 1; i < R; i++)
        A[L-R+i] = addc_cc (A[L-R+i], in[i]);
      A[L] = addc (A[L], 0);
      if (bk == blocks)
        break;
#pragma unroll
      for (int j = 0; j < KK; j++)
        {
          uint32_t q0 = s[GM_WS_Q + 2*j], q1 = s[GM_WS_Q + 2*j + 1];
          gm_step2<L, true> (g, A, U, C, t[2*j], t[2*j+1], b, m, mp0,
                             q0, q1, wo[2*j], wo[2*j+1]);
        }
    }

  /* r = A + U W + C, with the lane's overflow (position L) in h */
  r[0] = add_cc (A[0], C);
#pragma unroll
  for (int k = 1; k < L; k++)
    r[k] = addc_cc (A[k], U[k-1]);
  uint32_t h = addc (A[L], U[L-1]);
  gm_fold<L, false> (g, r, h);
}

/* Wide product: the form and steps per barrier that measured fastest for
   this L (gpu_dev/wide_mul_bench.cu; GM_WIDE_FORM/GM_WIDE_STEPS override). */
template<int L>
GM_DEV void gm_mul_dist (const gm_wgroup &g, uint32_t (&r)[L],
                         const uint32_t (&a)[L], const uint32_t (&b)[L],
                         const uint32_t (&m)[L], uint32_t mp0)
{
#if defined (GM_WIDE_FORM) && defined (GM_WIDE_STEPS)
  if constexpr (GM_WIDE_FORM == 1)
    gm_mul_wide_lead<L, GM_WIDE_STEPS> (g, r, a, b, m, mp0);
  else
    gm_mul_wide_block<L, GM_WIDE_STEPS> (g, r, a, b, m, mp0);
#else
  if constexpr (gm_wide_lead_better (L))
    gm_mul_wide_lead<L, 4> (g, r, a, b, m, mp0);
  else
    gm_mul_wide_block<L, 2> (g, r, a, b, m, mp0);
#endif
}

/* r = a * b / R mod m (lazy, see top of file).  r may alias a or b. */
template<int L, bool SOLO, class G>
GM_DEV void gm_mul (const G &g, uint32_t (&r)[L],
                    const uint32_t (&a)[L], const uint32_t (&b)[L],
                    const uint32_t (&m)[L], uint32_t mp0)
{
  if constexpr (!SOLO && L >= (G::wide ? 4 : 2))
    gm_mul_dist<L> (g, r, a, b, m, mp0);
  else
    {
      uint32_t acc[L];
#pragma unroll
      for (int k = 0; k < L; k++)
        acc[k] = 0;
      uint32_t c = 0;
      if (SOLO)
        gm_rows<L, SOLO, 0>::run (g, acc, c, 0, a, b, m, mp0);
      else
        {
#pragma unroll 1
          for (uint32_t s = 0; s < g.tpi; s++)
            gm_rows<L, SOLO, 0>::run (g, acc, c, s, a, b, m, mp0);
        }
      gm_fold<L, SOLO> (g, acc, c);
#pragma unroll
      for (int k = 0; k < L; k++)
        r[k] = acc[k];
    }
}

/* t += a * x[j] W^(base + j) for j = j0, j0 + 2, ... < L as one
 * fused carry chain (see gm_muladd); the carry is rippled up through limb
 * TOP and dropped there (the caller guarantees it is zero). */
template<int L>
GM_DEV void gm_sqr_chain (uint32_t (&t)[2 * L], uint32_t a,
                          const uint32_t (&x)[L], int j0, int base, int top)
{
  int end = -1;
#pragma unroll
  for (int j = j0; j < L; j += 2)
    {
      t[base + j] = j == j0 ? madlo_cc (a, x[j], t[base + j])
                            : madclo_cc (a, x[j], t[base + j]);
      t[base + j + 1] = madchi_cc (a, x[j], t[base + j + 1]);
      end = base + j + 1;
    }
  if (end >= 0)
    {
#pragma unroll
      for (int p = end + 1; p <= top; p++)
        t[p] = p < top ? addc_cc (t[p], 0) : addc (t[p], 0);
    }
}

/* Solo-only squaring: off-diagonal products once, doubled, plus the
 * diagonal, then a separate Montgomery reduction. */
template<int L>
GM_DEV void gm_sqr_solo (uint32_t (&r)[L], const uint32_t (&a)[L],
                         const uint32_t (&m)[L], uint32_t mp0)
{
  uint32_t t[2 * L];
#pragma unroll
  for (int k = 0; k < 2 * L; k++)
    t[k] = 0;

  /* off-diagonal: row i adds a_i * a_j W^(i+j) for j > i.  The sum of
     rows 0..i is below W^(i+L+1), so nothing carries out of limb i+L. */
#pragma unroll
  for (int i = 0; i < L - 1; i++)
    {
      gm_sqr_chain<L> (t, a[i], a, i + 1, i, i + L);
      gm_sqr_chain<L> (t, a[i], a, i + 2, i, i + L);
    }

  /* double */
#pragma unroll
  for (int k = 2 * L - 1; k > 0; k--)
    t[k] = __funnelshift_l (t[k-1], t[k], 1);
  t[0] = t[0] << 1;

  /* diagonal */
#pragma unroll
  for (int i = 0; i < L; i++)
    {
      t[2*i] = i == 0 ? madlo_cc (a[i], a[i], t[2*i])
                      : madclo_cc (a[i], a[i], t[2*i]);
      t[2*i + 1] = i == L - 1 ? madchi (a[i], a[i], t[2*i + 1])
                              : madchi_cc (a[i], a[i], t[2*i + 1]);
    }

  /* reduce.  Row i adds q m W^i, as an even-limb chain and an odd-limb
     chain; each ends at limb i+L-1 or i+L, and their carries out of limb
     i+L are collected in E, which is added at limb i+L by the next row
     (nothing carries out of limb 2L-1 at all: the result is < 2m R). */
  uint32_t e = 0;
#pragma unroll
  for (int i = 0; i < L; i++)
    {
      uint32_t q = t[i] * mp0;
      uint32_t ee, eo = 0;
#pragma unroll
      for (int k = 0; k < L; k += 2)
        {
          t[i + k] = k == 0 ? madlo_cc (q, m[k], t[i + k])
                            : madclo_cc (q, m[k], t[i + k]);
          t[i + k + 1] = madchi_cc (q, m[k], t[i + k + 1]);
        }
      if (L % 2 == 0)
        {
          /* even chain ended at limb i+L-1 */
          if (i + L < 2 * L)
            t[i + L] = addc_cc (t[i + L], e);
          ee = addc (0, 0);
        }
      else
        ee = addc (0, 0);
      if (L > 1)
        {
#pragma unroll
          for (int k = 1; k < L; k += 2)
            {
              t[i + k] = k == 1 ? madlo_cc (q, m[k], t[i + k])
                                : madclo_cc (q, m[k], t[i + k]);
              t[i + k + 1] = madchi_cc (q, m[k], t[i + k + 1]);
            }
          if (L % 2 == 1)
            {
              /* odd chain ended at limb i+L-1 */
              t[i + L] = addc_cc (t[i + L], e);
            }
          eo = addc (0, 0);
        }
      e = ee + eo;
    }
#pragma unroll
  for (int k = 0; k < L; k++)
    r[k] = t[L + k];
}

template<int L, bool SOLO, class G>
GM_DEV void gm_sqr (const G &g, uint32_t (&r)[L],
                    const uint32_t (&a)[L], const uint32_t (&m)[L],
                    uint32_t mp0)
{
  if (SOLO && L > 1)
    gm_sqr_solo<L> (r, a, m, mp0);
  else
    gm_mul<L, SOLO> (g, r, a, a, m, mp0);
}

/* r = a * d / 2^32 mod m for a 32-bit d (lazy: r < a + m). */
template<int L, bool SOLO, class G>
GM_DEV void gm_mul_ui (const G &g, uint32_t (&r)[L],
                       const uint32_t (&a)[L], uint32_t d,
                       const uint32_t (&m)[L], uint32_t mp0)
{
  uint32_t acc[L];
#pragma unroll
  for (int k = 0; k < L; k++)
    acc[k] = 0;
  uint32_t c = 0;
  gm_row<L, SOLO, 0> (g, acc, c, d, a, m, mp0);
  /* acc is now rotated by 1 */
  uint32_t t[L];
#pragma unroll
  for (int k = 0; k < L; k++)
    t[k] = acc[(k + 1) % L];
  gm_fold<L, SOLO> (g, t, c);
#pragma unroll
  for (int k = 0; k < L; k++)
    r[k] = t[k];
}

#endif /* _GPU_MONT_CUH */
