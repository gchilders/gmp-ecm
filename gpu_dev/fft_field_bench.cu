/* fft_field_bench.cu: throughput of one radix-2 FFT/NTT butterfly in the
   candidate arithmetic for a GPU FFT multiplication backend.

   nvcc -O3 -arch=sm_86 -o fft_field_bench fft_field_bench.cu

   Each thread keeps 16 points in registers and runs radix-2 passes over
   them with twiddles from registers, so the result is the ALU cost of a
   butterfly with no memory traffic: the compute ceiling of each choice.

     shoup32  Z/p, p = 119*2^23+1 (29.9 bits), Shoup multiplication, lazy
              values in [0, 2p)
     m31c     GF(M31^2), M31 = 2^31-1, complex (a+bi), 4 real products
     m61c     GF(M61^2), M61 = 2^61-1, complex
     gold     Z/p, p = 2^64 - 2^32 + 1 (Goldilocks)
     fp32c    complex float

   Reported: butterflies per second, and (with the capacity per point of
   each choice for a product of 2^17 points) input bits per second per
   FFT level, the number that decides cost per bit. */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#ifndef NPTS
#define NPTS 16
#endif
#define REPS 4096
#define LOGN (NPTS == 16 ? 4 : NPTS == 8 ? 3 : 2)

/* ---- Z/p with p < 2^31, Shoup multiplication ---- */
#define P32 998244353u    /* 119 * 2^23 + 1 (29.9 bits): 4p < 2^32 for lazy sums */

__device__ __forceinline__ uint32_t
shoup_mul (uint32_t a, uint32_t w, uint32_t wp)   /* a < 2^32, result < 2p */
{
  uint32_t q = __umulhi (a, wp);
  return a * w - q * P32;
}

__device__ __forceinline__ void
bf_shoup (uint32_t &x, uint32_t &y, uint32_t w, uint32_t wp)
{
  /* x, y in [0, 2p) */
  uint32_t t = shoup_mul (y, w, wp);            /* [0, 2p) */
  uint32_t u = x >= 2 * P32 ? x - 2 * P32 : x;  /* keep x < 2p */
  x = u + t;                                    /* < 4p < 2^32 */
  x = x >= 2 * P32 ? x - 2 * P32 : x;
  y = u - t + 2 * P32;
  y = y >= 2 * P32 ? y - 2 * P32 : y;
}

/* ---- M31 ---- */
#define M31 0x7fffffffu
__device__ __forceinline__ uint32_t
m31_red64 (uint64_t v)       /* v < 2^62 -> [0, M31] (lazy: may equal M31) */
{
  uint32_t r = (uint32_t) (v & M31) + (uint32_t) (v >> 31);
  return (r & M31) + (r >> 31);
}
__device__ __forceinline__ uint32_t m31_add (uint32_t a, uint32_t b)
{ uint32_t r = a + b; return (r & M31) + (r >> 31); }
__device__ __forceinline__ uint32_t m31_sub (uint32_t a, uint32_t b)
{ uint32_t r = a + (M31 - b); return (r & M31) + (r >> 31); }

__device__ __forceinline__ void
bf_m31c (uint32_t &xr, uint32_t &xi, uint32_t &yr, uint32_t &yi,
         uint32_t wr, uint32_t wi)
{
  /* t = y * w, 4 products; each product < 2^62, sums of two < 2^63 */
  uint64_t rr = (uint64_t) yr * wr + (uint64_t) (M31 - yi) * wi;
  uint64_t ii = (uint64_t) yr * wi + (uint64_t) yi * wr;
  uint32_t tr = m31_red64 (rr), ti = m31_red64 (ii);
  uint32_t ar = m31_add (xr, tr), ai = m31_add (xi, ti);
  yr = m31_sub (xr, tr); yi = m31_sub (xi, ti);
  xr = ar; xi = ai;
}

/* ---- M61 ---- */
#define M61 0x1fffffffffffffffull
__device__ __forceinline__ uint64_t
m61_mul (uint64_t a, uint64_t b)    /* a, b <= M61 */
{
  uint64_t lo = a * b, hi = __umul64hi (a, b);
  uint64_t r = (lo & M61) + ((lo >> 61) | (hi << 3));
  return (r & M61) + (r >> 61);
}
__device__ __forceinline__ uint64_t m61_add (uint64_t a, uint64_t b)
{ uint64_t r = a + b; return (r & M61) + (r >> 61); }
__device__ __forceinline__ uint64_t m61_sub (uint64_t a, uint64_t b)
{ uint64_t r = a + (M61 - b); return (r & M61) + (r >> 61); }

__device__ __forceinline__ void
bf_m61c (uint64_t &xr, uint64_t &xi, uint64_t &yr, uint64_t &yi,
         uint64_t wr, uint64_t wi)
{
  /* Karatsuba-style complex product: 3 real products */
  uint64_t k1 = m61_mul (wr, m61_add (yr, yi));
  uint64_t k2 = m61_mul (yr, m61_sub (wi, wr));
  uint64_t k3 = m61_mul (yi, m61_add (wr, wi));
  uint64_t tr = m61_sub (k1, k3), ti = m61_add (k1, k2);
  uint64_t ar = m61_add (xr, tr), ai = m61_add (xi, ti);
  yr = m61_sub (xr, tr); yi = m61_sub (xi, ti);
  xr = ar; xi = ai;
}

/* ---- Goldilocks p = 2^64 - 2^32 + 1 ---- */
#define GP 0xffffffff00000001ull
__device__ __forceinline__ uint64_t
g_red128 (uint64_t lo, uint64_t hi)
{
  /* 2^64 = 2^32 - 1, 2^96 = -1 (mod p) */
  uint64_t hh = hi >> 32, hl = hi & 0xffffffffull;
  uint64_t t = lo - hh;                     /* lo - hh * 2^96 */
  if (lo < hh) t += GP;                     /* borrow */
  uint64_t m = (hl << 32) - hl;             /* hl * (2^32 - 1) */
  uint64_t r = t + m;
  if (r < m) r += 0xffffffffull;            /* carry: + 2^64 = + (2^32 - 1) */
  return r;
}
__device__ __forceinline__ uint64_t g_mul (uint64_t a, uint64_t b)
{ return g_red128 (a * b, __umul64hi (a, b)); }
__device__ __forceinline__ uint64_t g_add (uint64_t a, uint64_t b)
{ uint64_t r = a + b; if (r < a || r >= GP) r -= GP; return r; }
__device__ __forceinline__ uint64_t g_sub (uint64_t a, uint64_t b)
{ uint64_t r = a - b; if (a < b) r += GP; return r; }

__device__ __forceinline__ void
bf_gold (uint64_t &x, uint64_t &y, uint64_t w)
{
  uint64_t t = g_mul (y, w);
  uint64_t a = g_add (x, t);
  y = g_sub (x, t);
  x = a;
}

/* ---- complex float ---- */
__device__ __forceinline__ void
bf_fp32c (float &xr, float &xi, float &yr, float &yi, float wr, float wi)
{
  float tr = fmaf (yr, wr, -yi * wi), ti = fmaf (yr, wi, yi * wr);
  float ar = xr + tr, ai = xi + ti;
  yr = xr - tr; yi = xi - ti;
  xr = ar; xi = ai;
}

/* ---- kernels: 16 points, 4 radix-2 passes, REPS times ---- */

__global__ void k_shoup (uint32_t *out, uint32_t seed)
{
  uint32_t v[NPTS], w[4], wp[4];
  for (int k = 0; k < NPTS; k++) v[k] = (seed * (k + 1) + threadIdx.x) % P32;
  for (int k = 0; k < 4; k++)
    {
      w[k] = (seed ^ (k * 0x9e3779b9u)) % P32;
      wp[k] = (uint32_t) (((uint64_t) w[k] << 32) / P32);
    }
  for (int r = 0; r < REPS; r++)
#pragma unroll
    for (int s = 0; s < LOGN; s++)
#pragma unroll
      for (int k = 0; k < NPTS; k++)
        if (!(k & (1 << s)))
          bf_shoup (v[k], v[k | (1 << s)], w[s], wp[s]);
  uint32_t acc = 0;
  for (int k = 0; k < NPTS; k++) acc ^= v[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

__global__ void k_m31c (uint32_t *out, uint32_t seed)
{
  uint32_t vr[NPTS], vi[NPTS], wr[4], wi[4];
  for (int k = 0; k < NPTS; k++)
    { vr[k] = (seed * (k + 1) + threadIdx.x) % M31; vi[k] = (seed * (k + 7)) % M31; }
  for (int k = 0; k < 4; k++)
    { wr[k] = (seed ^ (k * 0x9e3779b9u)) % M31; wi[k] = (seed + k * 12345u) % M31; }
  for (int r = 0; r < REPS; r++)
#pragma unroll
    for (int s = 0; s < LOGN; s++)
#pragma unroll
      for (int k = 0; k < NPTS; k++)
        if (!(k & (1 << s)))
          bf_m31c (vr[k], vi[k], vr[k | (1 << s)], vi[k | (1 << s)], wr[s], wi[s]);
  uint32_t acc = 0;
  for (int k = 0; k < NPTS; k++) acc ^= vr[k] ^ vi[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

__global__ void k_m61c (uint32_t *out, uint32_t seed)
{
  uint64_t vr[NPTS], vi[NPTS], wr[4], wi[4];
  for (int k = 0; k < NPTS; k++)
    { vr[k] = ((uint64_t) seed * (k + 1) * 0x123456789ull + threadIdx.x) % M61;
      vi[k] = ((uint64_t) seed * (k + 7) * 0x987654321ull) % M61; }
  for (int k = 0; k < 4; k++)
    { wr[k] = ((uint64_t) seed * 0x9e3779b97f4a7c15ull * (k + 1)) % M61;
      wi[k] = ((uint64_t) seed * 0x3c6ef372fe94f82aull * (k + 3)) % M61; }
  for (int r = 0; r < REPS; r++)
#pragma unroll
    for (int s = 0; s < LOGN; s++)
#pragma unroll
      for (int k = 0; k < NPTS; k++)
        if (!(k & (1 << s)))
          bf_m61c (vr[k], vi[k], vr[k | (1 << s)], vi[k | (1 << s)], wr[s], wi[s]);
  uint64_t acc = 0;
  for (int k = 0; k < NPTS; k++) acc ^= vr[k] ^ vi[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = (uint32_t) (acc ^ (acc >> 32));
}

__global__ void k_gold (uint32_t *out, uint32_t seed)
{
  uint64_t v[NPTS], w[4];
  for (int k = 0; k < NPTS; k++)
    v[k] = ((uint64_t) seed * (k + 1) * 0x123456789ull + threadIdx.x) % GP;
  for (int k = 0; k < 4; k++)
    w[k] = ((uint64_t) seed * 0x9e3779b97f4a7c15ull * (k + 1)) % GP;
  for (int r = 0; r < REPS; r++)
#pragma unroll
    for (int s = 0; s < LOGN; s++)
#pragma unroll
      for (int k = 0; k < NPTS; k++)
        if (!(k & (1 << s)))
          bf_gold (v[k], v[k | (1 << s)], w[s]);
  uint64_t acc = 0;
  for (int k = 0; k < NPTS; k++) acc ^= v[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = (uint32_t) (acc ^ (acc >> 32));
}

__global__ void k_fp32c (uint32_t *out, uint32_t seed)
{
  float vr[NPTS], vi[NPTS], wr[4], wi[4];
  for (int k = 0; k < NPTS; k++)
    { vr[k] = (float) ((seed * (k + 1) + threadIdx.x) & 1023); vi[k] = (float) (k & 7); }
  for (int k = 0; k < 4; k++)
    { float a = (seed + k) * 0.001f; wr[k] = cosf (a) * 0.7071f; wi[k] = sinf (a) * 0.7071f; }
  for (int r = 0; r < REPS; r++)
#pragma unroll
    for (int s = 0; s < LOGN; s++)
#pragma unroll
      for (int k = 0; k < NPTS; k++)
        if (!(k & (1 << s)))
          bf_fp32c (vr[k], vi[k], vr[k | (1 << s)], vi[k | (1 << s)], wr[s], wi[s]);
  float acc = 0;
  for (int k = 0; k < NPTS; k++) acc += vr[k] + vi[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = __float_as_uint (acc);
}

typedef void (*kfn) (uint32_t *, uint32_t);

int main ()
{
  int dev = 0, sms;
  cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, dev);
  cudaDeviceGetAttribute (&sms, cudaDevAttrMultiProcessorCount, dev);
  printf ("%s, %d SMs, %.0f MHz\n", prop.name, sms, prop.clockRate / 1e3);

  struct { const char *name; kfn fn; double cap; int fields; int complex_; } S[] = {
    /* cap: bits of exact capacity per point-coefficient; fields: how many
       such transforms one product needs (3 primes for CRT, 1 M31 + 1 M61) */
    { "shoup32 x3", k_shoup, 3 * 29.895, 3, 0 },
    { "m31c (+m61c)", k_m31c, 31 + 61, 1, 1 },
    { "m61c (+m31c)", k_m61c, 31 + 61, 1, 1 },
    { "goldilocks", k_gold, 64, 1, 0 },
    { "fp32c", k_fp32c, 0, 1, 1 },
  };
  uint32_t *out;
  cudaMalloc (&out, 64 << 20);
  cudaEvent_t a, b;
  cudaEventCreate (&a); cudaEventCreate (&b);
  double bf_rate[5];
  for (int i = 0; i < 5; i++)
    {
      int nb = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor (&nb, S[i].fn, 256, 0);
      int blocks = nb * sms * 4;
      float ms = 0;
      for (int rep = 0; rep < 3; rep++)
        {
          cudaEventRecord (a);
          S[i].fn<<<blocks, 256>>> (out, 12345 + rep);
          cudaEventRecord (b);
          cudaEventSynchronize (b);
          cudaEventElapsedTime (&ms, a, b);
        }
      double bfs = (double) blocks * 256 * REPS * LOGN * (NPTS / 2);
      bf_rate[i] = bfs / (ms * 1e-3);
      printf ("%-14s %7.1f G butterflies/s  (%d blocks/SM)\n", S[i].name,
              bf_rate[i] / 1e9, nb);
    }
  cudaError_t e = cudaGetLastError ();
  if (e) printf ("error %s\n", cudaGetErrorString (e));

  /* Input bits per second per FFT level, for a product transform of
     L = 2^17 points (log2 L = 17), balanced inputs of b bits per word:
       exact fields: 2b + log2(L) + 1 <= capacity
       complex fields hold two real words per point (right-angle packing)
       fp32: error bound, b ~ (24 - 0.5 log2 L - 3) / 2 per real word */
  const int lg = 17;
  double b_exact_m = floor ((3 * 29.895 - lg - 1) / 2);
  double b_m3161 = floor ((92 - lg - 1) / 2);
  double b_gold = floor ((64 - lg - 1) / 2);
  double b_fp32 = floor ((24 - 0.5 * lg - 3) / 2);
  /* bits through one level per second: rate * 2 points per butterfly *
     words per point * b, divided by the transforms needed per product */
  double t_shoup = bf_rate[0] * 2 * 1 * b_exact_m / 3;
  double t_m3161 = 1.0 / (1.0 / (bf_rate[1] * 2 * 2 * b_m3161)
                          + 1.0 / (bf_rate[2] * 2 * 2 * b_m3161));
  double t_gold = bf_rate[3] * 2 * 1 * b_gold;
  double t_fp32 = bf_rate[4] * 2 * 2 * b_fp32;
  printf ("\nbits per word at L=2^%d: 3 x 31-bit primes %g, M31+M61 %g, "
          "Goldilocks %g, FP32 ~%g\n", lg, b_exact_m, b_m3161, b_gold, b_fp32);
  printf ("input Gbit per second per FFT level:\n"
          "  3 x shoup32 (CRT)   %8.1f\n  M31c + M61c (CRT)   %8.1f\n"
          "  Goldilocks          %8.1f\n  FP32 complex        %8.1f\n",
          t_shoup / 1e9, t_m3161 / 1e9, t_gold / 1e9, t_fp32 / 1e9);
  return 0;
}
