/* gpu_op_bench.cu: time single ops (ours vs CGBN) at one shape.
   nvcc -O2 -arch=sm_89 -I CGBN/include -DTPI_=8 -DL_=8 gpu_op_bench.cu -lgmp */
#include <stdio.h>
#include <stdint.h>
#include <gmp.h>
#include <cgbn/cgbn.h>
#include "gpu_mont.cuh"

#ifndef TPI_
#define TPI_ 8
#endif
#ifndef L_
#define L_ 8
#endif
#define BITS (32 * TPI_ * L_)
#define REPS 2000

struct params_t {
  static const uint32_t TPB = 0, MAX_ROTATION = 4, SHM_LIMIT = 0;
  static const bool CONSTANT_TIME = false;
};
typedef cgbn_context_t<TPI_, params_t> ctx_t;
typedef cgbn_env_t<ctx_t, BITS> env_t;

/* op: 0 mul, 1 sqr, 2 add, 3 sub, 4 full ladder-ish mix */
__global__ void ours (int op, uint32_t *io)
{
  gm_group g;
  g.init (TPI_);
  uint32_t a[L_], b[L_], N[L_], N2[L_];
  for (int k = 0; k < L_; k++)
    {
      a[k] = io[k] + threadIdx.x;
      b[k] = io[k + 1] ^ blockIdx.x;
      N[k] = io[k + 2] | 1;
    }
  N[L_ - 1] &= 0x0fffffff;
  gm_dbl<L_, false> (g, N2, N);
  for (int i = 0; i < REPS; i++)
    switch (op)
      {
      case 0: gm_mul<L_, false> (g, a, a, b, N, 0x12345677); break;
      case 1: gm_sqr<L_, false> (g, a, a, N, 0x12345677); break;
      case 2: gm_add<L_, false> (g, a, a, b); break;
      default: gm_sub<L_, false> (g, a, a, b, N2); break;
      }
  for (int k = 0; k < L_; k++)
    io[threadIdx.x * L_ + k + 64] = a[k];
}

__global__ void theirs (int op, uint32_t *io)
{
  ctx_t ctx (cgbn_no_checks);
  env_t env (ctx);
  typename env_t::cgbn_t a, b, N, t;
  /* operands from memory, so the compiler can't see zero limbs */
  cgbn_load (env, a, (cgbn_mem_t<BITS> *) (io + 1));
  cgbn_load (env, b, (cgbn_mem_t<BITS> *) (io + 2));
  cgbn_load (env, N, (cgbn_mem_t<BITS> *) (io + 3));
  cgbn_shift_right (env, N, N, 4);
  cgbn_add_ui32 (env, a, a, threadIdx.x);
  for (int i = 0; i < REPS; i++)
    switch (op)
      {
      case 0: cgbn_mont_mul (env, a, a, b, N, 0x12345677); break;
      case 1: cgbn_mont_sqr (env, a, a, N, 0x12345677); break;
      case 2: cgbn_add (env, a, a, b);
              if (cgbn_compare (env, a, N) >= 0) cgbn_sub (env, a, a, N); break;
      default: if (cgbn_sub (env, a, a, b)) cgbn_add (env, a, a, N); break;
      }
  cgbn_store (env, (cgbn_mem_t<BITS> *) (io + 64) + threadIdx.x / TPI_, a);
}

int main ()
{
  uint32_t *io;
  cudaMalloc (&io, 64 << 20);
  cudaMemset (io, 0x5a, 64 << 20);
  const char *names[] = { "mul", "sqr", "add", "sub" };
  cudaEvent_t s, e;
  cudaEventCreate (&s); cudaEventCreate (&e);
  int sms; cudaDeviceGetAttribute (&sms, cudaDevAttrMultiProcessorCount, 0);
  for (int op = 0; op < 4; op++)
    {
      float t[2];
      for (int w = 0; w < 2; w++)
        {
          int nb = 0;
          if (w == 0) cudaOccupancyMaxActiveBlocksPerMultiprocessor (&nb, ours, 128, 0);
          else cudaOccupancyMaxActiveBlocksPerMultiprocessor (&nb, theirs, 128, 0);
          int blocks = nb * sms * 2;
          for (int rep = 0; rep < 2; rep++)
            {
              cudaEventRecord (s);
              if (w == 0) ours<<<blocks, 128>>> (op, io);
              else theirs<<<blocks, 128>>> (op, io);
              cudaEventRecord (e);
              cudaEventSynchronize (e);
              cudaEventElapsedTime (&t[w], s, e);
            }
          double inst = (double) blocks * 128 / TPI_;
          t[w] = t[w] * 1e6 / (inst * REPS);  /* ns per op per instance */
          if (op == 0) printf ("%s: %d blocks/SM\n", w ? "cgbn" : "ours", nb);
        }
      printf ("TPI=%d L=%d %-4s ours %8.3f ns  cgbn %8.3f ns  ratio %.2f\n",
              TPI_, L_, names[op], t[0], t[1], t[1] / t[0]);
    }
  cudaError_t err = cudaGetLastError ();
  if (err) printf ("error %s\n", cudaGetErrorString (err));
  return 0;
}
