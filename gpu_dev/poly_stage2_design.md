# GPU polynomial stage 2: design

A GPU port of GMP-ECM's FFT continuation (`stage2.c`), so that GPU stage 2 covers
GMP-ECM's default B2 (about 1000-2000 B1) instead of the baby-step giant-step
continuation's 100 B1, at a cost far below one modular product per prime. The
design follows the ideas of gpuecm2 (github.com/ebranger/gpuecm2: polynomials
over Z/N in RNS form with batched NTTs, curves batched when one does not fill the
GPU); the code is our own, ported from GMP-ECM's stage 2 in this tree and built on
our GPU arithmetic (no CGBN).

## What it computes (same as stage2())

With GMP-ECM's parameters for N, B1, B2 (`set_stage_2_params`: dF, k, d1, d2, i0,
S; Dickson or x^S polynomial):

1. roots f_i of F: Weierstrass x of D_S(j d2) P for the dF values j of
   `ecm_rootsF` (j coprime to d1, finite differences);
2. F = prod (x - f_i), keeping its product tree (lg dF levels);
3. 1/rev(F) to dF terms (Newton), for the reductions;
4. for each of the k blocks: roots g_i of G (`ecm_rootsG`, the progression state
   carried from block to block), G = prod (x - g_i), H <- G H mod F
   (H = G - F for the first block);
5. the values H(f_i) by transposed multipoint evaluation over F's tree;
6. their product, gcd with N.

GMP-ECM's "Product of G(f_i)" (`-v -v`) is the product in step 6 when no factor is
found: the GPU result must equal it bit for bit, which checks every step at once.
Each step also has a CPU reference in this tree (`ecm_rootsF`, `PolyFromRoots_Tree`,
`ntt_PolyInvert`, `ntt_PrerevertDivision`, `ntt_polyevalT`), so the GPU pieces can
be tested one by one against the very functions they replace.

## Arithmetic

**Residues mod N** (roots, coefficients before and after the polynomial work): n
32-bit limbs, the representation of the stage 1 plan where the root kernels need
products (one-warp, wide or NTT kernels; see "Roots").

**Polynomials over Z/N: RNS.** As `mpzspm_init` does: P primes p = c 2^K + 1 < 2^31
(32-bit Shoup and Montgomery products without overflow tests), with 2^K the longest
transform, and enough of them that prod p > 4 len (N sum p)^2, so that a convolution
of length len of coefficients < N is exact.

- **Limbs -> RNS:** for each coefficient and prime, sum_j a_j (2^(32j) mod p): a
  (coefficients x limbs) by (limbs x primes) matrix product, tiled through shared
  memory, products in 64 bits, reduced once per tile.
- **RNS -> limbs mod N (explicit CRT):** x = sum_i ((y_i M_i^-1) mod p_i) M_i - alpha M
  with M = prod p, M_i = M / p_i, and alpha = round(sum_i ((y_i M_i^-1) mod p_i) / p_i)
  from a floating point sum (exact enough because x is far below M/2): a matrix
  product with the table of M_i mod N, the correction -alpha (M mod N), then a
  reduction below N.
- **NTT:** batched over (polynomials x primes), power-of-two lengths 2^6 to 2^K;
  lengths that fit shared memory in one pass per block (as gpu_ntt.cuh), longer
  ones as two passes through global memory (four-step), twiddle tables per prime.
  Products longer than 2^K are split and summed in the transform domain.
- Pointwise products, sums and middle products (for the transposed evaluation)
  stay in RNS; conversions back to Z/N only where GMP-ECM's algorithms reduce mod N.

These conversions cost P x n multiply-adds per coefficient (P is about 2n), so they
grow as n^2 and dominate for large N, together with the memory: one coefficient in
RNS is P words (about 26 KB at 100K bits). Large N therefore gets smaller dF (fitted
to the GPU memory), and the place to improve later is a transform in the number
dimension as well (each coefficient transformed once instead of reduced modulo P
primes), as discussed for prime95's stage 2.

## Roots

GMP-ECM's roots are affine Weierstrass x of multiples of the stage 1 point along
arithmetic progressions (finite-difference tables of D_S(j d2) P): full point
additions, not the x-only ladder. On the GPU: projective point arithmetic in a
model with cheap additions (twisted Edwards form of the Montgomery curve, extended
coordinates, as gpuecm2 does), built on our modular products (any size), then one
batched inversion per block of dF roots (Montgomery's trick: 3 products per root
plus one inversion, done by exponentiation or on the host) and the map to GMP-ECM's
Weierstrass x, so the roots are the same numbers as GMP-ECM's.

As built (gpu_roots.cu, M4): the progressions are our own choice (F: j = 1 mod 6
prime to a product m of small primes of d1, stride 6 m k, with a slot table that
skips the other j; G: residues of m prime to d2, stride d2 k, so a block of dF roots
is a run of steps of every progression and the state carries over between
blocks); every (curve, progression) is an instance of a schoolbook group (2 to 32
threads, L in a short list of limb counts, chosen for N) and loops over a few steps
per launch, writing numerator and denominator of
x_W into the root's slot; the batched inversion is a product tree over all curves'
denominators with one inversion on the host. The tables start from fixed-base
windowed scalar multiplications (w <= 4, one table per curve shared by F and G).
The number of progressions balances filling the GPU against the setup cost (S K
additions per progression for K digits). Up to 24572 bits (32 threads of 24 limbs)
so far; larger N comes with M7.

## Integration

- cudawrapper.c: GPU stage 2 uses the polynomial continuation with GMP-ECM's own
  parameter choice (default B2 as on the CPU, its Dickson polynomial), for every
  curve of the batch; dF and k are raised/lowered to fit the free GPU memory, as
  GMP-ECM's -maxmem does. The baby-step giant-step stage 2 stays as the fallback
  (option, or where the polynomial one cannot run). As of M5 the polynomial one is
  opt-in ("-gpupoly", params->gpu = 3) until the tuning (M6) shows where it wins;
  where it cannot run, stage 2 goes to the CPU with the same parameters.
- New files: gpu_rns.cu / gpu_rns.cuh (primes, conversions, NTT, products),
  gpu_polyalg.cu (trees, inverse, division, transposed evaluation),
  gpu_roots.cu (roots), gpu_s2poly.cu (driver); host parameters come from GMP-ECM.
- Curves: one at a time first (as stage2()), then batches of curves with the same N
  and parameters side by side in every kernel for small dF.

## Milestones

| | Content | Check |
|---|---|---|
| M1 | this design | |
| M2 | RNS engine: primes, tables, conversions both ways, batched NTT (short and long), polynomial product with splitting | products of random polynomials mod N vs GMP, sizes 512 to 32K bits |
| M3 | product tree, 1/rev(F), PrerevertDivision, transposed evaluation, product of values | vs the CPU functions on the same inputs |
| M4 | roots of F and G (projective Edwards, batched inversion, Weierstrass x) | equal to ecm_rootsF / ecm_rootsG. **Done** (gpu_dev/roots_test.cu: every root equal, x^S and Dickson, S up to 30, several shapes, curves, blocks and progression counts) |
| M5 | end-to-end driver, integration in cudawrapper.c, memory fitting | "Product of G(f_i)" equal to CPU GMP-ECM per curve; same factors; test.gpuecm. **Done** (gpu_s2poly.cu, option -gpupoly; GMP-ECM's parameters with use_ntt, so dF is a power of 2, lowered to fit the GPU memory; equal products for all 32 curves at 300, 600, 1000, 1500 and 2200 bits, B1 up to 1e6; the same stage 2 factors; test.gpuecm covers both) |
| M6 | performance: curve batching, streams, tuning | against the BSGS stage 2 and gpuecm2 at 1024-4096 bits. **Done for now** (A10, 1024 bits): B1 = 1e6 parameters (dF 4096, k 6) 139 -> 14.7 ms per curve, B1 = 3e6 (dF 16384, k 2) -> 34 ms (gpuecm2 on the same GPU: 12.6, 29). Curve batching (all of gpu_roots, gpu_polyalg and the driver take B curves side by side), short transforms many to a block, radix-8 in-block transforms (up to 2^14 when that is the longest, else 2^13 and four-step) with the twiddle and scaling folded in, 340-440 GB/s up to 2^12; conversions as matrix products; the low levels of the product trees and of the transposed evaluation in Z/N (gpu_roots_tree, gpu_roots_mp); the product of the values on the GPU; tiled Z/N kernels; roots of the next block of G on their own stream during the polynomial work (~3%: with batching both already fill the GPU). Not done: the dedicated Edwards addition (1 of 11 products, but it fails when the two points are equal, which a progression of x^1 can start with) |
| M7 | large N: memory-bounded dF, conversion cost; number-dimension transform | see "Large N" below |

## Large N (M7)

Where the limits are (A10, -gpupoly, GMP-ECM's parameters):

- **Roots and Z/N kernels**: schoolbook groups of 2 to 32 threads stopped at 32 x 24
  limbs (24572 bits). Wide groups (64 or 128 threads per curve, gm_wgroup, as stage 1
  and the baby-step giant-step stage 2 use) take them to 128 x 24 limbs (98300 bits),
  the range of the wide stage 1 kernels. The roots prefer L <= 12 (no spills) and so
  run without spills up to 128 x 12 limbs (49148 bits); above that L = 16 or 24
  spill but work. Above 98300 bits (stage 1's NTT range) the roots would need the
  NTT products: not done.
- **NTT primes** p = c 2^K + 1 < 2^30: enough (with inputs below 2N, prod p > 2^(K+3)
  N^2) for transforms of 2^14 up to about 65K-bit N and 2^13 up to 98K bits, so
  dF up to 2^13 and 2^12 there; memory caps dF before that.
- **Memory**: per curve about (l + 19) dF NL words of numbers and 11 P 2 dF words
  of RNS work (gpu_ecm_stage2_poly_mem); at 65K bits and dF = 4096 about 1.8 GB,
  so a batch of a few curves. cudawrapper lowers dF (through GMP-ECM's -maxmem
  logic) when even one curve does not fit.
- **Cost**: conversions are P x NL ~ 2 NL^2 multiply-adds per coefficient and the
  roots' products NL^2, so both grow as N^2; the transforms only as N.

Steps: (1) wide groups for the roots and Z/N kernels, checked root for root
against GMP-ECM with the wide shapes forced at small N (gpu_dev/roots_test.cu) and
by equal products; (2) timings against GMP-ECM's CPU stage 2 from 16K to 65K bits;
(3) then the number-dimension transform, if the conversions dominate there.

**Done: (1) and (2).** -gpupoly now runs up to 98300 bits. Against CPU GMP-ECM with
-ntt (the same parameters): the same "Product of G(f_i)" for every curve at 23997
and 39967 bits (32 curves, B1 = 500), and the same stage 2 factors (four curves, two
factors, at 39967 bits). Per curve on the A10, GMP-ECM's parameters:

| N bits | B1 = 1.1e5 (dF 1024, k 4, x^2) | B1 = 1e6 (dF 4096, k 6, Dickson(6)) |
|---|---|---|
| 8192 | 0.049 s (CPU, one core: 6.9 s) | 0.41 s (CPU: 75 s) |
| 16384 | 0.21 s | 1.7 s |
| 32768 | 0.93 s (CPU: 66 s) | 8.1 s |
| 49152 | 2.3 s | 19.7 s |
| 65536 | 3.1 s (was 4.0) | 48.3 s (was 72.6) |

The roots take half or more of the time at these sizes (Dickson(6): six additions
per root, each eleven products of N^2 cost). At 65536 bits N needs 2049 limbs (the
4 spare bits), one more than 128 x 16, and the roots ran on 128 x 24 (50% wasted
limbs, and that kernel spills); with L = 17 for the roots too they run on
128 x 17 (6% wasted; it spills less): the roots' steps 1.9 times faster there.
Other boundaries remain: 1537 to 1664 limbs (49152 bits) take 128 x 16, a third
wasted; an L of 13 (and 20 above 2176 limbs) would close them, at some compile
time. The conversions are not yet the largest part, so the number-dimension
transform (3) can wait.
