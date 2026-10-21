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
| M6 | performance: curve batching, streams, tuning | against the BSGS stage 2 and gpuecm2 at 1024-4096 bits |
| M7 | large N: memory-bounded dF, conversion cost; number-dimension transform | |
