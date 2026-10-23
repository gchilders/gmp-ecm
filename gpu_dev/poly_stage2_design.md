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

## Tree-transform cache

As gpuecm2's -treecache does, the transposed evaluation can reuse the transforms
of F's tree. At level t it needs, for each node with children x^d + A and
x^d + B (d = n / 2^(t+1)) and values T, the middle products MP(A, T) and
MP(B, T). A middle product is a cyclic correlation: at length 2d,
C(f) = T(f) A(-f). The tree build already transformed x^d + A and x^d + B at
that length (gpu_rns_conv_monic), so it can keep them (keepx, keepy), and
gpu_rns_corr_kept reads them at -f and subtracts the x^d term ((-1)^f). The
position of -f for each transform length comes from transforming the
polynomial x once (negmap), so it holds for any output order of the NTT. Each
level then skips the conversion and forward transforms of the children: per
level, from 5 to 3 transform row-units of the evaluation.

gpu_polyalg_treecache (A, bytes) allocates, from the top level down, as many
levels as fit (2 P n B words each), skipping the levels done in Z/N. The
driver reserves the largest RNS scratch first, then gives the cache what is
left less 5% of the GPU's memory and 256 MB. GMPECM_S2POLY_TREECACHE=0
turns it off. The products are the same with and without it (polyalg_test
with no levels, all levels or only the top one; s2poly_bench from 1024 to
8192 bits).

The gain is small, because F's tree (3 row-units per level) and the k trees
of G (3 each) run at the same lengths and gain nothing. On the A10, per
curve, s2poly_bench (best of 5):

| N bits | dF, k | no cache | cache |
|---|---|---|---|
| 1024 | 256, 4 | 0.53 ms | 0.50 ms |
| 1024 | 4096, 6 | 11.97 ms | 11.90 ms |
| 1024 | 16384, 2 | 25.13 ms | 24.64 ms |
| 2048 | 4096, 6 | 39.69 ms | 38.67 ms |
| 4096 | 16384, 2 | 267.3 ms | 261.7 ms |
| 8192 | 16384, 2 | 1798 ms | 1783 ms |

ecm -gpu -resume on the residues compared with gpuecm2 before (stage 2 of all
residues, best of 3; the products are the same as gpuecm2's):

| N bits, B1 (residues) | no cache | cache | gpuecm2 -treecache |
|---|---|---|---|
| 600, 1e6 (64) | 321 ms | 317 ms | 384 ms |
| 1000, 5e4 (512) | 342 ms | 336 ms | 512 ms |
| 1000, 1e6 (64) | 582 ms | 572 ms | 640 ms |
| 1000, 3e6 (32) | 665 ms | 651 ms | 696 ms |
| 1500, 1e6 (64) | 1113 ms | 1087 ms | 1122 ms |

## Transform products for the Z/N arithmetic (Woltman's suggestion)

George Woltman suggested transforming numbers in the number dimension, prime95
style, instead of RNS + CRT. In this stage 2 the RNS conversions are only 10-19%
of the time; the products mod N in the roots (schoolbook gm_mul, gpu_roots.cu) are
the large part at large N. So the idea was tried there:
gpu_dev/ntt_modmul_bench.cu, a standalone batched Montgomery product mod a
general N by integer NTTs, timed against gm_mul in the shapes the roots use.

**Design.** Numbers are d = 2^lg digits of b ~ 17 bits (b d >= bits + 7; b <= 21,
so d = 1024 covers N up to 21.5K bits, 2048 to 43K, 4096 to 86K). Convolutions
run modulo two primes p < 2^30 with Garner CRT (FP64 is out at 1/64 rate; M61 has
no 2-power roots of unity without going to GF(M61^2)). Two choices carry most of
the speed:
- Montgomery with **R = B^d - 1** (B = 2^b) instead of a power of 2. Then T = ab is
  needed only mod B^d - 1 (cyclic) and mod B^d + 1 (negacyclic), both of length d;
  q = T_c N' mod (B^d - 1) is cyclic; and since R = -2 mod (B^d + 1),
  r = (T_n + qN mod (B^d + 1)) / -2 is negacyclic. Nothing is zero padded:
  10 length-d transforms per prime per product (8 when one operand's transforms
  are given), against 14 / 12 for R = 2^k with GMP's mulmod_bnm1 trick. Needs
  gcd (N, 2^(bd) - 1) = 1, which picks b d.
- **Unnormalized signed digits** (prime95): a single carry step
  (c_i = lo + mid B + hi B^2 -> lo_i + mid_(i-1) + hi_(i-2)) leaves digits below
  2^(b+1) + 2^15. The 2^58.6 CRT range has room for that, and outputs stay in
  (-0.3N, 2.35N). The representative of r comes from its top digit.

One group of d/8 threads per product. Digit vectors stay in registers in the
layout that the first forward and last inverse radix-8 passes use. Both primes
run through each shared-memory pass, and the CRT and the carry are fused onto
the last pass.

**Results** (A10, products/s, many independent products, x <- x b / R iterated;
gm_mul keeps everything in registers, so it is an upper bound for the roots'
products; every NTT result checked against GMP):

| N bits | roots' shape | gm_mul there | best gm_mul shape | NTT (d, b) | NTT, b's transforms given | NTT / roots' shape | NTT / best |
|---|---|---|---|---|---|---|---|
| 4096 | 16x9 | 60.6M | 8x17: 75.4M | 16.9M (256, 17) | 22.0M | 0.28-0.36 | 0.22-0.29 |
| 8192 | 32x9 | 15.1M | 16x17: 18.8M | 7.53M (512, 17) | 9.51M | 0.50-0.63 | 0.40-0.51 |
| 16384 | 64x9 | 2.73M | 32x17: 4.77M | 2.86M (1024, 17) | 3.61M | 1.05-1.32 | 0.60-0.76 |
| 32768 | 128x9 | 0.665M | 64x17: 0.942M | 1.20M (2048, 17) | 1.51M | 1.80-2.27 | 1.27-1.60 |
| 49152 | 128x16 | (128x17: 0.241M) | 128x17: 0.241M | 0.500M (4096, 13) | 0.642M | ~2.1-2.7 | 2.08-2.67 |
| 65536 | 128x17 | 0.235M | 128x17: 0.235M | 0.500M (4096, 17) | 0.653M | 2.13-2.77 | 2.13-2.77 |

The NTT cost is a staircase: flat over each d (21497 bits runs at 2.82M / 3.57M
like 10800 bits), while gm_mul grows about as bits^2.3. **Crossover** against the
best gm_mul shape (one-warp 32x17 / 64x17): about 18.5-20.5K bits inside the
d = 1024 step, then again from about 27-30K bits (d = 2048), and the NTT wins
everywhere above. Against the shapes the roots actually use (wide 64/128-thread
groups with L <= 12 above 12K bits, at most half as fast as 32x17), the NTT wins
from about 12-16K bits.

**Stage 2 estimate.** In an Edwards addition (11 products) with each operand's
transforms reused, a product costs about 8.9 transforms per prime, between the two
NTT columns. That gives 3.2M (16K), 1.35M (32K), 0.57M (49K, 65K) products/s,
against the roots' shapes: 1.18x, 2.03x, about 2.1x, 2.44x. The batched inversion
(Z/N shape) gains only from 32K bits (1.43x). Phase profiles (s2poly_bench,
GMPECM_S2POLY_PROF and GMPECM_ROOTS_PROF, GMP-ECM's parameters; the step kernel
is 72-93% of the roots' time):

| N bits | roots share, Dickson(6) dF 4096 k 6 | stage 2 saved | roots share, x^2 dF 1024 k 4 | stage 2 saved |
|---|---|---|---|---|
| 16384 | 61% | ~9% (1.10x) | 44% | ~6% |
| 32768 | 63% | ~31% (1.45x) | 40% | ~20% |
| 49152 | 62% | ~33% (1.48x) | - | - |
| 65536 | 55% | ~33% (1.48x) | 33% | ~19% |

These cover only the roots. The bottom Z/N levels of the product trees (in
"F and tree" and "G") would gain too at 32K bits and up. The figures are
estimates, roughly +-30%: the real step kernel runs below gm_mul's register-only
rate (at 65536 bits it spills), which favours the NTT, but an NTT roots kernel adds
point state in registers, operand transforms in global memory, and its own
barriers.

**Recommendation** (now implemented, see the next section). Not below about 16K bits: at 4-8K bits the schoolbook groups
are 2-4x faster. At 16K bits the gain (~10%) is smaller than what the one-warp
32x17 shape shows over the roots' 64x9 in this bench (1.75x), if the step
kernel's register use allowed that shape. Worth building from 32K bits up: about 1.45-1.5x on
stage 2 with Dickson(6), 1.2x with x^2. It also removes the 98300-bit ceiling of
the roots (d = 8192, 68 KB of shared memory per product, b <= 20 for the CRT
range, would go to about 160K bits). The way to do it is an NTT path in gpu_roots.cu: rt_add on digit vectors,
with the point's four coordinates in registers in the column layout, operand
transforms kept per addition, and a, d, N, N' transformed once per curve. The
Z/N tree and middle-product kernels can share it. Choose it per size by timing,
as stage 1 does with gpu_ntt.cu. That code (3 primes, 32-bit digits, R = 2^k,
one block per curve) wins over the wide groups from 25K bits; this one does
about 2/3 of its transform work at equal length. Cheaper first steps, if 16-32K
matters: non-power-of-two lengths (3 2^k) or larger b to smooth the staircase
(b up to 21 is free); there is headroom for 3 2^k at the same two primes.

**Special N = 2^p +- 1 (cofactors): DWT.** Working mod 2^p +- 1 needs no
reduction at all. 2^p + 1 is a negacyclic product (the psi weights this kernel
already applies), 2^p - 1 a cyclic one. When p is not a multiple of d, digits
are of variable size with Crandall-Fagin weights 2^(ceil(pj/d) - pj/d), which
need a d-th root of 2 mod each prime (choose primes where 2 is a d-th power).
A product is then 2-3 transforms per prime and one carry (the carry wraps
around, with sign for +1), against 8-10 transforms and four carries. From this
bench's per-transform cost, about 3x the transform-Montgomery rate: roughly
2M products/s at 65536 bits (8x gm_mul), 11M at 16K (2.3x), and break-even with
gm_mul near 4-8K bits. Halve those ratios against a schoolbook with special
(shift-add) reduction, which the GPU kernels do not have either. With R = B^d - 1,
choose b d not a multiple of p for 2^p - 1 cofactors (gcd).

## Transform products in the roots and the Z/N levels (done)

The previous section's recommendation, built. **gpu_nmul.cuh** is the
library version of the bench's arithmetic: D = 2^8 .. 2^12 signed digits of b
bits, two primes, Montgomery with R = B^D - 1, one carry step, the column
layout. Its group operations are `__noinline__` per D: nm_hat (a number's four
transforms), nm_mul (one hat given, 8 transforms per prime), nm_mulhh (two hats,
6), sums of products in the transform domain (nm_acc_*), and conversions from
limbs and to limbs reduced to [0, 2N). The bounds are b >= lg + 4 (for the
one-step carry) and b <= (53 - lg) / 2 (the CRT range with operands that are
sums of three numbers). B^D >= 2^10 N; b is raised past gcd (N, B^D - 1) != 1.

**gpu_roots.cu** gets an NTT backend. Numbers are D words, so G->n = D and
every table keeps its layout. There is one block of D/8 threads per instance,
looping over instances with its own scratch. A thread touches only its own
digits of a number, so global memory needs no barriers. The kernels:

- nt_setup (chain, fill, scalar; digits of zero are skipped, which gives the
  same point up to a projective factor).
- nt_step: num/den only for emitted roots.
- nt_vec: the batched inversion.
- nt_tree / nt_mp: the Z/N levels up to degree 16. Per node, hats of its
  numbers (b's or children times R^2 / R), then one transform-domain sum and one
  reduction per coefficient, about 12 / 18 transforms whatever the degree,
  where the schoolbook takes d + 1 products.

An Edwards addition is 98 transforms per prime. In a step, addition k + 1's
first point is addition k's second (not yet updated), so its five hats carry
over: 88.

**Choice** (rt_pick_ntt), from 12000 bits, made once per size and printed with
GMPECM_ROOTS_PROF. The roots and the Z/N levels are decided separately:

- Roots: the NTT is taken if its generic products (vec RV_UP) time within 1.3x
  of the roots' schoolbook shape. Hat reuse and the step kernels' register
  pressure make the real gain larger than the timed one.
- Z/N levels: one level of degree 8, timed both ways.

GMPECM_ROOTS_NTT=0/1 (both parts) and GMPECM_ROOTS_ZNTT=0/1 (Z/N only)
override. s2p_mem asks gpu_roots_words / gpu_roots_scratch_bytes for the
roots' share of memory: NTT numbers are about 2 nl words, and the scratch
is about 300 MB at 32K-65K bits on an A10.

**Tests**:

- gpu_dev/nmul_test.cu: chains of products from one or both hats, 4-term sums,
  differences and 3-term sums as operands, and limb conversions, at every D
  and the b limits, against GMP.
- roots_test.cu `ntt`: every existing case with the NTT forced (D = 2^8),
  plus 4096, 8192 and 24000 bits, root for root against GMP-ECM.
- gpu_dev/roots_zn_test.cu: tree and mp, both backends, 1000 to 65536 bits,
  d up to 32.
- s2poly_bench: "Product of G(f_i)" identical (md5 over all curves) between
  backends and every mixed choice, from 12500 to 65536 bits.

**Stage 2 on the A10** (s2poly_bench, GMP-ECM's parameters, ms per curve;
"NTT roots" is the first version without the Z/N levels or hat reuse; same
products in every column):

| N bits, parameters | schoolbook | NTT roots | final | speedup |
|---|---|---|---|---|
| 32768, x^2, dF 1024, k 4 | 893 | 688 | 538 | 1.66x |
| 65536, x^2, dF 1024, k 4 | 3012 | 2373 | 1576 | 1.91x |
| 32768, Dickson(6), dF 4096, k 6 | 7750 | 4958 | 4135 | 1.87x |
| 65536, Dickson(6), dF 4096, k 6 (4 curves) | 23442 | 17065 | 12512 | 1.87x |

"final" is with the NTT forced. The automatic choice picks NTT for both parts at
these sizes; its first gpu_roots_create per size and process adds the timing
(571, 1631 and 4169 ms per curve for the first three with 8 curves, before
the timing samples were trimmed). The roots' phases drop 2.2-2.8x, and the Z/N
levels in "F and tree", "G" and "evaluation" drop about 2x (65536 x^2: 977 ->
453 ms for G). With the NTT Z/N levels, 16 stays the best
GMPECM_S2POLY_LOWMAX/MPMAX of 0, 4, 8, 16 (65536 x^2: 2032, 1723, 1643,
1576 ms per curve). With the schoolbook Z/N kernels at 65536 bits, 4 had been
better than 16 (1941 against 2365 ms).

Near the crossover (x^2, dF 1024, ms per curve; roots / Z/N backend):

| N bits | sb / sb | sb / NTT | NTT / sb | NTT / NTT | auto picks |
|---|---|---|---|---|---|
| 12500 | 147.7 | 142.1 | 155.1 | 152.4 | sb / NTT: 141.9 |
| 16384 | 199.9 | 191.5 | 192.3 | 183.7 | NTT / NTT: 183.3 |
| 22000 | 409.1 | 352.2 | 433.9 | 380.1 | sb / NTT: 354.4 |
| 26000 | 666.5 | 547.2 | 538.5 | 425.4 | NTT / NTT: 422.3 |

The roots' NTT loses from 21.5K bits (D = 2048 begins) until the schoolbook
shape jumps from 64 x 12 to 128 x 8 at 24.6K bits. The Z/N levels gain from
12K bits on, and might from below (not tried under 12000 bits).

Open:
- The NTT d = 4096 kernels spill a little: 136-360 bytes at 64 registers
  (two blocks of 512 threads per SM).
- NT_ZMAX = 16 bounds the Z/N scratch: 259 D words per block. LOWMAX 32
  would need twice that and is untested.
- Non-power-of-two lengths (3 2^k) would smooth the 21.5K-24.6K bit dip.
- With D = 8192 (b <= 19, 68 KB of shared memory) the roots would go past
  81910 bits (the d = 4096 limit, below the schoolbook's 98300). Not done,
  as RNS limits the polynomial part there too.


## Roots kernel shape for large N

The roots' shape rule capped the limbs per thread at 12 (larger L spilled in the step
kernel at 1024 bits), so from 385 limbs (12284 bits) the roots ran in wide groups of 64
threads (64 x 8, 64 x 9, 64 x 12), whose two warps exchange through shared memory. One
warp with more limbs per thread is faster there: rt_shape now takes, after the shapes
within a warp and the budget, the fewest limbs among the shapes within a warp (any L)
and those within the budget (any group), ties to fewer threads. That gives 32 x 16 for
385-512 limbs, 32 x 17 for 513-544 and 32 x 24 for 577-768 (64 x 9 stays for 545-576,
where 32 x 24 has a third more limbs); the Z/N kernels keep their rule. Above 768 limbs
the NTT backend wins as before.

A10, stage 2 at B1 = 1e5 (dF = 1024, k = 4) on products of 1 Kbit primes, ms per curve,
the same "Product of G(f_i)" for all three:

| N bits | old rule | new rule | gpuecm2 -ntt |
|---|---|---|---|
| 12500 | 123.5 | 102.5 | 81.0 |
| 14000 | 132.5 | 111.3 | 95.3 |
| 16344 | 153.8 | 132.0 | 123.2 |
| 17000 | 167.2 | 142.7 | 151.2 |
| 20000 | 186.2 | 186.1 | 173.9 |
| 24000 | 319.5 | 286.2 | 242.8 |
| 28000 | 371.3 | 371.1 | 330.3 |
| 32760 | 470.2 | 469.9 | 439.2 |

At 20000 bits the NTT roots win against 32 x 24 (the timed choice is right: 5.96 s
against 9.92 s for 32 curves with schoolbook roots and Z/N levels), and at 28000 bits and
up nothing changes. What is left against gpuecm2 at these sizes is in the curve
arithmetic: its dedicated Edwards addition (9M + 1 product by a, the unified formula only
when Z3 = 0) against our unified 11M, and CGBN's products in one warp.
