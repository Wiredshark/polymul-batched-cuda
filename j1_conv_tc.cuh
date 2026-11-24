// Tensor-core J=1 conversions, verbatim from wt-j1conv commit 4e0ea38 (j1conv/j1_conv.cuh; its test_conv.py: ALL EXACT at
// N = 65536 on the local RTX 5070, rerun 2026-11-24).  Wired in by j1_path.h (FHERMA_J1_CONV=ref selects j1_conv_ref.cuh).
// J=1 pure-RNS tensor-core conversions (J1_DESIGN_20261124.md sections 2 and 4; bit-exact contract = j1/ref.py).
//   forward : 224 input bytes per coefficient -> 118 residues in [0, 2p)  (u8 x u8 -> s32 IMMA + 32-bit Barrett)
//   back    : 118 residues in [0, 2p) per coefficient -> 55 canonical limbs of c mod q
// Written as ROLE BODIES for a fused warp-role kernel: every body takes (block index, number of blocks, tensor-warp index
// within the block, ...) and a shared-memory base; a standalone wrapper kernel is at the end.  Budget: 64 registers
// (all warps of a fused kernel get max(role regs)), about 4 tensor warps per SMSP (16 per block).
// m16n8k32 .row.col .u8.u8.s32 fragments (lane = 4g + tg; checked by frag_selftest.py against the PTX ISA layout):
//   A: a0 = A[g][4tg..], a1 = A[g+8][4tg..], a2 = A[g][16+4tg..], a3 = A[g+8][16+4tg..]
//   B: b0 = B[4tg..][g], b1 = B[16+4tg..][g]          C: c0,c1 = C[g][2tg,2tg+1], c2,c3 = C[g+8][2tg,2tg+1]
// Data are the A operand (16 coefficients x k32) held in registers; constants are the B operand streamed from shared
// memory, one LDS.128 per two IMMAs (the two n8 tiles of a pair are host-repacked next to each other per lane).
#ifndef J1_CONV_CUH
#define J1_CONV_CUH
#ifndef J1_TYPES_DEFINED
typedef unsigned int u32;
typedef unsigned long long u64;
#endif
#define J1_K 118
#define J1_L 55
#define J1_DELTA 9699327u

__device__ __forceinline__ void j1_imma(u32 (&c)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void j1_cp16(void* smem, const void* gmem) {
    const u32 s = (u32)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void j1_cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void j1_cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void j1_bar(u32 id, u32 nthreads) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory"); }

// =============================================================================================== forward conversion
// x   : [n][55] u32 limbs (value < q), n % 16 == 0
// res : [118][n] u32 residues in [0, 2p)
// tab : [30 quads][7 ks][32 lanes] uint4 = (frag[ks][2q][lane][0..1], frag[ks][2q+1][lane][0..1]) of fwdA/fwdB_frag
// pm  : [120] uint2 = (p_k, floor(2^61/p_k)), zero padded
// A block serves the prime group grp = quads [grp*QG, grp*QG+QG) for every coefficient; its tensor warps walk the
// 16-coefficient m-tiles independently (no barrier after the table fill).  Shared: QG*7*512 + QG*32 bytes per block.
// Per m-tile and quad: 7 LDS.128, 14 IMMA (2 chains), 2 x (Barrett epilogue + store) per lane.
#ifndef J1F_QG
#define J1F_QG 15
#endif
#ifndef J1F_QUNROLL
#define J1F_QUNROLL 1
#endif
#define J1F_SMEM (J1F_QG * 7 * 512 + J1F_QG * 32)
__device__ __forceinline__ void j1_fwd_fill(unsigned char* smem, const uint4* __restrict__ tab, const uint2* __restrict__ pm,
                                            u32 grp, u32 tid, u32 nthr) {
    uint4* Bs = reinterpret_cast<uint4*>(smem);
    uint2* Ps = reinterpret_cast<uint2*>(smem + J1F_QG * 7 * 512);
    const uint4* src = tab + size_t(grp * J1F_QG) * 7 * 32;
    for (u32 i = tid; i < J1F_QG * 7 * 32; i += nthr) j1_cp16(Bs + i, src + i);
    j1_cp_commit();
    for (u32 i = tid; i < J1F_QG * 4; i += nthr) Ps[i] = pm[grp * J1F_QG * 4 + i];
    j1_cp_wait_all();
}
// call after j1_fwd_fill + a barrier over the filling threads.  bx/nbx: block index / count among blocks of this group;
// tw/ntw: tensor-warp index / count in the block.
__device__ __forceinline__ void j1_fwd_body(const unsigned char* smem, u32 grp, u32 bx, u32 nbx, u32 tw, u32 ntw,
                                            const u32* __restrict__ x, u32* __restrict__ res, u32 n) {
    const uint4* Bs = reinterpret_cast<const uint4*>(smem);
    const uint2* Ps = reinterpret_cast<const uint2*>(smem + J1F_QG * 7 * 512);
    const u32 lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    const u32 ntm = n >> 4, q0 = grp * J1F_QG;
    constexpr int QUNR = J1F_QUNROLL;
    u32 a[7][4];
#pragma unroll 1
    for (u32 mt = bx * ntw + tw; mt < ntm; mt += nbx * ntw) {
        const u32 r0 = mt * 16 + g;
#ifdef J1F_AONCE                                                         // timing probe only: A loaded once
        if (mt == bx * ntw + tw)
#endif
        {
            const u32* x0 = x + size_t(r0) * J1_L;
            const u32* x1 = x0 + 8 * J1_L;
#pragma unroll
            for (u32 ks = 0; ks < 7; ++ks) {
                const u32 w0 = 8 * ks + tg, w1 = 8 * ks + 4 + tg;   // limb 55 = zero pad (ks 6, tg 3)
                a[ks][0] = __ldg(x0 + w0);
                a[ks][1] = __ldg(x1 + w0);
                a[ks][2] = (w1 < J1_L) ? __ldg(x0 + w1) : 0u;
                a[ks][3] = (w1 < J1_L) ? __ldg(x1 + w1) : 0u;
            }
        }
        u32* o = res + size_t(q0 * 4 + tg) * n + r0;
        // epilogue of one quad: x = S0 + 2^8 S1 + 2^16 S2 + 2^24 S3 (< 2^46), 32-bit Barrett -> [0, 2p)
        auto epi = [&](const u32 (&c0)[4], const u32 (&c1)[4], u32 qq) {
#ifdef J1F_NOEPI                                                        // timing probe only: no epilogue except a checksum store
            if (qq + 1 < J1F_QG) return;
#endif
            const uint2 P = Ps[qq * 4 + tg];
#pragma unroll
            for (u32 h = 0; h < 2; ++h) {
                const u32 S0 = c0[2 * h], S1 = c0[2 * h + 1], S2 = c1[2 * h], S3 = c1[2 * h + 1];
                const u32 lo = S0 + (S1 << 8) + (S2 << 16) + (S3 << 24);
                const u32 hi = (S0 >> 16) + (S1 >> 8) + S2 + (S3 << 8);
                const u32 qh = __umulhi(hi, P.y) >> 13;
#ifdef J1F_NOSTORE                                                      // timing probe only: keep the epilogue, drop the store
                if (lo - qh * P.x == 0x9e3779b9u) o[size_t(qq * 4) * n + 8 * h] = 0;
#else
                if (P.x) o[size_t(qq * 4) * n + 8 * h] = lo - qh * P.x;   // primes 118, 119 of the last quad: p = 0
#endif
            }
        };
        auto mm = [&](u32 (&c0)[4], u32 (&c1)[4], u32 qq) {
#ifndef J1F_NOEPI
#pragma unroll
            for (u32 i = 0; i < 4; ++i) c0[i] = c1[i] = 0u;              // planes 0,1 | planes 2,3 of prime 4q+tg
#endif
            const uint4* bq = Bs + qq * 7 * 32 + lane;
#pragma unroll
            for (u32 ks = 0; ks < 7; ++ks) {
                const uint4 b = bq[ks * 32];
                j1_imma(c0, a[ks], b.x, b.y);
                j1_imma(c1, a[ks], b.z, b.w);
            }
        };
#if defined(J1F_DUAL)
        // two quads at once: 4 independent accumulator chains, 2 LDS.128 per k32 step
        static_assert(J1F_QG % 2 == 1 || true, "");
#pragma unroll 1
        for (u32 qq = 0; qq + 1 < J1F_QG; qq += 2) {
            u32 c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0}, d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0};
            const uint4* bq = Bs + qq * 7 * 32 + lane;
#pragma unroll
            for (u32 ks = 0; ks < 7; ++ks) {
                const uint4 b = bq[ks * 32], e = bq[(7 + ks) * 32];
                j1_imma(c0, a[ks], b.x, b.y);
                j1_imma(d0, a[ks], e.x, e.y);
                j1_imma(c1, a[ks], b.z, b.w);
                j1_imma(d1, a[ks], e.z, e.w);
            }
            epi(c0, c1, qq);
            epi(d0, d1, qq + 1);
        }
        if (J1F_QG & 1) { u32 c0[4], c1[4]; mm(c0, c1, J1F_QG - 1); epi(c0, c1, J1F_QG - 1); }
#elif defined(J1F_PIPE)
        // software pipeline: the epilogue of quad q-1 is issued behind the IMMAs of quad q (8 more accumulators)
        u32 e0[4], e1[4], d0[4], d1[4];
        mm(e0, e1, 0);
#pragma unroll 1
        for (u32 qq = 1; qq + 1 < J1F_QG; qq += 2) {
            mm(d0, d1, qq);
            epi(e0, e1, qq - 1);
            mm(e0, e1, qq + 1);
            epi(d0, d1, qq);
        }
        if (J1F_QG % 2 == 0) { mm(d0, d1, J1F_QG - 1); epi(e0, e1, J1F_QG - 2); epi(d0, d1, J1F_QG - 1); }
        else epi(e0, e1, J1F_QG - 1);
#else
#pragma unroll QUNR
        u32 c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
        for (u32 qq = 0; qq < J1F_QG; ++qq) {
            mm(c0, c1, qq);
            epi(c0, c1, qq);
        }
#endif
    }
}

// =============================================================================================== back conversion
// y   : [118][n] u32 residues in [0, 2p) (= c s_k mod p_k), n % 16 == 0
// out : [n][55] u32 canonical limbs of c mod q
// tm  : [14 pairs][15 ks][32 lanes] uint4 = (frag[ks][2c][lane], frag[ks][2c+1][lane]) of backM_frag [15][28][32][2]
// tf  : [15 ks][32 lanes] uint2 = backF_frag
// 64-register structure: the 60 A registers of a 16-coefficient m-tile do not fit one warp, so the K dimension is split
// over a warp PAIR: half 0 holds k32 steps 0..7 (32 regs), half 1 steps 8..14 (28 regs, incl. the v byte 472).
// The 107.5 KB backM table does not fit shared memory next to an integer role, so the block's 16 tensor warps (8 pairs,
// 8 m-tiles = 128 coefficients per round) stream it in 14 column pairs (7.5 KB each) through a cp.async double buffer,
// one named barrier per column pair.  Per column pair c: both halves run 2 accumulator chains (half 0: 16 IMMA, half 1:
// 14 IMMA); half 1 parks its partial sums in shared memory; after the next barrier half 0 adds them, stages the 16 x 16
// column block row-major and lanes 0..15 (one coefficient row each) run the carry chain, storing 4 limbs per row.
// Fraction pass: both halves accumulate their k-range of the backF tile; half 0's partial goes through shared memory at
// barrier 0, half 1 completes Tf, v = (Tf + 2^44) >> 45 and writes v into its A register (k byte 472).  After the last
// barrier half 0 does the fold and conditional subtraction (fast path on limbs 0, 1, 54; exact ripple otherwise).
// Warp roles: h = (tw >> 2) & 1, pair = (tw & 3) | ((tw >> 3) << 2): each SMSP gets two warps of each half.
#define J1B_TW 16                                  // tensor warps per block (fixed: 8 pairs)
#define J1B_CST 20                                 // row-stage stride (words): 16 columns + 4 pad, conflict-free LDS.128
#define J1B_PAIR_BYTES (3 * 16 * J1B_CST * 4 + 512)
#define J1B_SMEM (2 * 7680 + 3840 + 8 * J1B_PAIR_BYTES)
#ifndef J1B_KS1
#define J1B_KS1 7                                  // k32 steps held by half 1 (half 0 holds 15 - J1B_KS1 and runs the carry)
#endif
#ifdef J1B_PROF
__device__ unsigned long long j1_prof[4 * 16 * 1024];
#endif
// One half of a pair (H = 0: k32 steps 0..14-KS1, H = 1: the last KS1 steps incl. the v byte; default 8 + 7).  Both halves execute the same barrier sequence.
template <u32 H>
__device__ __forceinline__ void j1_back_half(unsigned char* smem, u32 bx, u32 nbx, u32 pair, u32 tid, u32 bar_id,
                                             const u32* __restrict__ y, u32* __restrict__ out, const uint4* __restrict__ tm, u32 n) {
    constexpr u32 NT = J1B_TW * 32, KS = H ? J1B_KS1 : 15 - J1B_KS1, KS0 = H ? 15 - J1B_KS1 : 0;
    uint4* Bm = reinterpret_cast<uint4*>(smem);                         // [2][15][32]
    const uint2* Bf = reinterpret_cast<const uint2*>(smem + 2 * 7680);  // [15][32]
    const u32 lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    unsigned char* ps = smem + 2 * 7680 + 3840 + pair * J1B_PAIR_BYTES;
    u32* csB = reinterpret_cast<u32*>(ps);                              // [2][16][20] half-1 row stage (double buffer)
    u32* csA = csB + 2 * 16 * J1B_CST;                                  // [16][20]    half-0 row stage
    uint4* fbuf = reinterpret_cast<uint4*>(csA + 16 * J1B_CST) + lane;  // [32] uint4 : half-0 fraction partial
    const u32 ntm = n >> 4, per_round = nbx * 8, rounds = (ntm + per_round - 1) / per_round;
#ifdef J1B_PROF
    unsigned long long j1_prof_acc[3] = {0, 0, 0};
#endif
    u32 a[KS][4];
    auto loadA = [&](u32 mt) {
        const bool act = mt < ntm;
        const u32* yp = y + mt * 16 + g;
#pragma unroll
        for (u32 k = 0; k < KS; ++k) {
            const u32 w0 = 8 * (KS0 + k) + tg, w1 = w0 + 4;
            a[k][0] = act ? __ldg(yp + size_t(w0) * n) : 0u;
            a[k][1] = act ? __ldg(yp + size_t(w0) * n + 8) : 0u;
            const bool ok1 = act && (8 * (KS0 + k) + 7 < J1_K || w1 < J1_K);       // k-step 14: words 116 + tg
            a[k][2] = ok1 ? __ldg(yp + size_t(w1) * n) : 0u;                       // (118 = v slot, 119 = 0)
            a[k][3] = ok1 ? __ldg(yp + size_t(w1) * n + 8) : 0u;
        }
    };
    loadA(bx * 8 + pair);
#pragma unroll 1
    for (u32 rd = 0; rd < rounds; ++rd) {
        const u32 mt = (rd * nbx + bx) * 8 + pair;
        const bool active = mt < ntm;
        if (rd == 0) j1_bar(bar_id, NT);                                 // Bf filled
        u32 fs[4];
        {
            u32 f0[4] = {0, 0, 0, 0}, f1[4] = {0, 0, 0, 0};
#pragma unroll
            for (u32 k = 0; k < KS; ++k) {
                const uint2 b = Bf[(KS0 + k) * 32 + lane];
                if (k & 1) j1_imma(f1, a[k], b.x, b.y); else j1_imma(f0, a[k], b.x, b.y);
            }
#pragma unroll
            for (u32 i = 0; i < 4; ++i) fs[i] = f0[i] + f1[i];
            if (H == 0) *fbuf = make_uint4(fs[0], fs[1], fs[2], fs[3]);
        }
        u32 carry = 0, limb55 = 0;
        const u32 crow = lane & 15;
        u32* orow = out + size_t(mt * 16 + crow) * J1_L;
#pragma unroll 1
        for (u32 c = 0; c <= 14; ++c) {
#ifdef J1B_PROF
            unsigned long long t0 = clock64();
#endif
            if (c <= 13) j1_cp_wait_all();
#ifdef J1B_NOBAR
            if (c == 0)
#endif
            j1_bar(bar_id, NT);
#ifdef J1B_PROF
            unsigned long long t1 = clock64(); j1_prof_acc[0] += t1 - t0;
#endif
#ifdef J1B_NOSTREAM
            if (false) {
#else
            if (c < 13 || (c == 13 && rd + 1 < rounds)) {
#endif                // prefetch the next column-pair table
                const u32 nc = c < 13 ? c + 1 : 0;
                const uint4* src = tm + size_t(nc) * 15 * 32;
                uint4* dst = Bm + (nc & 1) * 15 * 32;
                for (u32 i = tid; i < 15 * 32; i += NT) j1_cp16(dst + i, src + i);
                j1_cp_commit();
            }
            if (!active) continue;
            if (H == 1 && c == 0) {                                      // complete Tf, v -> data byte 472
                const uint4 o = *fbuf;
                const u32 s[4] = {fs[0] + o.x, fs[1] + o.y, fs[2] + o.z, fs[3] + o.w};
#pragma unroll
                for (u32 r = 0; r < 2; ++r) {
                    u64 t = ((u64)s[2 * r] + ((u64)s[2 * r + 1] << 8)) << (16 * tg);
                    t += __shfl_xor_sync(0xffffffffu, t, 1);
                    t += __shfl_xor_sync(0xffffffffu, t, 2);
                    const u32 v = (u32)((t + (1ull << 44)) >> 45);
                    if (tg == 2) a[KS - 1][2 + r] = v;                   // word 118 (k-step 14, half 1, byte 0)
                }
            }
#ifdef J1B_NOCARRY
            if (H == 0 && c >= 1 && lane < 16 && carry == 0x9e3779b9u) {
#else
            if (H == 0 && c >= 1 && lane < 16) {                         // carry chain of column pair c-1
#endif
                const u32* sa = csA + crow * J1B_CST;
                const u32* sb = csB + ((c - 1) & 1) * 16 * J1B_CST + crow * J1B_CST;
#pragma unroll
                for (u32 i = 0; i < 4; ++i) {
                    const uint4 x = *reinterpret_cast<const uint4*>(sa + 4 * i);
                    const uint4 z = *reinterpret_cast<const uint4*>(sb + 4 * i);
                    const u64 acc = (u64)carry + (x.x + z.x) + ((u64)(x.y + z.y) << 8) + ((u64)(x.z + z.z) << 16) +
                                    ((u64)(x.w + z.w) << 24);
                    const u32 j = 4 * (c - 1) + i;
                    if (j < J1_L) orow[j] = (u32)acc; else limb55 = (u32)acc;
                    carry = (u32)(acc >> 32);
                }
            }
            if (H == 0) __syncwarp();
#ifdef J1B_PROF
            unsigned long long t2 = clock64(); j1_prof_acc[1] += t2 - t1;
#endif
            if (c < 14) {
                u32 c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                const uint4* bp = Bm + (c & 1) * 15 * 32 + KS0 * 32 + lane;
#pragma unroll
                for (u32 k = 0; k < KS; ++k) { const uint4 b = bp[k * 32]; j1_imma(c0, a[k], b.x, b.y); j1_imma(c1, a[k], b.z, b.w); }
                // row-major stage of this half's 16 x 16 partial column block (half 1 double-buffered)
                u32* st = H ? csB + (c & 1) * 16 * J1B_CST : csA;
                *reinterpret_cast<uint2*>(st + g * J1B_CST + 2 * tg) = make_uint2(c0[0], c0[1]);
                *reinterpret_cast<uint2*>(st + (g + 8) * J1B_CST + 2 * tg) = make_uint2(c0[2], c0[3]);
                *reinterpret_cast<uint2*>(st + g * J1B_CST + 8 + 2 * tg) = make_uint2(c1[0], c1[1]);
                *reinterpret_cast<uint2*>(st + (g + 8) * J1B_CST + 8 + 2 * tg) = make_uint2(c1[2], c1[3]);
#ifdef J1B_PROF
                { unsigned long long t3 = clock64(); j1_prof_acc[2] += t3 - t2; }
#endif
#ifndef J1B_NOPREFETCH
                if (c == 13 && rd + 1 < rounds) loadA(mt + per_round);  // next round's A (registers dead now)
#endif
            }
        }
#ifdef J1B_NOPREFETCH
        if (rd + 1 < rounds) loadA(mt + per_round);
#endif
        // ---- fold + conditional subtraction (half 0, lanes 0..15 = rows)
        if (H == 0 && active && lane < 16) {
            const u32 sh = 1747 - 32 * 54;                                // 19
            const u32 L54 = orow[54];
            const u32 Hh = (L54 >> sh) | (limb55 << (32 - sh));           // < 2^17
            const u64 hd = (u64)Hh * J1_DELTA;                             // < 2^41
            const u64 low = ((u64)orow[1] << 32) | orow[0];
            const u64 f = low + hd;
            const u64 t = f + J1_DELTA;
            if (f >= low && t >= f) {                                      // no carry into limb 2 from the fold nor from
                orow[0] = (u32)f; orow[1] = (u32)(f >> 32);                // +delta: Z' + delta < 2^1747, i.e. Z' < q
                orow[54] = L54 & ((1u << sh) - 1);
            } else {                                                       // exact ripple in ref.py order
                orow[54] = L54 & ((1u << sh) - 1);
                u64 cr = hd;
#pragma unroll 1
                for (u32 j = 0; j < J1_L; ++j) { const u64 s = (u64)orow[j] + cr; orow[j] = (u32)s; cr = s >> 32; }
                cr = J1_DELTA;
                u32 top = 0;
#pragma unroll 1
                for (u32 j = 0; j < J1_L; ++j) { const u64 s = (u64)orow[j] + cr; top = (u32)s; cr = s >> 32; }
                if (((top >> sh) != 0) || cr != 0) {                       // Z' >= q: write Z' + delta - 2^1747
                    cr = J1_DELTA;
#pragma unroll 1
                    for (u32 j = 0; j < J1_L; ++j) {
                        const u64 s = (u64)orow[j] + cr; u32 w = (u32)s; cr = s >> 32;
                        if (j == 54) w &= (1u << sh) - 1;
                        orow[j] = w;
                    }
                }
            }
        }
    }
#ifdef J1B_PROF
    if (lane == 0) { const u32 w = (blockIdx.x * 16 + (threadIdx.x >> 5)) * 4; j1_prof[w] = j1_prof_acc[0]; j1_prof[w + 1] = j1_prof_acc[1]; j1_prof[w + 2] = j1_prof_acc[2]; j1_prof[w + 3] = H; }
#endif
}
// Role body: tw = tensor-warp index in the block (0..15), tid = thread index among the 512 tensor threads,
// bar_id = named barrier reserved for the tensor role.  Shared: J1B_SMEM bytes (tables 19.2 KB + 3.75 KB per pair).
__device__ __forceinline__ void j1_back_body(unsigned char* smem, u32 bx, u32 nbx, u32 tw, u32 tid, u32 bar_id,
                                             const u32* __restrict__ y, u32* __restrict__ out, const uint4* __restrict__ tm,
                                             const uint2* __restrict__ tf, u32 n) {
    uint4* Bm = reinterpret_cast<uint4*>(smem);
    uint2* Bf = reinterpret_cast<uint2*>(smem + 2 * 7680);
    for (u32 i = tid; i < 15 * 32; i += J1B_TW * 32) Bf[i] = tf[i];
    for (u32 i = tid; i < 15 * 32; i += J1B_TW * 32) j1_cp16(Bm + i, tm + i);
    j1_cp_commit();
    const u32 pair = (tw & 3) | ((tw >> 3) << 2);
    if ((tw >> 2) & 1) j1_back_half<1>(smem, bx, nbx, pair, tid, bar_id, y, out, tm, n);
    else j1_back_half<0>(smem, bx, nbx, pair, tid, bar_id, y, out, tm, n);
}

// =============================================================================================== standalone wrappers
extern "C" __global__ void __launch_bounds__(512, 2)
j1_fwd(const u32* __restrict__ x, u32* __restrict__ res, const uint4* __restrict__ tab, const uint2* __restrict__ pm, u32 n) {
    extern __shared__ __align__(16) unsigned char j1_sm[];
    j1_fwd_fill(j1_sm, tab, pm, blockIdx.y, threadIdx.x, blockDim.x);
    __syncthreads();
    j1_fwd_body(j1_sm, blockIdx.y, blockIdx.x, gridDim.x, threadIdx.x >> 5, blockDim.x >> 5, x, res, n);
}
extern "C" __global__ void __launch_bounds__(J1B_TW * 32, 2)
j1_back(const u32* __restrict__ y, u32* __restrict__ out, const uint4* __restrict__ tm, const uint2* __restrict__ tf, u32 n) {
    extern __shared__ __align__(16) unsigned char j1_sm[];
    j1_back_body(j1_sm, blockIdx.x, gridDim.x, threadIdx.x >> 5, threadIdx.x, 1, y, out, tm, tf, n);
}
#endif
