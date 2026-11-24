// j1_diet.cuh: wt-j1conv bc485ca j1conv/j1_diet.cuh verbatim below this header (do not edit here; resync from wt-j1conv).
// Needs j1_conv_tc.cuh and j1_back2.cuh first.
// J=1 conversions, "diet" versions (issue slots per coefficient as the metric): j1_fwd3_* and j1_back3_*.
// Needs j1_conv.cuh and j1_back2.cuh first (j1_imma, j1_cp16, J1F_*, J1C_* macros, j1_back2_fill).
// Same body interfaces as j1_fwd_body / j1_back2_body (block index, number of blocks, tensor-warp index, number of
// tensor warps, shared-memory base, pointers, n).  Bit-exact vs j1/ref.py like the older kernels.
//
// Optional J1C_N: compile-time n (the body then ignores its n argument, so n must equal J1C_N): A-load (back) and
// residue-store (forward) offsets become immediates.  Both builds are bit-exact; J1C_N saves ~12 (F) / ~24 (B) slots per
// coefficient.
//
// Forward (j1_fwd3): same tables, same shared layout (J1F_SMEM) as j1_fwd; the fill stores (-p mod 2^32, m) instead of
//  (p, m).  MMA rows g, g + 8 = coefficients 2g, 2g + 1 of the m-tile, so each quad ends in ONE 8-byte store per lane.
//  Per (m-tile, prime group): one A base pointer (lane offset tg folded in, every load an immediate offset), quads 0..13
//  in a loop unrolled J1F3_U = 7 (runtime n) or fully (J1C_N) plus a peeled guarded quad 14, LDS / P addresses as
//  immediates, epilogue r = lo + qh * (-p) (one IMAD).  qh and r are computed exactly as in j1_fwd (identical output).
//
// Back (j1_back3_range + j1_back3_fin): same structure and table residency as back2 (30 KB table + 1920 B stage per
// tensor warp), with:
//  1. table columns permuted (tab3, built by the host from backM/backF frags) so that inside each tile pair (A_p, B_p),
//     p = 0..2, lane (g, tg) holds planes 0, 1 (tile A_p) and 2, 3 (tile B_p) of limb 4p + tg for its two rows:
//     12 of the 14 limbs of a range are recombined in-lane with no shuffle (only tile 6 = limbs 12, 13 keeps the
//     1-shuffle pair form);
//  2. MMA rows g, g + 8 = coefficients 2g, 2g + 1, so a lane's two A words of a k-row are adjacent: 30 LDG.64 per
//     m-tile instead of 60 LDG; one base pointer per lane, stepped by a 32-bit byte offset (runtime n) or immediates;
//  3. carries per 7-limb SEGMENT (8 per coefficient, segment 7's is 0): no cross-half shuffle/ripple, no lane-dependent
//     switch; every lane stores its limbs straight to out / the side row and its carry-out c_s to the side row.
//     Bound as in back2: c_s = floor(V_s / 2^224), V_s < Smax 2^224 / 255  =>  c_s < 2^17;
//  4. plane recombination in 32-bit: X01 = S0 + 2^8 S1 and X23 = S2 + 2^8 S3 fit a u32 because every accumulator is
//     S < 2^32 / 257 (host-asserted in test_conv.back3_tab on the actual tables: 255 x column sum of backM/backF bytes;
//     real max 2^23.8).  The stage holds (X01, X23); the chain forms X01 + 2^16 X23 + c with 2 PRMT + 2 carry adds;
//  5. the lane's three stage addresses are packed into two registers outside the m-tile loop (16-bit shared byte
//     addresses: the stage areas must lie in the first 64 KB of the block's shared window).
//  side row (J1D_SROW = 20 words, 80 B): [0..6] c_0..c_6, [7] limb 55, [8 + s] limb 7s (s = 0..7), [16] limb 1,
//  [17] limb 54.  The range pass writes every other limb to out; j1_back3_fin reads the side row (5 x 16 B), adds c_s to
//  limb 7s + 7 (fast path: no carry out of that limb), folds, checks Z' + delta and writes limbs 0, 1, 7, ..., 49, 54;
//  any warp with a miss runs the exact ripple in shared memory (ref.py order).
#ifndef J1_DIET_CUH
#define J1_DIET_CUH
#define J1D_SROW 20
#ifndef J1F3_U
#define J1F3_U 7                                    // quad-loop unroll of j1_fwd3_body (runtime n)
#endif

// ============================================================================================ forward
__device__ __forceinline__ void j1_fwd3_fill(unsigned char* smem, const uint4* __restrict__ tab, const uint2* __restrict__ pm,
                                             u32 grp, u32 tid, u32 nthr) {
    uint4* Bs = reinterpret_cast<uint4*>(smem);
    uint2* Ps = reinterpret_cast<uint2*>(smem + J1F_QG * 7 * 512);
    const uint4* src = tab + size_t(grp * J1F_QG) * 7 * 32;
    for (u32 i = tid; i < J1F_QG * 7 * 32; i += nthr) j1_cp16(Bs + i, src + i);
    j1_cp_commit();
    for (u32 i = tid; i < J1F_QG * 4; i += nthr) { const uint2 v = pm[grp * J1F_QG * 4 + i]; Ps[i] = make_uint2(0u - v.x, v.y); }
    j1_cp_wait_all();
}

// one quad (4 primes 4q + tg) of an m-tile: 14 IMMA, then x = S0 + 2^8 S1 + 2^16 S2 + 2^24 S3 (< 2^46) -> 32-bit Barrett
// r = lo + qh (-p) in [0, 2p) for coefficients 2g (h = 0) and 2g + 1 (h = 1) -> one 8-byte store.
template <bool GUARD>
__device__ __forceinline__ void j1_fwd3_quad(const u32 (&a)[7][4], const uint4* bq, uint2 P, u32* o, bool ok) {
    u32 c0[4] = {0u, 0u, 0u, 0u}, c1[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (u32 ks = 0; ks < 7; ++ks) {
        const uint4 b = bq[ks * 32];
        j1_imma(c0, a[ks], b.x, b.y);
        j1_imma(c1, a[ks], b.z, b.w);
    }
    u32 r[2];
#pragma unroll
    for (u32 h = 0; h < 2; ++h) {
        const u32 S0 = c0[2 * h], S1 = c0[2 * h + 1], S2 = c1[2 * h], S3 = c1[2 * h + 1];
        const u32 lo = S0 + (S1 << 8) + (S2 << 16) + (S3 << 24);
        u32 hi;                                                          // (S0 >> 16) + (S1 >> 8) + S2 + (S3 << 8)
        asm("{\n\t.reg .u32 t0, t1;\n\tshr.u32 t0, %1, 16;\n\tadd.u32 t0, t0, %3;\n\tshr.u32 t1, %2, 8;\n\t"
            "add.u32 t0, t0, t1;\n\tshl.b32 t1, %4, 8;\n\tadd.u32 %0, t0, t1;\n\t}"
            : "=r"(hi) : "r"(S0), "r"(S1), "r"(S2), "r"(S3));
        const u32 qh = __umulhi(hi, P.y) >> 13;
        r[h] = lo + qh * P.x;                                            // P.x = -p mod 2^32
    }
    if (!GUARD || ok) *reinterpret_cast<uint2*>(o) = make_uint2(r[0], r[1]);
}

// MMA rows g, g + 8 = coefficients 2g, 2g + 1 of the m-tile (adjacent in res: one STG.64 per quad and lane).
__device__ __forceinline__ void j1_fwd3_body(const unsigned char* smem, u32 grp, u32 bx, u32 nbx, u32 tw, u32 ntw,
                                             const u32* __restrict__ x, u32* __restrict__ res, u32 n) {
#ifdef J1C_N
    n = J1C_N;                                                           // compile-time n: immediate store offsets
#endif
    const u32 lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    const uint4* Bl = reinterpret_cast<const uint4*>(smem) + lane;
    const uint2* Pl = reinterpret_cast<const uint2*>(smem + J1F_QG * 7 * 512) + tg;
    const u32 ntm = n >> 4, q0 = grp * J1F_QG;
    const bool last_ok = (q0 + J1F_QG - 1) * 4 + tg < J1_K;
    const bool tail_ok = tg < 3;                                         // limb 52 + tg < 55 (ks 6, words w1)
    u32 a[7][4];
#pragma unroll 1
    for (u32 mt = bx * ntw + tw; mt < ntm; mt += nbx * ntw) {
        const u32 r0 = mt * 16 + 2 * g;
        const u32* xa = x + size_t(r0) * J1_L + tg;                      // all A loads: immediate offsets from xa
#pragma unroll
        for (u32 ks = 0; ks < 7; ++ks) {
            a[ks][0] = __ldg(xa + 8 * ks);
            a[ks][1] = __ldg(xa + J1_L + 8 * ks);
            if (ks < 6) {
                a[ks][2] = __ldg(xa + 8 * ks + 4);
                a[ks][3] = __ldg(xa + J1_L + 8 * ks + 4);
            } else {
                a[ks][2] = tail_ok ? __ldg(xa + 8 * ks + 4) : 0u;
                a[ks][3] = tail_ok ? __ldg(xa + J1_L + 8 * ks + 4) : 0u;
            }
        }
        u32* o = res + size_t(q0 * 4 + tg) * n + r0;
#ifdef J1C_N
#pragma unroll
        for (u32 qq = 0; qq + 1 < J1F_QG; ++qq)                          // offsets qq * 4n words: immediates
            j1_fwd3_quad<false>(a, Bl + qq * 7 * 32, Pl[qq * 4], o + size_t(qq) * 4 * J1C_N, true);
        j1_fwd3_quad<true>(a, Bl + (J1F_QG - 1) * 7 * 32, Pl[(J1F_QG - 1) * 4], o + size_t(J1F_QG - 1) * 4 * J1C_N, last_ok);
#else
        const size_t o_step = size_t(4) * n;
#pragma unroll J1F3_U
        for (u32 qq = 0; qq + 1 < J1F_QG; ++qq) {
            j1_fwd3_quad<false>(a, Bl + qq * 7 * 32, Pl[qq * 4], o, true);
            o += o_step;
        }
        j1_fwd3_quad<true>(a, Bl + (J1F_QG - 1) * 7 * 32, Pl[(J1F_QG - 1) * 4], o, last_ok);
#endif
    }
}

extern "C" __global__ void __launch_bounds__(512, 2)
j1_fwd3(const u32* __restrict__ x, u32* __restrict__ res, const uint4* __restrict__ tab, const uint2* __restrict__ pm, u32 n) {
    extern __shared__ __align__(16) unsigned char j1_sm3[];
    j1_fwd3_fill(j1_sm3, tab, pm, blockIdx.y, threadIdx.x, blockDim.x);
    __syncthreads();
    j1_fwd3_body(j1_sm3, blockIdx.y, blockIdx.x, gridDim.x, threadIdx.x >> 5, blockDim.x >> 5, x, res, n);
}

// ============================================================================================ back
__device__ __forceinline__ void j1_sts64(u32 a, u32 x, u32 y) { asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(a), "r"(x), "r"(y) : "memory"); }
__device__ __forceinline__ uint2 j1_lds64(u32 a) {
    uint2 v; asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ u32 j1_opaque(u32 x) { asm volatile("" : "+r"(x)); return x; }   // keep in a register

__device__ __forceinline__ void j1_back3_body(const unsigned char* smem, u32* stage, u32 r, u32 bx, u32 nbx, u32 tw, u32 ntw,
                                              const u32* __restrict__ y, u32* __restrict__ out, u32* __restrict__ side, u32 n) {
#ifdef J1C_N
    n = J1C_N;                                                           // compile-time n: immediate A-load offsets
#endif
    const uint4* T = reinterpret_cast<const uint4*>(smem) + (threadIdx.x & 31);
    const u32 lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    const u32 ntm = n >> 4;
    const u32 crow = lane & 15, hh = lane >> 4;
    const u32 seg = 2 * r + hh;
    // shared byte addresses of this lane's stage slots, packed in two registers that the loop only unpacks
    // (otherwise the compiler rematerialises them from %tid in every m-tile):
    //   st_v: reduce, limb 4p + tg of coefficient 2g + h;  st_6: reduce, tile 6 (even tg);  st_c: chain, row crow, limbs 7hh..
    const u32 sbase = (u32)__cvta_generic_to_shared(stage);
    const u32 pk_vc = j1_opaque((sbase + 4 * (2 * g * J1C_STR + 2 * tg)) | ((sbase + 4 * (crow * J1C_STR + 14 * hh)) << 16));
    const u32 pk_6 = j1_opaque(sbase + 4 * (2 * g * J1C_STR + 2 * (12 + (tg >> 1))));
#ifndef J1C_N
    const u32 st4 = j1_opaque(16 * n);                                   // bytes per 4 rows of y (one 32-bit add + carry)
#endif
    const bool v_ok = tg < 2;                                            // words 116 + tg < 118 (ks 14, w1)
#pragma unroll 1
    for (u32 mt = bx * ntw + tw; mt < ntm; mt += nbx * ntw) {
        u32 acc[8][4];
#pragma unroll
        for (u32 t = 0; t < 8; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0u;
        const u32* yt = y + size_t(tg) * n + mt * 16 + 2 * g;           // MMA rows g, g + 8 = coefficients 2g, 2g + 1
#pragma unroll
        for (u32 p = 0; p < (15 + J1C_PK - 1) / J1C_PK; ++p) {
            constexpr u32 PK = J1C_PK;
            u32 a[PK][4];
#pragma unroll
            for (u32 k = 0; k < PK; ++k) {
                const u32 ks = PK * p + k;
                if (ks >= 15) break;
#ifdef J1C_N
                const u32* p0 = yt + size_t(8 * ks) * J1C_N;
                const u32* p1 = yt + size_t(8 * ks + 4) * J1C_N;
#else
                const u32* p0 = yt;
                const u32* p1 = reinterpret_cast<const u32*>(reinterpret_cast<const char*>(yt) + st4);   // + 4n words
                yt = reinterpret_cast<const u32*>(reinterpret_cast<const char*>(p1) + st4);
#endif
                const uint2 u0 = __ldg(reinterpret_cast<const uint2*>(p0));
                a[k][0] = u0.x; a[k][1] = u0.y;
                if (ks < 14) {
                    const uint2 u1 = __ldg(reinterpret_cast<const uint2*>(p1));
                    a[k][2] = u1.x; a[k][3] = u1.y;
                } else {                                                 // words 116 + tg: 118 = v slot, 119 = 0
                    const uint2 u1 = v_ok ? __ldg(reinterpret_cast<const uint2*>(p1)) : make_uint2(0u, 0u);
                    a[k][2] = u1.x; a[k][3] = u1.y;
                }
            }
#pragma unroll
            for (u32 k = 0; k < PK; ++k) {
                const u32 ks = PK * p + k;
                if (ks >= 15) break;
                const uint4* bk = T + ks * 4 * 32;
                const uint4 b0 = bk[0], b1 = bk[32], b2 = bk[64], b3 = bk[96];
                j1_imma(acc[7], a[k], b3.z, b3.w);
                if (ks == 14) {
#pragma unroll
                    for (u32 h = 0; h < 2; ++h) {
                        u64 t = (u64)(acc[7][2 * h] + (acc[7][2 * h + 1] << 8)) << (16 * tg);   // < 2^32 (table bound)
                        t += __shfl_xor_sync(0xffffffffu, t, 1);
                        t += __shfl_xor_sync(0xffffffffu, t, 2);
                        const u32 v = (u32)((t + (1ull << 44)) >> 45);
                        if (tg == 2) a[k][2 + h] = v;
                    }
                }
                j1_imma(acc[0], a[k], b0.x, b0.y);
                j1_imma(acc[1], a[k], b0.z, b0.w);
                j1_imma(acc[2], a[k], b1.x, b1.y);
                j1_imma(acc[3], a[k], b1.z, b1.w);
                j1_imma(acc[4], a[k], b2.x, b2.y);
                j1_imma(acc[5], a[k], b2.z, b2.w);
                j1_imma(acc[6], a[k], b3.x, b3.y);
            }
        }
        // ---- limb parts (X01, X23) of the range -> stage[coefficient][2 * local limb]; V = X01 + 2^16 X23
#pragma unroll
        for (u32 pp = 0; pp < 3; ++pp) {
#pragma unroll
            for (u32 h = 0; h < 2; ++h) {
                const u32 X01 = acc[2 * pp][2 * h] + (acc[2 * pp][2 * h + 1] << 8);          // < 2^32 (table bound)
                const u32 X23 = acc[2 * pp + 1][2 * h] + (acc[2 * pp + 1][2 * h + 1] << 8);
                j1_sts64((pk_vc & 0xffffu) + 4 * (h * J1C_STR + 8 * pp), X01, X23);
            }
        }
#pragma unroll
        for (u32 h = 0; h < 2; ++h) {                                    // tile 6: limbs 12, 13 (pair form)
            const u32 x2 = acc[6][2 * h] + (acc[6][2 * h + 1] << 8);
            const u32 xo = __shfl_xor_sync(0xffffffffu, x2, 1);
            if (!(tg & 1)) j1_sts64(pk_6 + 4 * h * J1C_STR, x2, xo);
        }
        __syncwarp();
        {
            u32 L[7], c = 0;
#pragma unroll
            for (u32 i = 0; i < 7; ++i) {
                const uint2 xx = j1_lds64((pk_vc >> 16) + 8 * i);       // (X01, X23)
                const u32 sl = __byte_perm(xx.y, 0u, 0x1044), sh16 = __byte_perm(xx.y, 0u, 0x4432);   // X23 << 16, X23 >> 16
                u32 lo, hi;                                              // (hi, lo) = X01 + 2^16 X23 + c  (< 2^49.01)
                asm("add.cc.u32 %0, %2, %3;\n\taddc.u32 %1, %4, 0;\n\tadd.cc.u32 %0, %0, %5;\n\taddc.u32 %1, %1, 0;"
                    : "=r"(lo), "=r"(hi) : "r"(xx.x), "r"(sl), "r"(sh16), "r"(c));
                L[i] = lo;
                c = hi;
            }
            const size_t row = size_t(mt * 16 + crow);
            u32* o7 = out + row * J1_L + 7 * seg;                        // limbs 7 seg .. 7 seg + 6 of this row
            u32* srow = side + row * J1D_SROW;
#pragma unroll
            srow[8 + seg] = L[0];                                        // limb 7 seg
            if (seg == 0) srow[16] = L[1]; else o7[1] = L[1];
            o7[2] = L[2]; o7[3] = L[3]; o7[4] = L[4];
            if (seg == 7) { srow[17] = L[5]; srow[7] = L[6]; }           // limbs 54, 55
            else { o7[5] = L[5]; o7[6] = L[6]; srow[seg] = c; }
        }
        __syncwarp();
    }
}

extern "C" __global__ void __launch_bounds__(128)
j1_back3_fin(u32* __restrict__ out, const u32* __restrict__ side, u32 n) {
    __shared__ u32 rows[4][32 * 56];
    const u32 lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const u32 row = blockIdx.x * 128 + threadIdx.x;
    const u32 sh = 1747 - 32 * 54;                                       // 19
    const bool live = row < n;
    const uint4* sp = reinterpret_cast<const uint4*>(side + size_t(live ? row : 0) * J1D_SROW);
    const uint4 s0 = sp[0], s1 = sp[1], s2 = sp[2], s3 = sp[3], s4 = sp[4];
    const u32 c[7] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z};
    const u32 limb55 = s1.w;
    const u32 A0 = s2.x, A1 = s4.x, L54 = s4.y;
    const u32 R7[7] = {s2.y, s2.z, s2.w, s3.x, s3.y, s3.z, s3.w};       // limbs 7, 14, ..., 49 before the carries
    u32 M[7];
    bool fast = true;
#pragma unroll
    for (u32 i = 0; i < 7; ++i) { M[i] = R7[i] + c[i]; fast &= M[i] >= c[i]; }
    const u32 Hh = (L54 >> sh) | (limb55 << (32 - sh));                  // < 2^17
    const u64 hd = (u64)Hh * J1_DELTA;
    const u64 low = ((u64)A1 << 32) | A0;
    const u64 f = low + hd, t = f + J1_DELTA;
    fast &= (f >= low) && (t >= f);
#ifdef J1C_FORCESLOW
    fast = false;
#endif
    u32* orow = out + size_t(live ? row : 0) * J1_L;
    if (__all_sync(0xffffffffu, fast || !live)) {
        if (live) {
            orow[0] = (u32)f; orow[1] = (u32)(f >> 32);
#pragma unroll
            for (u32 i = 0; i < 7; ++i) orow[7 * i + 7] = M[i];
            orow[54] = L54 & ((1u << sh) - 1);
        }
        return;
    }
    const u32 r0 = blockIdx.x * 128 + w * 32;
    u32* R = rows[w];
    for (u32 i = lane; i < 32 * J1_L; i += 32) { const u32 rr = i / J1_L, j = i % J1_L; R[rr * 56 + j] = (r0 + rr < n) ? out[size_t(r0) * J1_L + i] : 0u; }
    __syncwarp();
    if (live) {
        u32* Z = R + lane * 56;
        Z[0] = A0; Z[1] = A1; Z[54] = L54; Z[55] = limb55;
#pragma unroll
        for (u32 i = 0; i < 7; ++i) Z[7 * i + 7] = R7[i];
#pragma unroll 1
        for (u32 q = 0; q < 7; ++q) {                                    // add the segment carry-outs
            u64 cc = c[q];
#pragma unroll 1
            for (u32 j = 7 * q + 7; j < 56 && cc; ++j) { const u64 v = (u64)Z[j] + cc; Z[j] = (u32)v; cc = v >> 32; }
        }
        const u32 H = (Z[54] >> sh) | (Z[55] << (32 - sh));
        Z[54] &= (1u << sh) - 1; Z[55] = 0;
        u64 cc = (u64)H * J1_DELTA;
#pragma unroll 1
        for (u32 j = 0; j < J1_L && cc; ++j) { const u64 v = (u64)Z[j] + cc; Z[j] = (u32)v; cc = v >> 32; }
        cc = J1_DELTA; u32 top = 0;
#pragma unroll 1
        for (u32 j = 0; j < J1_L; ++j) { const u64 v = (u64)Z[j] + cc; top = (u32)v; cc = v >> 32; }
        if (((top >> sh) != 0) || cc != 0) {
            cc = J1_DELTA;
#pragma unroll 1
            for (u32 j = 0; j < J1_L; ++j) { const u64 v = (u64)Z[j] + cc; u32 xx = (u32)v; cc = v >> 32; if (j == 54) xx &= (1u << sh) - 1; Z[j] = xx; }
        }
    }
    __syncwarp();
    for (u32 i = lane; i < 32 * J1_L; i += 32) { const u32 rr = i / J1_L, j = i % J1_L; if (r0 + rr < n) out[size_t(r0) * J1_L + i] = R[rr * 56 + j]; }
}

extern "C" __global__ void __launch_bounds__(J1C_W * 32, J1C_MINB)
j1_back3_range(const u32* __restrict__ y, u32* __restrict__ out, u32* __restrict__ side, const uint4* __restrict__ tab3, u32 n) {
    extern __shared__ __align__(16) unsigned char j1_sm4[];
    j1_back2_fill(j1_sm4, tab3, blockIdx.y, threadIdx.x, blockDim.x);
    __syncthreads();
    const u32 w = threadIdx.x >> 5;
    j1_back3_body(j1_sm4, reinterpret_cast<u32*>(j1_sm4 + J1C_TAB_BYTES + w * J1C_STAGE_BYTES), blockIdx.y, blockIdx.x,
                  gridDim.x, w, J1C_W, y, out, side, n);
}
#endif
