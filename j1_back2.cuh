// j1_back2.cuh: wt-j1conv 5f0a0cc j1conv/j1_back2.cuh verbatim below this header (do not edit here; resync from wt-j1conv).
// Needs j1_conv_tc.cuh first.
// J=1 back conversion v2 ("back2"): column-range blocks, no lockstep.  Needs j1_conv.cuh first (j1_imma, j1_cp16, ...).
// Contract unchanged: y [118][n] residues in [0, 2p) -> out [n][55] canonical limbs of c mod q (bit-exact vs j1/ref.py).
//
// Stage 1, j1_back2_range (block = one of 4 output-column ranges of 56 columns = 7 n8 tiles, r = blockIdx.y):
//  * the range's slice of backM plus the backF tile is resident in shared memory for the whole launch (30 KB, filled
//    once): [15 ks][4 tile pairs][32 lanes] uint4, pairs (0,1) (2,3) (4,5) (6,F)  (F = fraction tile);
//  * every tensor warp walks its own 16-coefficient m-tiles; K is done in 4 sequential passes of J1C_PK = 4 k32 steps
//    inside the warp (16 A registers live), accumulating 7 + 1 tiles (32 accumulators live across the passes): 8
//    independent chains, one LDS.128 per 2 IMMA, 120 IMMA per m-tile and range (105 main + 15 fraction);
//  * fraction tile complete after k-step 14's fraction IMMA (v slot still 0) -> v = (Tf + 2^44) >> 45 -> data byte 472
//    (shfl into the tg == 2 lanes) -> the 7 main IMMAs of k-step 14 (v is recomputed by each of the 4 ranges: +15 IMMA
//    per range and m-tile, i.e. 30 IMMA per coefficient instead of 27.2, and no cross-block wait);
//  * carries only inside the range, in redundant form: limb values (lo, hi) = S[4j] + 2^8 S[4j+1] + 2^16 S[4j+2] +
//    2^24 S[4j+3] < 2^49.01 are formed in registers (pairs of lanes via one 64-bit shuffle), parked in a 1.9 KB per-warp
//    stage, then all 32 lanes run one carry chain each: row = lane & 15, h = lane >> 4, local limbs 7h .. 7h+6 (28
//    columns); lane h = 1 adds lane h = 0's carry-out (shfl 16) with a full 7-limb register ripple, so the range r
//    (14 limbs = 56 columns) leaves one carry-out word c_r, written to side[row][r].  Limbs 0, 1, 14, 28, 42, 54 and 55
//    go to the side row; the other limbs are restaged row-major and stored coalesced to out.
//    Bound: c_r = floor(V_r / 2^448), V_r = sum_{c<56} S[c] 256^c < Smax * 2^448 / 255, Smax < 2^25 (generic; real
//    tables 2^23.8)  =>  c_r < 2^17 (+1 from the in-register ripple), so c_r fits a u32 and adding it to limb 14(r+1)
//    overflows only if that limb is >= 2^32 - 2^17 (detected).  Range 3: V_3 2^1344 <= Z < 2^1763.7  =>  c_3 = 0.
// Stage 2, j1_back2_fin (thread per coefficient): Z = sum_r (limbs of r) 2^(448 r) + sum_{r<3} c_r 2^(448 (r+1)):
//  fast path adds c_r to limb 14r+14 (checking no carry out of it), folds H = Z >> 1747 into limbs 0, 1 and checks
//  Z' + delta for a carry into limb 2; if any lane of a warp misses the fast path the warp stages its 32 rows
//  (coalesced) in shared memory and runs the exact ripple there (ref.py order), then writes them back coalesced.
//  J1C_FORCESLOW=1 forces that exact path for every row (test probe; exact on all dumped cases).
// side buffer: n x 16 words (64 B per coefficient), scratch between the two launches.
#ifndef J1C_W
#define J1C_W 16                                   // tensor warps per range block
#endif
#ifndef J1C_PK
#define J1C_PK 4                                   // k32 steps per sequential K pass (A registers = 4 x J1C_PK)
#endif
#ifndef J1C_STAGGER
#define J1C_STAGGER 0
#endif
#ifndef J1C_MINB
#define J1C_MINB 2                                 // 2 => 64-register cap at 512 threads
#endif
#define J1C_TAB_BYTES (15 * 4 * 512)               // 30720
#define J1C_STR 30                                 // stage row stride (words): 14 limbs x (lo, hi) + 2 pad
#define J1C_STAGE_BYTES (16 * J1C_STR * 4)         // 1920 per warp
#define J1C_SMEM (J1C_TAB_BYTES + J1C_W * J1C_STAGE_BYTES)

__device__ __forceinline__ void j1_back2_fill(unsigned char* smem, const uint4* __restrict__ tab2, u32 r, u32 tid, u32 nthr) {
    uint4* T = reinterpret_cast<uint4*>(smem);
    const uint4* src = tab2 + size_t(r) * 15 * 4 * 32;
    for (u32 i = tid; i < 15 * 4 * 32; i += nthr) j1_cp16(T + i, src + i);
    j1_cp_commit();
    j1_cp_wait_all();
}

// limbs of segment SEG (global limbs 7 SEG .. 7 SEG + 6): 0, 1, 14, 28, 42, 54, 55 go to the side row, the rest to out.
template <u32 SEG>
__device__ __forceinline__ void j1_b2_side(const u32 (&L)[7], u32* srow) {
#pragma unroll
    for (u32 i = 0; i < 7; ++i) {
        const u32 jj = 7 * SEG + i;
        if (jj == 0) srow[4] = L[i];
        else if (jj == 1) srow[5] = L[i];
        else if (jj == 14 || jj == 28 || jj == 42) srow[5 + jj / 14] = L[i];
        else if (jj == 54) srow[9] = L[i];
        else if (jj == 55) srow[3] = L[i];
    }
}

// call after j1_back2_fill + a barrier.  r = column range (0..3), bx/nbx = block index/count among blocks of range r,
// tw/ntw = tensor warp index/count in the block, stage = this warp's private 1920-byte area.
__device__ __forceinline__ void j1_back2_body(const unsigned char* smem, u32* stage, u32 r, u32 bx, u32 nbx, u32 tw, u32 ntw,
                                              const u32* __restrict__ y, u32* __restrict__ out, u32* __restrict__ side, u32 n) {
    const uint4* T = reinterpret_cast<const uint4*>(smem);
    const u32 lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    const u32 ntm = n >> 4;
    const u32 crow = lane & 15, seg = 2 * r + (lane >> 4);
#if J1C_STAGGER
    __nanosleep((tw >> 2) * J1C_STAGGER);                                // de-phase the 4 warps of an SMSP (ns)
#endif
#pragma unroll 1
    for (u32 mt = bx * ntw + tw; mt < ntm; mt += nbx * ntw) {
        u32 acc[8][4];                                                   // tiles 0..6 of the range, 7 = fraction
#pragma unroll
        for (u32 t = 0; t < 8; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0u;
#ifdef J1C_AONCE
        const u32* yp = y + (mt & (J1C_AONCE - 1)) * 16 + g;              // timing probe only: small y window
#else
        const u32* yp = y + mt * 16 + g;
#endif
#pragma unroll
        for (u32 p = 0; p < (15 + J1C_PK - 1) / J1C_PK; ++p) {
            constexpr u32 PK = J1C_PK;
            u32 a[PK][4];
#pragma unroll
            for (u32 k = 0; k < PK; ++k) {
                const u32 ks = PK * p + k, w0 = 8 * ks + tg, w1 = w0 + 4;
                if (ks >= 15) break;
#ifdef J1C_NOLOAD
                if (mt != bx * ntw + tw) { a[k][0] = a[k][1] = a[k][2] = a[k][3] = ks * 0x01010101u + lane; continue; }  // probe
#endif
                a[k][0] = __ldg(yp + size_t(w0) * n);
                a[k][1] = __ldg(yp + size_t(w0) * n + 8);
                if (ks < 14) {
                    a[k][2] = __ldg(yp + size_t(w1) * n);
                    a[k][3] = __ldg(yp + size_t(w1) * n + 8);
                } else {                                                 // words 116 + tg: 118 = v slot, 119 = 0
                    a[k][2] = (w1 < J1_K) ? __ldg(yp + size_t(w1) * n) : 0u;
                    a[k][3] = (w1 < J1_K) ? __ldg(yp + size_t(w1) * n + 8) : 0u;
                }
            }
#pragma unroll
            for (u32 k = 0; k < PK; ++k) {
                const u32 ks = PK * p + k;
                if (ks >= 15) break;
                const uint4* bk = T + ks * 4 * 32 + lane;
                const uint4 b0 = bk[0], b1 = bk[32], b2 = bk[64], b3 = bk[96];
                j1_imma(acc[7], a[k], b3.z, b3.w);                       // fraction tile (v slot still 0 at ks 14)
                if (ks == 14) {                                          // Tf complete: v -> data byte 472
#pragma unroll
                    for (u32 h = 0; h < 2; ++h) {
                        u64 t = ((u64)acc[7][2 * h] + ((u64)acc[7][2 * h + 1] << 8)) << (16 * tg);
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
#ifdef J1C_NOEPI
        {   // timing probe only: checksum instead of the carry epilogue
            u32 x = 0;
#pragma unroll
            for (u32 t = 0; t < 8; ++t) x += acc[t][0] ^ acc[t][1] ^ acc[t][2] ^ acc[t][3];
            if (x == 0x9e3779b9u) out[mt] = x;
            continue;
        }
#endif
        // ---- limb values of the range in registers: lane (g, tg) has columns 2tg, 2tg+1 of each tile for rows g, g+8;
        //      x = S0 + 2^8 S1 (< 2^33.01); limb 2t + (tg >> 1) = x(tg even) + 2^16 x(tg odd) (< 2^49.01)
#pragma unroll
        for (u32 t = 0; t < 7; ++t) {
#pragma unroll
            for (u32 h = 0; h < 2; ++h) {
                const u64 x = (u64)acc[t][2 * h] + ((u64)acc[t][2 * h + 1] << 8);
                const u64 xo = __shfl_xor_sync(0xffffffffu, x, 1);
                const u64 val = x + (xo << 16);
                if (!(tg & 1))
                    *reinterpret_cast<uint2*>(stage + (g + 8 * h) * J1C_STR + 2 * (2 * t + (tg >> 1))) =
                        make_uint2((u32)val, (u32)(val >> 32));
            }
        }
        __syncwarp();
        // ---- one carry chain per lane: row crow, local limbs 7h .. 7h+6 (h = lane >> 4) of the range; then lane h = 1
        //      adds the carry-out of lane h = 0 (same row, shfl 16) with a full 7-limb ripple in registers, so each range
        //      leaves one carry-out word c_r (< 2^17 + 1; c_3 = 0 since Z < 2^1763.7).
        {
            const u32 h = lane >> 4;
            const u32* sr = stage + crow * J1C_STR + 14 * h;
            u32 L[7], c = 0;
#pragma unroll
            for (u32 i = 0; i < 7; ++i) {
                const uint2 lh = *reinterpret_cast<const uint2*>(sr + 2 * i);
                const u64 t = (u64)lh.x + c;
                L[i] = (u32)t;
                c = (u32)(t >> 32) + lh.y;
            }
            u32 cin = __shfl_xor_sync(0xffffffffu, c, 16);
            if (h) {
#pragma unroll
                for (u32 i = 0; i < 7; ++i) { const u64 t = (u64)L[i] + cin; L[i] = (u32)t; cin = (u32)(t >> 32); }
                c += cin;
            }
            const size_t row = size_t(mt * 16 + crow);
            u32* srow = side + row * 16;
            switch (r * 2 + h) {                                             // limbs 0, 1, 14, 28, 42, 54, 55 -> side
                case 0: j1_b2_side<0>(L, srow); break;
                case 1: j1_b2_side<1>(L, srow); break;
                case 2: j1_b2_side<2>(L, srow); break;
                case 3: j1_b2_side<3>(L, srow); break;
                case 4: j1_b2_side<4>(L, srow); break;
                case 5: j1_b2_side<5>(L, srow); break;
                case 6: j1_b2_side<6>(L, srow); break;
                default: j1_b2_side<7>(L, srow); break;
            }
            // all 14 limbs of the range -> out through the stage, row-major, so each warp store covers ~2 rows of
            // 56 contiguous bytes (limbs 0, 1, 14, 28, 42, 54 are rewritten by the final pass; limb 55 is skipped)
            __syncwarp();
#pragma unroll
            for (u32 i = 0; i < 7; ++i) stage[crow * 14 + 7 * h + i] = L[i];
            __syncwarp();
            u32* ob = out + size_t(mt * 16) * J1_L + 14 * r;
#pragma unroll
            for (u32 q = 0; q < 7; ++q) {
                const u32 idx = 32 * q + lane, rr = idx / 14, li = idx - 14 * rr;
#ifndef J1C_NOSTORE
                if (14 * r + li < J1_L) ob[rr * J1_L + li] = stage[idx];
#else
                if (stage[idx] == 0x9e3779b9u) ob[0] = 0;
#endif
            }
            if (h && r < 3) srow[r] = c;                                     // range carry-out
        }
        __syncwarp();
    }
}

// Stage 2: one thread per coefficient, 128-thread blocks (4 warps x 7 KB exact-path stage).
// side row (16 words): [0..2] carry-out of ranges 0..2, [3] limb 55, [4] L0, [5] L1, [6] L14, [7] L28, [8] L42, [9] L54.
// The range pass never writes limbs 0, 1, 14, 28, 42, 54 to out; this pass does (coalesced side read, 6 limb stores).
extern "C" __global__ void __launch_bounds__(128)
j1_back2_fin(u32* __restrict__ out, const u32* __restrict__ side, u32 n) {
    __shared__ u32 rows[4][32 * 56];
    const u32 lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const u32 row = blockIdx.x * 128 + threadIdx.x;
    const u32 sh = 1747 - 32 * 54;                                       // 19
    const bool live = row < n;
    const uint4* sp = reinterpret_cast<const uint4*>(side + size_t(live ? row : 0) * 16);
    const uint4 s0 = sp[0], s1 = sp[1], s2 = sp[2];
    const u32 cr0 = s0.x, cr1 = s0.y, cr2 = s0.z, limb55 = s0.w, L0 = s1.x, L1 = s1.y, L54 = s2.y;
    const u32 L14 = s1.z + cr0, L28 = s1.w + cr1, L42 = s2.x + cr2;
    bool fast = (L14 >= cr0) && (L28 >= cr1) && (L42 >= cr2);
    const u32 Hh = (L54 >> sh) | (limb55 << (32 - sh));                  // < 2^17
    const u64 hd = (u64)Hh * J1_DELTA;
    const u64 low = ((u64)L1 << 32) | L0;
    const u64 f = low + hd, t = f + J1_DELTA;
    fast &= (f >= low) && (t >= f);
#ifdef J1C_FORCESLOW
    fast = false;                                                        // test probe: exact path on every row
#endif
#ifdef J1C_NOEXACT
    fast = true;                                                         // timing probe only
#endif
    u32* orow = out + size_t(live ? row : 0) * J1_L;
    if (__all_sync(0xffffffffu, fast || !live)) {
        if (live) {
            orow[0] = (u32)f; orow[1] = (u32)(f >> 32); orow[14] = L14; orow[28] = L28; orow[42] = L42;
            orow[54] = L54 & ((1u << sh) - 1);
        }
        return;
    }
    // exact path for the whole warp: coalesced stage of its 32 rows (55 limbs each, row stride 56 words)
    const u32 r0 = blockIdx.x * 128 + w * 32;
    u32* R = rows[w];
    for (u32 i = lane; i < 32 * J1_L; i += 32) { const u32 rr = i / J1_L, j = i % J1_L; R[rr * 56 + j] = (r0 + rr < n) ? out[size_t(r0) * J1_L + i] : 0u; }
    __syncwarp();
    if (live) {
        u32* Z = R + lane * 56;
        Z[0] = L0; Z[1] = L1; Z[14] = s1.z; Z[28] = s1.w; Z[42] = s2.x; Z[54] = L54; Z[55] = limb55;
        const u32 cr[3] = {cr0, cr1, cr2};
#pragma unroll 1
        for (u32 q = 0; q < 3; ++q) {                                    // add the range carry-outs
            u64 c = cr[q];
#pragma unroll 1
            for (u32 j = 14 * q + 14; j < 56 && c; ++j) { const u64 v = (u64)Z[j] + c; Z[j] = (u32)v; c = v >> 32; }
        }
        const u32 H = (Z[54] >> sh) | (Z[55] << (32 - sh));
        Z[54] &= (1u << sh) - 1; Z[55] = 0;
        u64 c = (u64)H * J1_DELTA;
#pragma unroll 1
        for (u32 j = 0; j < J1_L && c; ++j) { const u64 v = (u64)Z[j] + c; Z[j] = (u32)v; c = v >> 32; }
        c = J1_DELTA; u32 top = 0;
#pragma unroll 1
        for (u32 j = 0; j < J1_L; ++j) { const u64 v = (u64)Z[j] + c; top = (u32)v; c = v >> 32; }
        if (((top >> sh) != 0) || c != 0) {                              // Z' >= q: Z' + delta - 2^1747
            c = J1_DELTA;
#pragma unroll 1
            for (u32 j = 0; j < J1_L; ++j) { const u64 v = (u64)Z[j] + c; u32 x = (u32)v; c = v >> 32; if (j == 54) x &= (1u << sh) - 1; Z[j] = x; }
        }
    }
    __syncwarp();
    for (u32 i = lane; i < 32 * J1_L; i += 32) { const u32 rr = i / J1_L, j = i % J1_L; if (r0 + rr < n) out[size_t(r0) * J1_L + i] = R[rr * 56 + j]; }
}

// Standalone stage-1 wrapper: grid (gx, 4 ranges), J1C_W warps, dynamic smem J1C_SMEM.
extern "C" __global__ void __launch_bounds__(J1C_W * 32, J1C_MINB)
j1_back2_range(const u32* __restrict__ y, u32* __restrict__ out, u32* __restrict__ side, const uint4* __restrict__ tab2, u32 n) {
    extern __shared__ __align__(16) unsigned char j1_sm2[];
    j1_back2_fill(j1_sm2, tab2, blockIdx.y, threadIdx.x, blockDim.x);
    __syncthreads();
    const u32 w = threadIdx.x >> 5;
    j1_back2_body(j1_sm2, reinterpret_cast<u32*>(j1_sm2 + J1C_TAB_BYTES + w * J1C_STAGE_BYTES), blockIdx.y, blockIdx.x,
                  gridDim.x, w, J1C_W, y, out, side, n);
}
