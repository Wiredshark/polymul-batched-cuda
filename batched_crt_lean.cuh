// Production CRT kernel bpo_crt_lean (default; FHERMA_CRT_OLD=1 falls back to bpo_crt_tc_cm). Port of the sweep variant
// "cxa_o" = sw_crt_lean<3,3973> of crt_sweep.cuh (branch crt2, 8c24c5a) with its flag set fixed: h read from the warp's
// smem stage (1), cp.async residue prefetch of digit j+1 (4), PTX digit chains (128), [base+imm] smem addressing (256),
// PTX final chains (512), per-coefficient output stores (1024), alpha*NP + OFF folded into the column sums (2048).
// Same launch geometry / parameters / static smem / launch bounds (128 threads, 3 blocks/SM) / output as bpo_crt_tc_cm;
// same MMA stage (production B fragments, h = 16-bit column sums, alpha from the reciprocal tile). On the CUDA cores:
//  * lean digit accumulation: Z_j*2^sh is never canonicalised on its own. Z_j = U_j + OFF - alpha*P exactly (the design
//    invariant of the production mod-2^512 form), so word w of Z_j*2^sh is the sum of <= 2 column sums h_k << s_k,
//    alpha*NP<<sh, OFF<<sh and (top word) -alpha*2^(512+sh): one carry chain into acc per digit.
//  * short final: one fold (the host bound gives x1 < 2^W + 2^(W-2) < 2q), t = x1 - bias, then a single +-delta chain
//    selected by (borrow, bit W, t >= q) instead of fold 2, -q, -bias, +q.
#ifdef BTC_F
typedef long long i64;
__device__ __forceinline__ void swc_cpa16(u32 dst,const void* src){asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(dst),"l"(src):"memory");}
__device__ __forceinline__ void swc_commit(){asm volatile("cp.async.commit_group;":::"memory");}
__device__ __forceinline__ void swc_wait0(){asm volatile("cp.async.wait_group 0;":::"memory");}
// ---- generated (PTX carry chains: each row updates acc in place, no 64-bit temporaries) ----
constexpr u32 SWC_NPW[16]={BTC_NP};
__host__ __device__ constexpr u32 swc_nps(u32 sh,u32 w){return sh?((w<16?SWC_NPW[w]<<sh:0u)|(w?SWC_NPW[w-1]>>(32-sh):0u)):(w<16?SWC_NPW[w]:0u);}
template<u32 M> __device__ __forceinline__ void swc_mac(u32* a,const u32* x){   // a[0..17] += sum_i x_i*M*2^(32i)
    asm volatile("mad.lo.cc.u32 %0, %18, %34, %0;\n\t" "madc.lo.cc.u32 %1, %19, %34, %1;\n\t" "madc.lo.cc.u32 %2, %20, %34, %2;\n\t" "madc.lo.cc.u32 %3, %21, %34, %3;\n\t" "madc.lo.cc.u32 %4, %22, %34, %4;\n\t" "madc.lo.cc.u32 %5, %23, %34, %5;\n\t" "madc.lo.cc.u32 %6, %24, %34, %6;\n\t" "madc.lo.cc.u32 %7, %25, %34, %7;\n\t" "madc.lo.cc.u32 %8, %26, %34, %8;\n\t" "madc.lo.cc.u32 %9, %27, %34, %9;\n\t" "madc.lo.cc.u32 %10, %28, %34, %10;\n\t" "madc.lo.cc.u32 %11, %29, %34, %11;\n\t" "madc.lo.cc.u32 %12, %30, %34, %12;\n\t" "madc.lo.cc.u32 %13, %31, %34, %13;\n\t" "madc.lo.cc.u32 %14, %32, %34, %14;\n\t" "madc.lo.cc.u32 %15, %33, %34, %15;\n\t" "addc.cc.u32 %16, %16, 0;\n\t" "addc.u32 %17, %17, 0;\n\t" "mad.hi.cc.u32 %1, %18, %34, %1;\n\t" "madc.hi.cc.u32 %2, %19, %34, %2;\n\t" "madc.hi.cc.u32 %3, %20, %34, %3;\n\t" "madc.hi.cc.u32 %4, %21, %34, %4;\n\t" "madc.hi.cc.u32 %5, %22, %34, %5;\n\t" "madc.hi.cc.u32 %6, %23, %34, %6;\n\t" "madc.hi.cc.u32 %7, %24, %34, %7;\n\t" "madc.hi.cc.u32 %8, %25, %34, %8;\n\t" "madc.hi.cc.u32 %9, %26, %34, %9;\n\t" "madc.hi.cc.u32 %10, %27, %34, %10;\n\t" "madc.hi.cc.u32 %11, %28, %34, %11;\n\t" "madc.hi.cc.u32 %12, %29, %34, %12;\n\t" "madc.hi.cc.u32 %13, %30, %34, %13;\n\t" "madc.hi.cc.u32 %14, %31, %34, %14;\n\t" "madc.hi.cc.u32 %15, %32, %34, %15;\n\t" "madc.hi.cc.u32 %16, %33, %34, %16;\n\t" "addc.u32 %17, %17, 0;\n\t"
        :"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]),"+r"(a[4]),"+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(a[8]),"+r"(a[9]),"+r"(a[10]),"+r"(a[11]),"+r"(a[12]),"+r"(a[13]),"+r"(a[14]),"+r"(a[15]),"+r"(a[16]),"+r"(a[17])
        :"r"(x[0]),"r"(x[1]),"r"(x[2]),"r"(x[3]),"r"(x[4]),"r"(x[5]),"r"(x[6]),"r"(x[7]),"r"(x[8]),"r"(x[9]),"r"(x[10]),"r"(x[11]),"r"(x[12]),"r"(x[13]),"r"(x[14]),"r"(x[15]),"n"(M));
}
template<u32 SH> __device__ __forceinline__ void swc_macn(u32* a,u32 alpha){   // a[0..18] += alpha*(NP<<SH)
    asm volatile("mad.lo.cc.u32 %0, %19, %20, %0;\n\t" "madc.lo.cc.u32 %1, %19, %21, %1;\n\t" "madc.lo.cc.u32 %2, %19, %22, %2;\n\t" "madc.lo.cc.u32 %3, %19, %23, %3;\n\t" "madc.lo.cc.u32 %4, %19, %24, %4;\n\t" "madc.lo.cc.u32 %5, %19, %25, %5;\n\t" "madc.lo.cc.u32 %6, %19, %26, %6;\n\t" "madc.lo.cc.u32 %7, %19, %27, %7;\n\t" "madc.lo.cc.u32 %8, %19, %28, %8;\n\t" "madc.lo.cc.u32 %9, %19, %29, %9;\n\t" "madc.lo.cc.u32 %10, %19, %30, %10;\n\t" "madc.lo.cc.u32 %11, %19, %31, %11;\n\t" "madc.lo.cc.u32 %12, %19, %32, %12;\n\t" "madc.lo.cc.u32 %13, %19, %33, %13;\n\t" "madc.lo.cc.u32 %14, %19, %34, %14;\n\t" "madc.lo.cc.u32 %15, %19, %35, %15;\n\t" "madc.lo.cc.u32 %16, %19, %36, %16;\n\t" "addc.cc.u32 %17, %17, 0;\n\t" "addc.u32 %18, %18, 0;\n\t" "mad.hi.cc.u32 %1, %19, %20, %1;\n\t" "madc.hi.cc.u32 %2, %19, %21, %2;\n\t" "madc.hi.cc.u32 %3, %19, %22, %3;\n\t" "madc.hi.cc.u32 %4, %19, %23, %4;\n\t" "madc.hi.cc.u32 %5, %19, %24, %5;\n\t" "madc.hi.cc.u32 %6, %19, %25, %6;\n\t" "madc.hi.cc.u32 %7, %19, %26, %7;\n\t" "madc.hi.cc.u32 %8, %19, %27, %8;\n\t" "madc.hi.cc.u32 %9, %19, %28, %9;\n\t" "madc.hi.cc.u32 %10, %19, %29, %10;\n\t" "madc.hi.cc.u32 %11, %19, %30, %11;\n\t" "madc.hi.cc.u32 %12, %19, %31, %12;\n\t" "madc.hi.cc.u32 %13, %19, %32, %13;\n\t" "madc.hi.cc.u32 %14, %19, %33, %14;\n\t" "madc.hi.cc.u32 %15, %19, %34, %15;\n\t" "madc.hi.cc.u32 %16, %19, %35, %16;\n\t" "madc.hi.cc.u32 %17, %19, %36, %17;\n\t" "addc.u32 %18, %18, 0;\n\t"
        :"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]),"+r"(a[4]),"+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(a[8]),"+r"(a[9]),"+r"(a[10]),"+r"(a[11]),"+r"(a[12]),"+r"(a[13]),"+r"(a[14]),"+r"(a[15]),"+r"(a[16]),"+r"(a[17]),"+r"(a[18])
        :"r"(alpha),"n"(swc_nps(SH,0)),"n"(swc_nps(SH,1)),"n"(swc_nps(SH,2)),"n"(swc_nps(SH,3)),"n"(swc_nps(SH,4)),"n"(swc_nps(SH,5)),"n"(swc_nps(SH,6)),"n"(swc_nps(SH,7)),"n"(swc_nps(SH,8)),"n"(swc_nps(SH,9)),"n"(swc_nps(SH,10)),"n"(swc_nps(SH,11)),"n"(swc_nps(SH,12)),"n"(swc_nps(SH,13)),"n"(swc_nps(SH,14)),"n"(swc_nps(SH,15)),"n"(swc_nps(SH,16)));
}
// ---- end generated ----
// ---- generated 2 (final stage chains) ----
constexpr u32 SWC_BW[55]={BCRT_BIAS},SWC_QW[55]={BCRT_Q};
__device__ __forceinline__ void swc_fold(u32* a,const u32* hw,u32 d){   // a[0..54] += d*sum hw_w 2^(32w)
    asm volatile("mad.lo.cc.u32 %0, %55, %65, %0;\n\t" "madc.lo.cc.u32 %1, %56, %65, %1;\n\t" "madc.lo.cc.u32 %2, %57, %65, %2;\n\t" "madc.lo.cc.u32 %3, %58, %65, %3;\n\t" "madc.lo.cc.u32 %4, %59, %65, %4;\n\t" "madc.lo.cc.u32 %5, %60, %65, %5;\n\t" "madc.lo.cc.u32 %6, %61, %65, %6;\n\t" "madc.lo.cc.u32 %7, %62, %65, %7;\n\t" "madc.lo.cc.u32 %8, %63, %65, %8;\n\t" "madc.lo.cc.u32 %9, %64, %65, %9;\n\t" "addc.cc.u32 %10, %10, 0;\n\t" "addc.cc.u32 %11, %11, 0;\n\t" "addc.cc.u32 %12, %12, 0;\n\t" "addc.cc.u32 %13, %13, 0;\n\t" "addc.cc.u32 %14, %14, 0;\n\t" "addc.cc.u32 %15, %15, 0;\n\t" "addc.cc.u32 %16, %16, 0;\n\t" "addc.cc.u32 %17, %17, 0;\n\t" "addc.cc.u32 %18, %18, 0;\n\t" "addc.cc.u32 %19, %19, 0;\n\t" "addc.cc.u32 %20, %20, 0;\n\t" "addc.cc.u32 %21, %21, 0;\n\t" "addc.cc.u32 %22, %22, 0;\n\t" "addc.cc.u32 %23, %23, 0;\n\t" "addc.cc.u32 %24, %24, 0;\n\t" "addc.cc.u32 %25, %25, 0;\n\t" "addc.cc.u32 %26, %26, 0;\n\t" "addc.cc.u32 %27, %27, 0;\n\t" "addc.cc.u32 %28, %28, 0;\n\t" "addc.cc.u32 %29, %29, 0;\n\t" "addc.cc.u32 %30, %30, 0;\n\t" "addc.cc.u32 %31, %31, 0;\n\t" "addc.cc.u32 %32, %32, 0;\n\t" "addc.cc.u32 %33, %33, 0;\n\t" "addc.cc.u32 %34, %34, 0;\n\t" "addc.cc.u32 %35, %35, 0;\n\t" "addc.cc.u32 %36, %36, 0;\n\t" "addc.cc.u32 %37, %37, 0;\n\t" "addc.cc.u32 %38, %38, 0;\n\t" "addc.cc.u32 %39, %39, 0;\n\t" "addc.cc.u32 %40, %40, 0;\n\t" "addc.cc.u32 %41, %41, 0;\n\t" "addc.cc.u32 %42, %42, 0;\n\t" "addc.cc.u32 %43, %43, 0;\n\t" "addc.cc.u32 %44, %44, 0;\n\t" "addc.cc.u32 %45, %45, 0;\n\t" "addc.cc.u32 %46, %46, 0;\n\t" "addc.cc.u32 %47, %47, 0;\n\t" "addc.cc.u32 %48, %48, 0;\n\t" "addc.cc.u32 %49, %49, 0;\n\t" "addc.cc.u32 %50, %50, 0;\n\t" "addc.cc.u32 %51, %51, 0;\n\t" "addc.cc.u32 %52, %52, 0;\n\t" "addc.cc.u32 %53, %53, 0;\n\t" "addc.u32 %54, %54, 0;\n\t" "mad.hi.cc.u32 %1, %55, %65, %1;\n\t" "madc.hi.cc.u32 %2, %56, %65, %2;\n\t" "madc.hi.cc.u32 %3, %57, %65, %3;\n\t" "madc.hi.cc.u32 %4, %58, %65, %4;\n\t" "madc.hi.cc.u32 %5, %59, %65, %5;\n\t" "madc.hi.cc.u32 %6, %60, %65, %6;\n\t" "madc.hi.cc.u32 %7, %61, %65, %7;\n\t" "madc.hi.cc.u32 %8, %62, %65, %8;\n\t" "madc.hi.cc.u32 %9, %63, %65, %9;\n\t" "madc.hi.cc.u32 %10, %64, %65, %10;\n\t" "addc.cc.u32 %11, %11, 0;\n\t" "addc.cc.u32 %12, %12, 0;\n\t" "addc.cc.u32 %13, %13, 0;\n\t" "addc.cc.u32 %14, %14, 0;\n\t" "addc.cc.u32 %15, %15, 0;\n\t" "addc.cc.u32 %16, %16, 0;\n\t" "addc.cc.u32 %17, %17, 0;\n\t" "addc.cc.u32 %18, %18, 0;\n\t" "addc.cc.u32 %19, %19, 0;\n\t" "addc.cc.u32 %20, %20, 0;\n\t" "addc.cc.u32 %21, %21, 0;\n\t" "addc.cc.u32 %22, %22, 0;\n\t" "addc.cc.u32 %23, %23, 0;\n\t" "addc.cc.u32 %24, %24, 0;\n\t" "addc.cc.u32 %25, %25, 0;\n\t" "addc.cc.u32 %26, %26, 0;\n\t" "addc.cc.u32 %27, %27, 0;\n\t" "addc.cc.u32 %28, %28, 0;\n\t" "addc.cc.u32 %29, %29, 0;\n\t" "addc.cc.u32 %30, %30, 0;\n\t" "addc.cc.u32 %31, %31, 0;\n\t" "addc.cc.u32 %32, %32, 0;\n\t" "addc.cc.u32 %33, %33, 0;\n\t" "addc.cc.u32 %34, %34, 0;\n\t" "addc.cc.u32 %35, %35, 0;\n\t" "addc.cc.u32 %36, %36, 0;\n\t" "addc.cc.u32 %37, %37, 0;\n\t" "addc.cc.u32 %38, %38, 0;\n\t" "addc.cc.u32 %39, %39, 0;\n\t" "addc.cc.u32 %40, %40, 0;\n\t" "addc.cc.u32 %41, %41, 0;\n\t" "addc.cc.u32 %42, %42, 0;\n\t" "addc.cc.u32 %43, %43, 0;\n\t" "addc.cc.u32 %44, %44, 0;\n\t" "addc.cc.u32 %45, %45, 0;\n\t" "addc.cc.u32 %46, %46, 0;\n\t" "addc.cc.u32 %47, %47, 0;\n\t" "addc.cc.u32 %48, %48, 0;\n\t" "addc.cc.u32 %49, %49, 0;\n\t" "addc.cc.u32 %50, %50, 0;\n\t" "addc.cc.u32 %51, %51, 0;\n\t" "addc.cc.u32 %52, %52, 0;\n\t" "addc.cc.u32 %53, %53, 0;\n\t" "addc.u32 %54, %54, 0;\n\t"
        :"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]),"+r"(a[4]),"+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(a[8]),"+r"(a[9]),"+r"(a[10]),"+r"(a[11]),"+r"(a[12]),"+r"(a[13]),"+r"(a[14]),"+r"(a[15]),"+r"(a[16]),"+r"(a[17]),"+r"(a[18]),"+r"(a[19]),"+r"(a[20]),"+r"(a[21]),"+r"(a[22]),"+r"(a[23]),"+r"(a[24]),"+r"(a[25]),"+r"(a[26]),"+r"(a[27]),"+r"(a[28]),"+r"(a[29]),"+r"(a[30]),"+r"(a[31]),"+r"(a[32]),"+r"(a[33]),"+r"(a[34]),"+r"(a[35]),"+r"(a[36]),"+r"(a[37]),"+r"(a[38]),"+r"(a[39]),"+r"(a[40]),"+r"(a[41]),"+r"(a[42]),"+r"(a[43]),"+r"(a[44]),"+r"(a[45]),"+r"(a[46]),"+r"(a[47]),"+r"(a[48]),"+r"(a[49]),"+r"(a[50]),"+r"(a[51]),"+r"(a[52]),"+r"(a[53]),"+r"(a[54])
        :"r"(hw[0]),"r"(hw[1]),"r"(hw[2]),"r"(hw[3]),"r"(hw[4]),"r"(hw[5]),"r"(hw[6]),"r"(hw[7]),"r"(hw[8]),"r"(hw[9]),"r"(d));
}
__device__ __forceinline__ u32 swc_subbias(u32* a){   // a -= bias (mod 2^1760); returns 0xffffffff on borrow
    u32 br;
    asm volatile("sub.cc.u32 %0, %0, %56;\n\t" "subc.cc.u32 %1, %1, %57;\n\t" "subc.cc.u32 %2, %2, %58;\n\t" "subc.cc.u32 %3, %3, %59;\n\t" "subc.cc.u32 %4, %4, %60;\n\t" "subc.cc.u32 %5, %5, %61;\n\t" "subc.cc.u32 %6, %6, %62;\n\t" "subc.cc.u32 %7, %7, %63;\n\t" "subc.cc.u32 %8, %8, %64;\n\t" "subc.cc.u32 %9, %9, %65;\n\t" "subc.cc.u32 %10, %10, %66;\n\t" "subc.cc.u32 %11, %11, %67;\n\t" "subc.cc.u32 %12, %12, %68;\n\t" "subc.cc.u32 %13, %13, %69;\n\t" "subc.cc.u32 %14, %14, %70;\n\t" "subc.cc.u32 %15, %15, %71;\n\t" "subc.cc.u32 %16, %16, %72;\n\t" "subc.cc.u32 %17, %17, %73;\n\t" "subc.cc.u32 %18, %18, %74;\n\t" "subc.cc.u32 %19, %19, %75;\n\t" "subc.cc.u32 %20, %20, %76;\n\t" "subc.cc.u32 %21, %21, %77;\n\t" "subc.cc.u32 %22, %22, %78;\n\t" "subc.cc.u32 %23, %23, %79;\n\t" "subc.cc.u32 %24, %24, %80;\n\t" "subc.cc.u32 %25, %25, %81;\n\t" "subc.cc.u32 %26, %26, %82;\n\t" "subc.cc.u32 %27, %27, %83;\n\t" "subc.cc.u32 %28, %28, %84;\n\t" "subc.cc.u32 %29, %29, %85;\n\t" "subc.cc.u32 %30, %30, %86;\n\t" "subc.cc.u32 %31, %31, %87;\n\t" "subc.cc.u32 %32, %32, %88;\n\t" "subc.cc.u32 %33, %33, %89;\n\t" "subc.cc.u32 %34, %34, %90;\n\t" "subc.cc.u32 %35, %35, %91;\n\t" "subc.cc.u32 %36, %36, %92;\n\t" "subc.cc.u32 %37, %37, %93;\n\t" "subc.cc.u32 %38, %38, %94;\n\t" "subc.cc.u32 %39, %39, %95;\n\t" "subc.cc.u32 %40, %40, %96;\n\t" "subc.cc.u32 %41, %41, %97;\n\t" "subc.cc.u32 %42, %42, %98;\n\t" "subc.cc.u32 %43, %43, %99;\n\t" "subc.cc.u32 %44, %44, %100;\n\t" "subc.cc.u32 %45, %45, %101;\n\t" "subc.cc.u32 %46, %46, %102;\n\t" "subc.cc.u32 %47, %47, %103;\n\t" "subc.cc.u32 %48, %48, %104;\n\t" "subc.cc.u32 %49, %49, %105;\n\t" "subc.cc.u32 %50, %50, %106;\n\t" "subc.cc.u32 %51, %51, %107;\n\t" "subc.cc.u32 %52, %52, %108;\n\t" "subc.cc.u32 %53, %53, %109;\n\t" "subc.cc.u32 %54, %54, %110;\n\t" "subc.u32 %55, 0, 0;\n\t"
        :"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]),"+r"(a[4]),"+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(a[8]),"+r"(a[9]),"+r"(a[10]),"+r"(a[11]),"+r"(a[12]),"+r"(a[13]),"+r"(a[14]),"+r"(a[15]),"+r"(a[16]),"+r"(a[17]),"+r"(a[18]),"+r"(a[19]),"+r"(a[20]),"+r"(a[21]),"+r"(a[22]),"+r"(a[23]),"+r"(a[24]),"+r"(a[25]),"+r"(a[26]),"+r"(a[27]),"+r"(a[28]),"+r"(a[29]),"+r"(a[30]),"+r"(a[31]),"+r"(a[32]),"+r"(a[33]),"+r"(a[34]),"+r"(a[35]),"+r"(a[36]),"+r"(a[37]),"+r"(a[38]),"+r"(a[39]),"+r"(a[40]),"+r"(a[41]),"+r"(a[42]),"+r"(a[43]),"+r"(a[44]),"+r"(a[45]),"+r"(a[46]),"+r"(a[47]),"+r"(a[48]),"+r"(a[49]),"+r"(a[50]),"+r"(a[51]),"+r"(a[52]),"+r"(a[53]),"+r"(a[54]),"=r"(br)
        :"n"(SWC_BW[0]),"n"(SWC_BW[1]),"n"(SWC_BW[2]),"n"(SWC_BW[3]),"n"(SWC_BW[4]),"n"(SWC_BW[5]),"n"(SWC_BW[6]),"n"(SWC_BW[7]),"n"(SWC_BW[8]),"n"(SWC_BW[9]),"n"(SWC_BW[10]),"n"(SWC_BW[11]),"n"(SWC_BW[12]),"n"(SWC_BW[13]),"n"(SWC_BW[14]),"n"(SWC_BW[15]),"n"(SWC_BW[16]),"n"(SWC_BW[17]),"n"(SWC_BW[18]),"n"(SWC_BW[19]),"n"(SWC_BW[20]),"n"(SWC_BW[21]),"n"(SWC_BW[22]),"n"(SWC_BW[23]),"n"(SWC_BW[24]),"n"(SWC_BW[25]),"n"(SWC_BW[26]),"n"(SWC_BW[27]),"n"(SWC_BW[28]),"n"(SWC_BW[29]),"n"(SWC_BW[30]),"n"(SWC_BW[31]),"n"(SWC_BW[32]),"n"(SWC_BW[33]),"n"(SWC_BW[34]),"n"(SWC_BW[35]),"n"(SWC_BW[36]),"n"(SWC_BW[37]),"n"(SWC_BW[38]),"n"(SWC_BW[39]),"n"(SWC_BW[40]),"n"(SWC_BW[41]),"n"(SWC_BW[42]),"n"(SWC_BW[43]),"n"(SWC_BW[44]),"n"(SWC_BW[45]),"n"(SWC_BW[46]),"n"(SWC_BW[47]),"n"(SWC_BW[48]),"n"(SWC_BW[49]),"n"(SWC_BW[50]),"n"(SWC_BW[51]),"n"(SWC_BW[52]),"n"(SWC_BW[53]),"n"(SWC_BW[54]));
    return br;
}
__device__ __forceinline__ u32 swc_ltq(const u32* a){   // 0xffffffff iff a < q
    u32 t;
    asm volatile("sub.cc.u32 %0, %1, %56;\n\t" "subc.cc.u32 %0, %2, %57;\n\t" "subc.cc.u32 %0, %3, %58;\n\t" "subc.cc.u32 %0, %4, %59;\n\t" "subc.cc.u32 %0, %5, %60;\n\t" "subc.cc.u32 %0, %6, %61;\n\t" "subc.cc.u32 %0, %7, %62;\n\t" "subc.cc.u32 %0, %8, %63;\n\t" "subc.cc.u32 %0, %9, %64;\n\t" "subc.cc.u32 %0, %10, %65;\n\t" "subc.cc.u32 %0, %11, %66;\n\t" "subc.cc.u32 %0, %12, %67;\n\t" "subc.cc.u32 %0, %13, %68;\n\t" "subc.cc.u32 %0, %14, %69;\n\t" "subc.cc.u32 %0, %15, %70;\n\t" "subc.cc.u32 %0, %16, %71;\n\t" "subc.cc.u32 %0, %17, %72;\n\t" "subc.cc.u32 %0, %18, %73;\n\t" "subc.cc.u32 %0, %19, %74;\n\t" "subc.cc.u32 %0, %20, %75;\n\t" "subc.cc.u32 %0, %21, %76;\n\t" "subc.cc.u32 %0, %22, %77;\n\t" "subc.cc.u32 %0, %23, %78;\n\t" "subc.cc.u32 %0, %24, %79;\n\t" "subc.cc.u32 %0, %25, %80;\n\t" "subc.cc.u32 %0, %26, %81;\n\t" "subc.cc.u32 %0, %27, %82;\n\t" "subc.cc.u32 %0, %28, %83;\n\t" "subc.cc.u32 %0, %29, %84;\n\t" "subc.cc.u32 %0, %30, %85;\n\t" "subc.cc.u32 %0, %31, %86;\n\t" "subc.cc.u32 %0, %32, %87;\n\t" "subc.cc.u32 %0, %33, %88;\n\t" "subc.cc.u32 %0, %34, %89;\n\t" "subc.cc.u32 %0, %35, %90;\n\t" "subc.cc.u32 %0, %36, %91;\n\t" "subc.cc.u32 %0, %37, %92;\n\t" "subc.cc.u32 %0, %38, %93;\n\t" "subc.cc.u32 %0, %39, %94;\n\t" "subc.cc.u32 %0, %40, %95;\n\t" "subc.cc.u32 %0, %41, %96;\n\t" "subc.cc.u32 %0, %42, %97;\n\t" "subc.cc.u32 %0, %43, %98;\n\t" "subc.cc.u32 %0, %44, %99;\n\t" "subc.cc.u32 %0, %45, %100;\n\t" "subc.cc.u32 %0, %46, %101;\n\t" "subc.cc.u32 %0, %47, %102;\n\t" "subc.cc.u32 %0, %48, %103;\n\t" "subc.cc.u32 %0, %49, %104;\n\t" "subc.cc.u32 %0, %50, %105;\n\t" "subc.cc.u32 %0, %51, %106;\n\t" "subc.cc.u32 %0, %52, %107;\n\t" "subc.cc.u32 %0, %53, %108;\n\t" "subc.cc.u32 %0, %54, %109;\n\t" "subc.cc.u32 %0, %55, %110;\n\t" "subc.u32 %0, 0, 0;\n\t"
        :"=r"(t)
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]),"r"(a[16]),"r"(a[17]),"r"(a[18]),"r"(a[19]),"r"(a[20]),"r"(a[21]),"r"(a[22]),"r"(a[23]),"r"(a[24]),"r"(a[25]),"r"(a[26]),"r"(a[27]),"r"(a[28]),"r"(a[29]),"r"(a[30]),"r"(a[31]),"r"(a[32]),"r"(a[33]),"r"(a[34]),"r"(a[35]),"r"(a[36]),"r"(a[37]),"r"(a[38]),"r"(a[39]),"r"(a[40]),"r"(a[41]),"r"(a[42]),"r"(a[43]),"r"(a[44]),"r"(a[45]),"r"(a[46]),"r"(a[47]),"r"(a[48]),"r"(a[49]),"r"(a[50]),"r"(a[51]),"r"(a[52]),"r"(a[53]),"r"(a[54]),"n"(SWC_QW[0]),"n"(SWC_QW[1]),"n"(SWC_QW[2]),"n"(SWC_QW[3]),"n"(SWC_QW[4]),"n"(SWC_QW[5]),"n"(SWC_QW[6]),"n"(SWC_QW[7]),"n"(SWC_QW[8]),"n"(SWC_QW[9]),"n"(SWC_QW[10]),"n"(SWC_QW[11]),"n"(SWC_QW[12]),"n"(SWC_QW[13]),"n"(SWC_QW[14]),"n"(SWC_QW[15]),"n"(SWC_QW[16]),"n"(SWC_QW[17]),"n"(SWC_QW[18]),"n"(SWC_QW[19]),"n"(SWC_QW[20]),"n"(SWC_QW[21]),"n"(SWC_QW[22]),"n"(SWC_QW[23]),"n"(SWC_QW[24]),"n"(SWC_QW[25]),"n"(SWC_QW[26]),"n"(SWC_QW[27]),"n"(SWC_QW[28]),"n"(SWC_QW[29]),"n"(SWC_QW[30]),"n"(SWC_QW[31]),"n"(SWC_QW[32]),"n"(SWC_QW[33]),"n"(SWC_QW[34]),"n"(SWC_QW[35]),"n"(SWC_QW[36]),"n"(SWC_QW[37]),"n"(SWC_QW[38]),"n"(SWC_QW[39]),"n"(SWC_QW[40]),"n"(SWC_QW[41]),"n"(SWC_QW[42]),"n"(SWC_QW[43]),"n"(SWC_QW[44]),"n"(SWC_QW[45]),"n"(SWC_QW[46]),"n"(SWC_QW[47]),"n"(SWC_QW[48]),"n"(SWC_QW[49]),"n"(SWC_QW[50]),"n"(SWC_QW[51]),"n"(SWC_QW[52]),"n"(SWC_QW[53]),"n"(SWC_QW[54]));
    return t;
}
__device__ __forceinline__ void swc_addx(u32* a,u32 dd,u32 ext){   // a += dd + ext*(2^1760-2^32)/(2^32-1)
    asm volatile("add.cc.u32 %0, %0, %55;\n\t" "addc.cc.u32 %1, %1, %56;\n\t" "addc.cc.u32 %2, %2, %56;\n\t" "addc.cc.u32 %3, %3, %56;\n\t" "addc.cc.u32 %4, %4, %56;\n\t" "addc.cc.u32 %5, %5, %56;\n\t" "addc.cc.u32 %6, %6, %56;\n\t" "addc.cc.u32 %7, %7, %56;\n\t" "addc.cc.u32 %8, %8, %56;\n\t" "addc.cc.u32 %9, %9, %56;\n\t" "addc.cc.u32 %10, %10, %56;\n\t" "addc.cc.u32 %11, %11, %56;\n\t" "addc.cc.u32 %12, %12, %56;\n\t" "addc.cc.u32 %13, %13, %56;\n\t" "addc.cc.u32 %14, %14, %56;\n\t" "addc.cc.u32 %15, %15, %56;\n\t" "addc.cc.u32 %16, %16, %56;\n\t" "addc.cc.u32 %17, %17, %56;\n\t" "addc.cc.u32 %18, %18, %56;\n\t" "addc.cc.u32 %19, %19, %56;\n\t" "addc.cc.u32 %20, %20, %56;\n\t" "addc.cc.u32 %21, %21, %56;\n\t" "addc.cc.u32 %22, %22, %56;\n\t" "addc.cc.u32 %23, %23, %56;\n\t" "addc.cc.u32 %24, %24, %56;\n\t" "addc.cc.u32 %25, %25, %56;\n\t" "addc.cc.u32 %26, %26, %56;\n\t" "addc.cc.u32 %27, %27, %56;\n\t" "addc.cc.u32 %28, %28, %56;\n\t" "addc.cc.u32 %29, %29, %56;\n\t" "addc.cc.u32 %30, %30, %56;\n\t" "addc.cc.u32 %31, %31, %56;\n\t" "addc.cc.u32 %32, %32, %56;\n\t" "addc.cc.u32 %33, %33, %56;\n\t" "addc.cc.u32 %34, %34, %56;\n\t" "addc.cc.u32 %35, %35, %56;\n\t" "addc.cc.u32 %36, %36, %56;\n\t" "addc.cc.u32 %37, %37, %56;\n\t" "addc.cc.u32 %38, %38, %56;\n\t" "addc.cc.u32 %39, %39, %56;\n\t" "addc.cc.u32 %40, %40, %56;\n\t" "addc.cc.u32 %41, %41, %56;\n\t" "addc.cc.u32 %42, %42, %56;\n\t" "addc.cc.u32 %43, %43, %56;\n\t" "addc.cc.u32 %44, %44, %56;\n\t" "addc.cc.u32 %45, %45, %56;\n\t" "addc.cc.u32 %46, %46, %56;\n\t" "addc.cc.u32 %47, %47, %56;\n\t" "addc.cc.u32 %48, %48, %56;\n\t" "addc.cc.u32 %49, %49, %56;\n\t" "addc.cc.u32 %50, %50, %56;\n\t" "addc.cc.u32 %51, %51, %56;\n\t" "addc.cc.u32 %52, %52, %56;\n\t" "addc.cc.u32 %53, %53, %56;\n\t" "addc.u32 %54, %54, %56;\n\t"
        :"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]),"+r"(a[4]),"+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(a[8]),"+r"(a[9]),"+r"(a[10]),"+r"(a[11]),"+r"(a[12]),"+r"(a[13]),"+r"(a[14]),"+r"(a[15]),"+r"(a[16]),"+r"(a[17]),"+r"(a[18]),"+r"(a[19]),"+r"(a[20]),"+r"(a[21]),"+r"(a[22]),"+r"(a[23]),"+r"(a[24]),"+r"(a[25]),"+r"(a[26]),"+r"(a[27]),"+r"(a[28]),"+r"(a[29]),"+r"(a[30]),"+r"(a[31]),"+r"(a[32]),"+r"(a[33]),"+r"(a[34]),"+r"(a[35]),"+r"(a[36]),"+r"(a[37]),"+r"(a[38]),"+r"(a[39]),"+r"(a[40]),"+r"(a[41]),"+r"(a[42]),"+r"(a[43]),"+r"(a[44]),"+r"(a[45]),"+r"(a[46]),"+r"(a[47]),"+r"(a[48]),"+r"(a[49]),"+r"(a[50]),"+r"(a[51]),"+r"(a[52]),"+r"(a[53]),"+r"(a[54])
        :"r"(dd),"r"(ext));
}
// ---- end generated 2 ----
constexpr u32 SWC_POS[8]={BCRT_POS},SWC_OFW[16]={BTC_OFF};
__host__ __device__ constexpr u32 swc_ofs(u32 sh,u32 w){return sh?((w<16?SWC_OFW[w]<<sh:0u)|(w?SWC_OFW[w-1]>>(32-sh):0u)):(w<16?SWC_OFW[w]:0u);}
__host__ __device__ constexpr u32 swc_ofs_first(u32 sh){for(u32 w=0;w<17;++w)if(swc_ofs(sh,w))return w;return 16;}
// digit J into acc[o..o+18] (o = POS_J/32, sh = POS_J%32): Z_J*2^sh = U*2^sh + alpha*NP*2^sh + OFF*2^sh - alpha*2^(512+sh),
// U = sum_k h_k 2^(16k): even h_2i at bit 32i+sh, odd h_2i+1 at bit 32i+16+sh. Words above o+16 are zero afterwards (Z_J < 2^512;
// acc[64..65] are scratch: x < 2^2041, arithmetic mod 2^2048).
// AIN: alpha*NP + OFF folded into the 16-bit column sums first (h_k += alpha*NP16_k + OFF16_k stays < 2^30.01: H < 2^30,
// alpha <= 16), so the alpha row needs 32 IMADs instead of two 17-word carry chains.
__host__ __device__ constexpr u32 swc_np16(u32 k){return (SWC_NPW[k/2]>>(16*(k&1)))&0xffffu;}
__host__ __device__ constexpr u32 swc_of16(u32 k){return (SWC_OFW[k/2]>>(16*(k&1)))&0xffffu;}
template<u32 J,bool AIN=false> __device__ __forceinline__ void swc_digit(u32 (&acc)[66],const u32* x,u32 alpha){
    constexpr u32 o=SWC_POS[J]/32,sh=SWC_POS[J]%32;
    u32 xe[16],xo[16];
    #pragma unroll
    for(u32 i=0;i<16;++i){const uint2 v=*reinterpret_cast<const uint2*>(x+2*i);xe[i]=v.x;xo[i]=v.y;
        if(AIN){xe[i]+=alpha*swc_np16(2*i)+swc_of16(2*i);xo[i]+=alpha*swc_np16(2*i+1)+swc_of16(2*i+1);}}
    swc_mac<(1u<<sh)>(acc+o,xe);
    if(sh<16)swc_mac<(1u<<((sh+16)&31))>(acc+o,xo);else swc_mac<(1u<<((sh+16)&31))>(acc+o+1,xo);
    if(!AIN)swc_macn<sh>(acc+o,alpha);
    constexpr u32 w0=AIN?16u:swc_ofs_first(sh);
    i64 c=0;
    #pragma unroll
    for(u32 w=w0;w<=18;++w){
        const u32 ex=w==16?(alpha<<sh):(w==17&&sh?(alpha>>((32-sh)&31)):0u);
        const i64 t=i64(acc[o+w])+i64(AIN?0u:swc_ofs(sh,w))-i64(ex)+c;acc[o+w]=u32(t);c=t>>32;}
}
// explicit [base+imm] shared addressing (XA): without it the compiler hoists ~50 loop-invariant per-store / per-load shared
// addresses out of the digit loop and keeps them in registers for the whole kernel.
template<u32 OFF> __device__ __forceinline__ void swc_sts(u32 base,u32 v){asm volatile("st.shared.u32 [%0+%1], %2;"::"r"(base),"n"(OFF),"r"(v):"memory");}
template<u32 OFF> __device__ __forceinline__ u32 swc_lds(u32 base){u32 v;asm volatile("ld.shared.u32 %0, [%1+%2];":"=r"(v):"r"(base),"n"(OFF):"memory");return v;}
// MMA tile T for both m tiles: column sums -> h stage (sb = stage + (gid*36+tig)*4 bytes)
template<u32 T> __device__ __forceinline__ void swc_tile(const u32 (&a)[2][2][4],const u32* bsh,u32 lane,u32 sb){
    const uint2 b0=*reinterpret_cast<const uint2*>(bsh+((T*2+0)*32+lane)*2);
    const uint2 b1=*reinterpret_cast<const uint2*>(bsh+((T*2+1)*32+lane)*2);
    u32 c[4]={0,0,0,0};
    imma16832(c,a[0][0][0],a[0][0][1],a[0][0][2],a[0][0][3],b0.x,b0.y);imma16832(c,a[0][1][0],a[0][1][1],a[0][1][2],a[0][1][3],b1.x,b1.y);
    swc_sts<(0*36+4*T)*4>(sb,c[0]+(c[1]<<8));swc_sts<(8*36+4*T)*4>(sb,c[2]+(c[3]<<8));
    u32 e[4]={0,0,0,0};
    imma16832(e,a[1][0][0],a[1][0][1],a[1][0][2],a[1][0][3],b0.x,b0.y);imma16832(e,a[1][1][0],a[1][1][1],a[1][1][2],a[1][1][3],b1.x,b1.y);
    swc_sts<(16*36+4*T)*4>(sb,e[0]+(e[1]<<8));swc_sts<(24*36+4*T)*4>(sb,e[2]+(e[3]<<8));
}
extern "C" __global__ void __launch_bounds__(128,3) bpo_crt_lean(const u32* __restrict__ residues,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){
    if(pfitems){
        const u32 total=gridDim.x*blockDim.x;
        for(u32 t=blockIdx.x*blockDim.x+threadIdx.x;t<pfitems*256u;t+=total){
            const u32 it=t>>8,hh=t&255u;const u32* ps=(it>=256u?pfb:pfa)+u64(it&255u)*BCRT_L+u64(hh*256u)*BCRT_L;
            asm volatile("prefetch.global.L2 [%0];"::"l"(ps));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+BCRT_L-1));
        }
    }
    constexpr u32 K=BCRT_K,SW=BCRT_SW,W=BCRT_W,L=BCRT_L,NW=64,NT=9,SST=36,F=BTC_F;
    static_assert(K==16&&SW==16,"tensor CRT geometry");
    constexpr u32 WO=W/32,WS=W%32;
    static_assert(WO+1==L&&WS>0,"q = 2^W - d with W in the top limb");
    constexpr u32 STW=32*L;   // per-warp stage words
    static_assert(32*SST+512<=STW,"prefetch buffer aliases the output stage");
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 stage_all[4][STW];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    u32* st=stage_all[warp];
    const u32 i0=blockIdx.x*128+warp*32;
    u32* const pbuf=st+32*SST;   // [r][32 coefficients], coefficient index ^ 8*(r&3) (conflict-free fragment reads)
    auto prefetch=[&](u32 j){
        #pragma unroll
        for(u32 k=0;k<4;++k){const u32 q=lane+32*k,r=q>>3,ch=q&7;
            swc_cpa16(u32(__cvta_generic_to_shared(pbuf+r*32+((4*ch)^(8*(r&3))))),residues+(u64(r)*8+j)*N+i0+4*ch);}
        swc_commit();};
    prefetch(0);
    for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128)bsh[x]=bfrag[x];
    __syncthreads();
    u32 acc[NW+2];   // [64..65]: scratch of the digit chains
    #pragma unroll
    for(u32 w=0;w<NW+2;++w)acc[w]=0;
    #pragma unroll
    for(u32 j=0;j<8;++j){
        u32 a[2][2][4];
        swc_wait0();__syncwarp();
        {   // pbk[k] = pbuf + 32*tig + gid + 8*(k^tig) words, k = 2m + (row half)
            const u32 pb=u32(__cvta_generic_to_shared(pbuf))+(32*tig+gid)*4;
            const u32 p0=pb+32*(0^tig),p1=pb+32*(1^tig),p2=pb+32*(2^tig),p3=pb+32*(3^tig);
#define SWC_FR(m,s,q0,q1) a[m][s][0]=swc_lds<(8*s)*128>(q0);a[m][s][1]=swc_lds<(8*s)*128>(q1);a[m][s][2]=swc_lds<(8*s+4)*128>(q0);a[m][s][3]=swc_lds<(8*s+4)*128>(q1);
            SWC_FR(0,0,p0,p1) SWC_FR(0,1,p0,p1) SWC_FR(1,0,p2,p3) SWC_FR(1,1,p2,p3)
#undef SWC_FR
        }
        __syncwarp();
        if(j<7)prefetch(j+1);
        {const u32 sb=u32(__cvta_generic_to_shared(st))+(gid*SST+tig)*4;
            swc_tile<0>(a,bsh,lane,sb);swc_tile<1>(a,bsh,lane,sb);swc_tile<2>(a,bsh,lane,sb);swc_tile<3>(a,bsh,lane,sb);swc_tile<4>(a,bsh,lane,sb);
            swc_tile<5>(a,bsh,lane,sb);swc_tile<6>(a,bsh,lane,sb);swc_tile<7>(a,bsh,lane,sb);swc_tile<8>(a,bsh,lane,sb);}
        __syncwarp();
        const u32* hs=st+lane*SST;
        const u64 S=u64(hs[32])+(u64(hs[33])<<16)+(u64(hs[34])<<32);
        const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);
        switch(j){case 0:swc_digit<0,true>(acc,hs,alpha);break;case 1:swc_digit<1,true>(acc,hs,alpha);break;case 2:swc_digit<2,true>(acc,hs,alpha);break;case 3:swc_digit<3,true>(acc,hs,alpha);break;
                  case 4:swc_digit<4,true>(acc,hs,alpha);break;case 5:swc_digit<5,true>(acc,hs,alpha);break;case 6:swc_digit<6,true>(acc,hs,alpha);break;default:swc_digit<7,true>(acc,hs,alpha);break;}
        __syncwarp();   // stage reads of this digit done
    }
    {   // fold x1 = (x mod 2^W) + d*(x >> W), t = x1 - bias, one +-delta chain
        u32 hw[10];
        #pragma unroll
        for(u32 w=0;w<10;++w){const u32 a_=w+WO<NW?acc[w+WO]:0,b_=w+WO+1<NW?acc[w+WO+1]:0;hw[w]=(a_>>WS)|(b_<<(32-WS));}
        acc[WO]&=(1u<<WS)-1;
        swc_fold(acc,hw,d);
        const u32 br=swc_subbias(acc);
        const bool hi=(acc[WO]>>WS)!=0u;
        acc[WO]&=(1u<<WS)-1;
        const u32 lt=swc_ltq(acc);
        swc_addx(acc,br?(0u-d):((hi||!lt)?d:0u),br);
        acc[WO]&=(1u<<WS)-1;
    }
    if(dfl&1)for(u32 x=lane;x<K*8u;x+=32u)bpo_discard(residues+u64(x)*N+i0);
    const u32 cc_=i0>>8,h0=i0&255u;
    {   // coefficient rr: words 0..31 by lanes 0..31, words 32..54 by lanes 0..22 ([base+imm] addresses, no index math)
        __syncwarp();
        #pragma unroll
        for(u32 w=0;w<L;++w)st[lane*L+w]=acc[w];
        __syncwarp();
        u32* const ob=output+u64(h0*256u+cc_)*L+lane;
        if(dfl&2){
            #pragma unroll
            for(u32 rr=0;rr<32;++rr){__stcs(ob+rr*256u*L,st[rr*L+lane]);if(lane<L-32)__stcs(ob+rr*256u*L+32,st[rr*L+32+lane]);}
        }else{
            #pragma unroll
            for(u32 rr=0;rr<32;++rr){ob[rr*256u*L]=st[rr*L+lane];if(lane<L-32)ob[rr*256u*L+32]=st[rr*L+32+lane];}
        }
    }
}
// pdl twin (host: FHERMA_PDL / PDL_DEFAULT): bpo_crt_lean with launch_dependents at block start and griddepcontrol.wait
// before the first D read (only the B fragments and the next-input L2 prefetch run before it). Same output.
extern "C" __global__ void __launch_bounds__(128,3) bpo_crt_leanp(const u32* __restrict__ residues,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){
    asm volatile("griddepcontrol.launch_dependents;":::"memory");   // pdl: the next same-chain P1 may launch
    if(pfitems){
        const u32 total=gridDim.x*blockDim.x;
        for(u32 t=blockIdx.x*blockDim.x+threadIdx.x;t<pfitems*256u;t+=total){
            const u32 it=t>>8,hh=t&255u;const u32* ps=(it>=256u?pfb:pfa)+u64(it&255u)*BCRT_L+u64(hh*256u)*BCRT_L;
            asm volatile("prefetch.global.L2 [%0];"::"l"(ps));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+BCRT_L-1));
        }
    }
    constexpr u32 K=BCRT_K,SW=BCRT_SW,W=BCRT_W,L=BCRT_L,NW=64,NT=9,SST=36,F=BTC_F;
    static_assert(K==16&&SW==16,"tensor CRT geometry");
    constexpr u32 WO=W/32,WS=W%32;
    static_assert(WO+1==L&&WS>0,"q = 2^W - d with W in the top limb");
    constexpr u32 STW=32*L;   // per-warp stage words
    static_assert(32*SST+512<=STW,"prefetch buffer aliases the output stage");
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 stage_all[4][STW];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    u32* st=stage_all[warp];
    const u32 i0=blockIdx.x*128+warp*32;
    u32* const pbuf=st+32*SST;   // [r][32 coefficients], coefficient index ^ 8*(r&3) (conflict-free fragment reads)
    auto prefetch=[&](u32 j){
        #pragma unroll
        for(u32 k=0;k<4;++k){const u32 q=lane+32*k,r=q>>3,ch=q&7;
            swc_cpa16(u32(__cvta_generic_to_shared(pbuf+r*32+((4*ch)^(8*(r&3))))),residues+(u64(r)*8+j)*N+i0+4*ch);}
        swc_commit();};
    for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128)bsh[x]=bfrag[x];
    asm volatile("griddepcontrol.wait;":::"memory");   // pdl: P3 complete (D written and visible) before the first D read
    prefetch(0);
    __syncthreads();
    u32 acc[NW+2];   // [64..65]: scratch of the digit chains
    #pragma unroll
    for(u32 w=0;w<NW+2;++w)acc[w]=0;
    #pragma unroll
    for(u32 j=0;j<8;++j){
        u32 a[2][2][4];
        swc_wait0();__syncwarp();
        {   // pbk[k] = pbuf + 32*tig + gid + 8*(k^tig) words, k = 2m + (row half)
            const u32 pb=u32(__cvta_generic_to_shared(pbuf))+(32*tig+gid)*4;
            const u32 p0=pb+32*(0^tig),p1=pb+32*(1^tig),p2=pb+32*(2^tig),p3=pb+32*(3^tig);
#define SWC_FR(m,s,q0,q1) a[m][s][0]=swc_lds<(8*s)*128>(q0);a[m][s][1]=swc_lds<(8*s)*128>(q1);a[m][s][2]=swc_lds<(8*s+4)*128>(q0);a[m][s][3]=swc_lds<(8*s+4)*128>(q1);
            SWC_FR(0,0,p0,p1) SWC_FR(0,1,p0,p1) SWC_FR(1,0,p2,p3) SWC_FR(1,1,p2,p3)
#undef SWC_FR
        }
        __syncwarp();
        if(j<7)prefetch(j+1);
        {const u32 sb=u32(__cvta_generic_to_shared(st))+(gid*SST+tig)*4;
            swc_tile<0>(a,bsh,lane,sb);swc_tile<1>(a,bsh,lane,sb);swc_tile<2>(a,bsh,lane,sb);swc_tile<3>(a,bsh,lane,sb);swc_tile<4>(a,bsh,lane,sb);
            swc_tile<5>(a,bsh,lane,sb);swc_tile<6>(a,bsh,lane,sb);swc_tile<7>(a,bsh,lane,sb);swc_tile<8>(a,bsh,lane,sb);}
        __syncwarp();
        const u32* hs=st+lane*SST;
        const u64 S=u64(hs[32])+(u64(hs[33])<<16)+(u64(hs[34])<<32);
        const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);
        switch(j){case 0:swc_digit<0,true>(acc,hs,alpha);break;case 1:swc_digit<1,true>(acc,hs,alpha);break;case 2:swc_digit<2,true>(acc,hs,alpha);break;case 3:swc_digit<3,true>(acc,hs,alpha);break;
                  case 4:swc_digit<4,true>(acc,hs,alpha);break;case 5:swc_digit<5,true>(acc,hs,alpha);break;case 6:swc_digit<6,true>(acc,hs,alpha);break;default:swc_digit<7,true>(acc,hs,alpha);break;}
        __syncwarp();   // stage reads of this digit done
    }
    {   // fold x1 = (x mod 2^W) + d*(x >> W), t = x1 - bias, one +-delta chain
        u32 hw[10];
        #pragma unroll
        for(u32 w=0;w<10;++w){const u32 a_=w+WO<NW?acc[w+WO]:0,b_=w+WO+1<NW?acc[w+WO+1]:0;hw[w]=(a_>>WS)|(b_<<(32-WS));}
        acc[WO]&=(1u<<WS)-1;
        swc_fold(acc,hw,d);
        const u32 br=swc_subbias(acc);
        const bool hi=(acc[WO]>>WS)!=0u;
        acc[WO]&=(1u<<WS)-1;
        const u32 lt=swc_ltq(acc);
        swc_addx(acc,br?(0u-d):((hi||!lt)?d:0u),br);
        acc[WO]&=(1u<<WS)-1;
    }
    if(dfl&1)for(u32 x=lane;x<K*8u;x+=32u)bpo_discard(residues+u64(x)*N+i0);
    const u32 cc_=i0>>8,h0=i0&255u;
    {   // coefficient rr: words 0..31 by lanes 0..31, words 32..54 by lanes 0..22 ([base+imm] addresses, no index math)
        __syncwarp();
        #pragma unroll
        for(u32 w=0;w<L;++w)st[lane*L+w]=acc[w];
        __syncwarp();
        u32* const ob=output+u64(h0*256u+cc_)*L+lane;
        if(dfl&2){
            #pragma unroll
            for(u32 rr=0;rr<32;++rr){__stcs(ob+rr*256u*L,st[rr*L+lane]);if(lane<L-32)__stcs(ob+rr*256u*L+32,st[rr*L+32+lane]);}
        }else{
            #pragma unroll
            for(u32 rr=0;rr<32;++rr){ob[rr*256u*L]=st[rr*L+lane];if(lane<L-32)ob[rr*256u*L+32]=st[rr*L+32+lane];}
        }
    }
}
// ---- crtl (FHERMA_CRTL mask, CRTL_DEFAULT): bpo_crt_lean with fewer shared-memory instructions. Same bytes through shared
// memory, same arithmetic (exact by construction: only which lane / word / register holds a value changes).
//  bit 1 FR: A-fragment row (gid, m, h) is coefficient 16m + 2gid + h (was 16m + 8h + gid). The fragment words (h = 0, 1)
//      of one prime row and m-tile are then 2 adjacent coefficients = one LDS.64 straight into the IMMA operand pair
//      (a0,a1) / (a2,a3): 8 LDS.64 instead of 16 LDS per digit (conflict-free under the existing pbuf swizzle: bank
//      ((16m + 2gid) ^ 8tig) + h). The h stage keeps one row per coefficient, row c at c*36 + 4*(c>>3) words: writer bank
//      8gid + 4(gid>>2) + col, reader LDS.128 chunk c + (c>>3) mod 8: both conflict-free.
//  bit 2 ST: tiles 2p, 2p+1 stored as one STS.64 at column x(tig) + 4p, x = 2*(tig&1) + 16*(tig>>1) (bank 4*gid + x:
//      conflict-free per half-warp; with FR: column 2*tig + 8p, bank 8gid + 4(gid>>2) + 2tig): 4 x 4 STS.64 + 4 STS (tile 8) = 20 instead of 36 STS per digit. The reader loads the
//      same 9 LDS.128 and renames registers (compile-time word map swcl_wpos).
//  bit 4 BQ: B fragments relaid [T][lane][s][2] at the block's smem fill: one LDS.128 per tile, 9 instead of 18 LDS.64.
// Per warp per digit: 79 -> 50 shared instructions with all three (smem wavefronts unchanged).
__host__ __device__ constexpr u32 swcl_wpos(bool fr,bool st,u32 k){return (!st||k>=32u)?k:fr?(2u*(k&3u)+8u*((k>>2)>>1)+((k>>2)&1u)):(2u*(k&1u)+16u*((k>>1)&1u)+4u*((k>>2)>>1)+((k>>2)&1u));}
// stage row offset (words) of the C-fragment row (m, h) of row-group 0 (the lane part is in the base)
__host__ __device__ constexpr u32 swcl_row(bool fr,u32 m,u32 h){return fr?((16u*m+h)*36u+8u*m):((16u*m+8u*h)*36u);}
template<u32 OFF> __device__ __forceinline__ uint2 swcl_lds2(u32 base){uint2 v;asm volatile("ld.shared.v2.u32 {%0,%1}, [%2+%3];":"=r"(v.x),"=r"(v.y):"r"(base),"n"(OFF):"memory");return v;}
template<u32 OFF> __device__ __forceinline__ void swcl_sts2(u32 base,u32 v0,u32 v1){asm volatile("st.shared.v2.u32 [%0+%1], {%2,%3};"::"r"(base),"n"(OFF),"r"(v0),"r"(v1):"memory");}
template<u32 OFF> __device__ __forceinline__ uint4 swcl_lds4(u32 base){uint4 v;asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4+%5];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(base),"n"(OFF):"memory");return v;}
// digit J from register h (AIN form of swc_digit: alpha*NP + OFF folded into the column sums)
template<u32 J> __device__ __forceinline__ void swcl_digit(u32 (&acc)[66],const u32 (&h)[32],u32 alpha){
    constexpr u32 o=SWC_POS[J]/32,sh=SWC_POS[J]%32;
    u32 xe[16],xo[16];
    #pragma unroll
    for(u32 i=0;i<16;++i){xe[i]=h[2*i]+alpha*swc_np16(2*i)+swc_of16(2*i);xo[i]=h[2*i+1]+alpha*swc_np16(2*i+1)+swc_of16(2*i+1);}
    swc_mac<(1u<<sh)>(acc+o,xe);
    if(sh<16)swc_mac<(1u<<((sh+16)&31))>(acc+o,xo);else swc_mac<(1u<<((sh+16)&31))>(acc+o+1,xo);
    i64 c=0;
    #pragma unroll
    for(u32 w=16;w<=18;++w){
        const u32 ex=w==16?(alpha<<sh):(w==17&&sh?(alpha>>((32-sh)&31)):0u);
        const i64 t=i64(acc[o+w])-i64(ex)+c;acc[o+w]=u32(t);c=t>>32;}
}
template<u32 FL,u32 T> __device__ __forceinline__ void swcl_b(const u32* bsh,u32 lane,uint2& b0,uint2& b1){
    if(FL&4u){const uint4 v=*reinterpret_cast<const uint4*>(bsh+(T*32+lane)*4);b0=make_uint2(v.x,v.y);b1=make_uint2(v.z,v.w);}
    else{b0=*reinterpret_cast<const uint2*>(bsh+((T*2+0)*32+lane)*2);b1=*reinterpret_cast<const uint2*>(bsh+((T*2+1)*32+lane)*2);}
}
template<u32 M> __device__ __forceinline__ void swcl_mm(const u32 (&a)[2][2][4],uint2 b0,uint2 b1,u32 (&c)[4]){
    c[0]=c[1]=c[2]=c[3]=0;
    imma16832(c,a[M][0][0],a[M][0][1],a[M][0][2],a[M][0][3],b0.x,b0.y);imma16832(c,a[M][1][0],a[M][1][1],a[M][1][2],a[M][1][3],b1.x,b1.y);
}
// one tile T, 32-bit stores (column 4T + tig in the base)
template<u32 FL,u32 T> __device__ __forceinline__ void swcl_tile1(const u32 (&a)[2][2][4],const u32* bsh,u32 lane,u32 sb){
    constexpr bool FR=(FL&1u)!=0;uint2 b0,b1;swcl_b<FL,T>(bsh,lane,b0,b1);u32 c[4];
    swcl_mm<0>(a,b0,b1,c);swc_sts<(swcl_row(FR,0,0)+4*T)*4>(sb,c[0]+(c[1]<<8));swc_sts<(swcl_row(FR,0,1)+4*T)*4>(sb,c[2]+(c[3]<<8));
    swcl_mm<1>(a,b0,b1,c);swc_sts<(swcl_row(FR,1,0)+4*T)*4>(sb,c[0]+(c[1]<<8));swc_sts<(swcl_row(FR,1,1)+4*T)*4>(sb,c[2]+(c[3]<<8));
}
// tiles 2P, 2P+1 as STS.64 (column x(tig) in the base, + 4P (FR: 8P))
template<u32 FL,u32 P> __device__ __forceinline__ void swcl_tile2(const u32 (&a)[2][2][4],const u32* bsh,u32 lane,u32 sx){
    constexpr bool FR=(FL&1u)!=0;constexpr u32 PO=FR?8*P:4*P;
    uint2 b0,b1,b2,b3;swcl_b<FL,2*P>(bsh,lane,b0,b1);swcl_b<FL,2*P+1>(bsh,lane,b2,b3);u32 c0[4],c1[4];
    swcl_mm<0>(a,b0,b1,c0);swcl_mm<0>(a,b2,b3,c1);
    swcl_sts2<(swcl_row(FR,0,0)+PO)*4>(sx,c0[0]+(c0[1]<<8),c1[0]+(c1[1]<<8));swcl_sts2<(swcl_row(FR,0,1)+PO)*4>(sx,c0[2]+(c0[3]<<8),c1[2]+(c1[3]<<8));
    swcl_mm<1>(a,b0,b1,c0);swcl_mm<1>(a,b2,b3,c1);
    swcl_sts2<(swcl_row(FR,1,0)+PO)*4>(sx,c0[0]+(c0[1]<<8),c1[0]+(c1[1]<<8));swcl_sts2<(swcl_row(FR,1,1)+PO)*4>(sx,c0[2]+(c0[3]<<8),c1[2]+(c1[3]<<8));
}
template<u32 FL> __device__ __forceinline__ void swcl_body(const u32* __restrict__ residues,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){
    if(pfitems){
        const u32 total=gridDim.x*blockDim.x;
        for(u32 t=blockIdx.x*blockDim.x+threadIdx.x;t<pfitems*256u;t+=total){
            const u32 it=t>>8,hh=t&255u;const u32* ps=(it>=256u?pfb:pfa)+u64(it&255u)*BCRT_L+u64(hh*256u)*BCRT_L;
            asm volatile("prefetch.global.L2 [%0];"::"l"(ps));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+BCRT_L-1));
        }
    }
    constexpr bool FR=(FL&1u)!=0,ST=(FL&2u)!=0,BQ=(FL&4u)!=0;
    constexpr u32 K=BCRT_K,SW=BCRT_SW,W=BCRT_W,L=BCRT_L,NW=64,NT=9,SST=36,F=BTC_F;
    static_assert(K==16&&SW==16&&NT==9,"tensor CRT geometry");
    constexpr u32 WO=W/32,WS=W%32;
    static_assert(WO+1==L&&WS>0,"q = 2^W - d with W in the top limb");
    constexpr u32 STW=32*L;                         // per-warp stage words
    constexpr u32 SGW=FR?32*SST+12:32*SST;          // h-stage words (FR: 4 words of pad after every 8 rows); prefetch buffer after it
    static_assert(SGW+512<=STW&&SGW%4==0,"prefetch buffer aliases the output stage");
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 stage_all[4][STW];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    u32* st=stage_all[warp];
    const u32 i0=blockIdx.x*128+warp*32;
    u32* const pbuf=st+SGW;   // [r][32 coefficients], coefficient index ^ 8*(r&3) (conflict-free fragment reads)
    auto prefetch=[&](u32 j){
        #pragma unroll
        for(u32 k=0;k<4;++k){const u32 q=lane+32*k,r=q>>3,ch=q&7;
            swc_cpa16(u32(__cvta_generic_to_shared(pbuf+r*32+((4*ch)^(8*(r&3))))),residues+(u64(r)*8+j)*N+i0+4*ch);}
        swc_commit();};
    prefetch(0);
    if(BQ){for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128){const u32 w=x&1u,ln=(x>>1)&31u,s=(x>>6)&1u,T=x>>7;bsh[((T*32+ln)*2+s)*2+w]=bfrag[x];}}
    else{for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128)bsh[x]=bfrag[x];}
    __syncthreads();
    u32 acc[NW+2];   // [64..65]: scratch of the digit chains
    #pragma unroll
    for(u32 w=0;w<NW+2;++w)acc[w]=0;
    const u32 stb=u32(__cvta_generic_to_shared(st));
    #pragma unroll
    for(u32 j=0;j<8;++j){
        u32 a[2][2][4];
        swc_wait0();__syncwarp();
        if(FR){   // prime row r = tig + 4*k4 (s = k4>>1, regs 2*(k4&1) + h); m-tile m: coefficients 16m + 2gid + (0, 1) = one LDS.64
            const u32 pq=u32(__cvta_generic_to_shared(pbuf))+tig*128u;
            const u32 q0=pq+(((2u*gid)^(8u*tig))*4u),q1=pq+(((16u+2u*gid)^(8u*tig))*4u);
#define SWCL_FR(m,q) {uint2 v=swcl_lds2<0>(q);a[m][0][0]=v.x;a[m][0][1]=v.y;v=swcl_lds2<512>(q);a[m][0][2]=v.x;a[m][0][3]=v.y;                       v=swcl_lds2<1024>(q);a[m][1][0]=v.x;a[m][1][1]=v.y;v=swcl_lds2<1536>(q);a[m][1][2]=v.x;a[m][1][3]=v.y;}
            SWCL_FR(0,q0) SWCL_FR(1,q1)
#undef SWCL_FR
        }else{   // pbk[k] = pbuf + 32*tig + gid + 8*(k^tig) words, k = 2m + (row half)
            const u32 pb=u32(__cvta_generic_to_shared(pbuf))+(32*tig+gid)*4;
            const u32 p0=pb+32*(0^tig),p1=pb+32*(1^tig),p2=pb+32*(2^tig),p3=pb+32*(3^tig);
#define SWC_FR(m,s,q0,q1) a[m][s][0]=swc_lds<(8*s)*128>(q0);a[m][s][1]=swc_lds<(8*s)*128>(q1);a[m][s][2]=swc_lds<(8*s+4)*128>(q0);a[m][s][3]=swc_lds<(8*s+4)*128>(q1);
            SWC_FR(0,0,p0,p1) SWC_FR(0,1,p0,p1) SWC_FR(1,0,p2,p3) SWC_FR(1,1,p2,p3)
#undef SWC_FR
        }
        __syncwarp();
        if(j<7)prefetch(j+1);
        {const u32 sr=stb+(FR?(2u*gid*SST+4u*(gid>>2)):gid*SST)*4u;   // row-group 0 row of this lane's gid
            if(ST){const u32 sx=sr+(FR?2u*tig:(2u*(tig&1u)+16u*(tig>>1)))*4u;
                swcl_tile2<FL,0>(a,bsh,lane,sx);swcl_tile2<FL,1>(a,bsh,lane,sx);swcl_tile2<FL,2>(a,bsh,lane,sx);swcl_tile2<FL,3>(a,bsh,lane,sx);
                swcl_tile1<FL,8>(a,bsh,lane,sr+tig*4u);}
            else{const u32 sb=sr+tig*4u;
                swcl_tile1<FL,0>(a,bsh,lane,sb);swcl_tile1<FL,1>(a,bsh,lane,sb);swcl_tile1<FL,2>(a,bsh,lane,sb);swcl_tile1<FL,3>(a,bsh,lane,sb);swcl_tile1<FL,4>(a,bsh,lane,sb);
                swcl_tile1<FL,5>(a,bsh,lane,sb);swcl_tile1<FL,6>(a,bsh,lane,sb);swcl_tile1<FL,7>(a,bsh,lane,sb);swcl_tile1<FL,8>(a,bsh,lane,sb);}}
        __syncwarp();
        const u32* hs=st+lane*SST+(FR?4u*(lane>>3):0u);   // this lane's coefficient row
        u32 v[36];
        #pragma unroll
        for(u32 c4=0;c4<9;++c4){const uint4 q=reinterpret_cast<const uint4*>(hs)[c4];v[4*c4]=q.x;v[4*c4+1]=q.y;v[4*c4+2]=q.z;v[4*c4+3]=q.w;}
        u32 h[32];
        #pragma unroll
        for(u32 k=0;k<32;++k)h[k]=v[swcl_wpos(FR,ST,k)];
        const u64 S=u64(v[32])+(u64(v[33])<<16)+(u64(v[34])<<32);
        const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);
        switch(j){case 0:swcl_digit<0>(acc,h,alpha);break;case 1:swcl_digit<1>(acc,h,alpha);break;case 2:swcl_digit<2>(acc,h,alpha);break;case 3:swcl_digit<3>(acc,h,alpha);break;
                  case 4:swcl_digit<4>(acc,h,alpha);break;case 5:swcl_digit<5>(acc,h,alpha);break;case 6:swcl_digit<6>(acc,h,alpha);break;default:swcl_digit<7>(acc,h,alpha);break;}
        __syncwarp();   // stage reads of this digit done
    }
    {   // fold x1 = (x mod 2^W) + d*(x >> W), t = x1 - bias, one +-delta chain
        u32 hw[10];
        #pragma unroll
        for(u32 w=0;w<10;++w){const u32 a_=w+WO<NW?acc[w+WO]:0,b_=w+WO+1<NW?acc[w+WO+1]:0;hw[w]=(a_>>WS)|(b_<<(32-WS));}
        acc[WO]&=(1u<<WS)-1;
        swc_fold(acc,hw,d);
        const u32 br=swc_subbias(acc);
        const bool hi=(acc[WO]>>WS)!=0u;
        acc[WO]&=(1u<<WS)-1;
        const u32 lt=swc_ltq(acc);
        swc_addx(acc,br?(0u-d):((hi||!lt)?d:0u),br);
        acc[WO]&=(1u<<WS)-1;
    }
    if(dfl&1)for(u32 x=lane;x<K*8u;x+=32u)bpo_discard(residues+u64(x)*N+i0);
    const u32 cc_=i0>>8,h0=i0&255u;   // lane owns coefficient i0 + lane in every variant
    {   // coefficient rr: words 0..31 by lanes 0..31, words 32..54 by lanes 0..22 ([base+imm] addresses, no index math)
        __syncwarp();
        #pragma unroll
        for(u32 w=0;w<L;++w)st[lane*L+w]=acc[w];
        __syncwarp();
        u32* const ob=output+u64(h0*256u+cc_)*L+lane;
        if(dfl&2){
            #pragma unroll
            for(u32 rr=0;rr<32;++rr){__stcs(ob+rr*256u*L,st[rr*L+lane]);if(lane<L-32)__stcs(ob+rr*256u*L+32,st[rr*L+32+lane]);}
        }else{
            #pragma unroll
            for(u32 rr=0;rr<32;++rr){ob[rr*256u*L]=st[rr*L+lane];if(lane<L-32)ob[rr*256u*L+32]=st[rr*L+32+lane];}
        }
    }
}
#define SWCL_K(nm,fl) extern "C" __global__ void __launch_bounds__(128,3) nm(const u32* __restrict__ residues,u32* __restrict__ output, const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){swcl_body<fl>(residues,output,bfrag,N,d,pfa,pfb,pfitems,dfl);}
SWCL_K(bpo_crt_l1,1u)
SWCL_K(bpo_crt_l2,2u)
SWCL_K(bpo_crt_l4,4u)
SWCL_K(bpo_crt_l7,7u)
#undef SWCL_K
#endif
