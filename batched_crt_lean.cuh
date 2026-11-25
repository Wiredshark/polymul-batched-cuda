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
#ifdef BPO_TWC   // ovl compute twin: no next-product L2 prefetch, block 0's coefficients
    if(false){
#else
    if(pfitems){
#endif
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
#ifdef BPO_TWC   // ovl compute twin: 32 coefficient blocks only (2 MB, L2-resident); stores predicated off (dfl bit 30)
    const u32 i0=(blockIdx.x&31u)*128+warp*32;
#define CRTST if(dfl&0x40000000u)
#else
#define CRTST
    const u32 i0=blockIdx.x*128+warp*32;
#endif
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
        CRTST if(dfl&2){
            #pragma unroll
            for(u32 rr=0;rr<32;++rr){__stcs(ob+rr*256u*L,st[rr*L+lane]);if(lane<L-32)__stcs(ob+rr*256u*L+32,st[rr*L+32+lane]);}
        }else{
            #pragma unroll
            for(u32 rr=0;rr<32;++rr){ob[rr*256u*L]=st[rr*L+lane];if(lane<L-32)ob[rr*256u*L+32]=st[rr*L+32+lane];}
        }
    }
#undef CRTST
}
#ifdef BPO_TWM
#ifndef BPO_MA
#define BPO_MA 0
#endif
// memd sink: keeps loads live when the stores are off (flag bit 30 is never set)
#define MA_SINK(x,ptr,flags) if(((x)==0x9e3779b9u)&&((flags)&0x40000000u))(ptr)[0]=(x)
// ---- ovl memory-only twins (production signatures, grids, blocks): the production global loads and stores (addresses, widths,
// cache operators, order, tables) with the arithmetic removed; loaded words XOR-folded into the stored words.
extern "C" __global__ void __launch_bounds__(128) wl4_p2m(const u32* A,const u32* B,u32* C,const u32* primes,const u64* r16,u32 N,u32 untw){
    const u32 g=blockIdx.x,j=blockIdx.y,prime=blockIdx.z;(void)primes;(void)N;
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5,s=lane&7u,c2=4u*warp+(lane>>3);
    const u64* tf=r16+u64(prime)*1024u;const u64* const tmf=tf+(512u+c2);const u64* const tmi=tf+(768u+c2);
#ifdef BPO_P2CU   // production p2cu: stage twiddles from the constant table
    const u64* const tsf=bpo_p2cu+((untw>>24)+prime)*256u;const u64* const tsi=tsf+128u;
#else
    const u64* const tsf=tf;const u64* const tsi=tf+256u;
#endif
    const u32 gl=(BPO_MA&1)?(g&15u):g,gs=(BPO_MA&4)?(g&15u):g;   // memd: load / store tile windows
    const u64 base=u64(prime*8+((BPO_MA&1)?0u:j))*65536u,sbase=u64(prime*8+((BPO_MA&4)?0u:j))*65536u;const u32 off=gl*2048u+c2*8u+s;
    u32 x=0;
    {const u32* a0=A+base+off;const u32* b0=B+base+off;
        #pragma unroll
        for(u32 c1=0;c1<16;++c1){if(untw&4){x^=__ldcs(a0+128u*c1)+c1;x^=__ldcs(b0+128u*c1);}else{x^=a0[128u*c1]+c1;x^=b0[128u*c1];}}}
    u64 w=0;
    if(!(BPO_MA&8)){
    #pragma unroll
    for(u32 e=1;e<8;++e)w^=tsf[16u*e];   // forward stage twiddles (uniform)
    #pragma unroll
    for(u32 i=1;i<16;++i)w^=__ldg(tmf+16u*i);}   // middle twiddles
    x+=__syncthreads_or(x==0x9e3779b9u);   // the barrier consumes the loads (as R: discard after use)
    if(untw&8)bpo_discard((threadIdx.x<64u?A:B)+base+gl*2048u+(threadIdx.x&63u)*32u);
    if(!(BPO_MA&8)){
    #pragma unroll
    for(u32 e=1;e<8;++e)w^=tsi[16u*e];   // inverse stage twiddles
    #pragma unroll
    for(u32 i=1;i<16;++i)w^=__ldg(tmi+16u*i);}
    x^=u32(w)^u32(w>>32);
    u32* o=C+sbase+(c2*256u+gs*8u+s);
    if(BPO_MA&2){MA_SINK(x,o,untw);return;}
    #pragma unroll
    for(u32 c1=0;c1<16;++c1)o[4096u*c1]=x+c1;
}
extern "C" __global__ void __launch_bounds__(256) wl4_p3m(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){
    constexpr u32 NN=65536u;(void)primes;(void)N;(void)logR;
    const u32 c=blockIdx.x,prime=blockIdx.y,t=threadIdx.x,lane=t&31u,warp=t>>5,l3=lane&3u;
    u32 x=0;
    {const u32* src=C+u64(prime*8u)*NN+(c*256u+t);
        #pragma unroll
        for(u32 d=0;d<8;++d)x^=src[d*NN]+d;}
    u64 w=__ldg(untwist+(u64(prime)*65536u+(c*256u+t)));
    const u64* sit=stage_itw+prime*256u;
    w^=sit[32]^sit[64]^sit[96]^sit[253];   // J-DIT / first R-DIT twiddles (uniform)
    x+=__syncthreads_or(x==0x9e3779b9u);   // the barrier consumes x: the C loads land before the discard below (as in R)
    const u64* const tn=t2n+(u64(prime*8u+warp)*512u+((c>>7)*256u+lane));
    u64 tv=0;
    #pragma unroll
    for(u32 m=0;m<8;++m)tv^=__ldg(tn+32u*m)+m;
    if((dfl&1)&&lane<8u)bpo_discard(C+u64(prime*8u+lane)*NN+(c*256u+32u*warp));
    w^=sit[248u+l3]^sit[240u+l3]^sit[244u+l3]^sit[224u+l3]^sit[228u+l3]^sit[232u+l3]^sit[236u+l3];   // R-DIT lane twiddles
    w^=sit[192u+lane]^sit[128u+lane]^sit[160u+lane]^sit[lane]^sit[32u+lane]^sit[64u+lane]^sit[96u+lane];
    x^=u32(w)^u32(w>>32)^u32(tv)^u32(tv>>32);
    u32* const out=D+(u64(prime*8u+warp)*NN+(c*256u+lane));
#ifdef BPO_P3EL
    const u32 pm=12u;
#else
    const u32 pm=4u;
#endif
    if(dfl&pm){
        u64 pol;if(dfl&pm&8u)asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
        #pragma unroll
        for(u32 m=0;m<8;++m){const u32 r=x+m;asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;"::"l"(out+32u*m),"r"(r),"l"(pol):"memory");}
    }else{
        #pragma unroll
        for(u32 m=0;m<8;++m)out[32u*m]=x+m;
    }
}
#ifdef BTC_F
extern "C" __global__ void __launch_bounds__(128,3) wl4_crtm(const u32* __restrict__ residues,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){
    (void)d;
    if(pfitems&&!(BPO_MA&16)){
        const u32 total=gridDim.x*blockDim.x;
        for(u32 t=blockIdx.x*blockDim.x+threadIdx.x;t<pfitems*256u;t+=total){
            const u32 it=t>>8,hh=t&255u;const u32* ps=(it>=256u?pfb:pfa)+u64(it&255u)*BCRT_L+u64(hh*256u)*BCRT_L;
            asm volatile("prefetch.global.L2 [%0];"::"l"(ps));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+BCRT_L-1));
        }
    }
    constexpr u32 K=BCRT_K,L=BCRT_L,NT=9,SST=36,STW=32*L;
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 stage_all[4][STW];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31;
    u32* st=stage_all[warp];
    const u32 i0=((BPO_MA&1)?(blockIdx.x&31u):blockIdx.x)*128+warp*32,is0=((BPO_MA&4)?(blockIdx.x&31u):blockIdx.x)*128+warp*32;   // memd: load / store windows
    u32* const pbuf=st+32*SST;
    auto prefetch=[&](u32 j){
        #pragma unroll
        for(u32 k=0;k<4;++k){const u32 q=lane+32*k,r=q>>3,ch=q&7;
            swc_cpa16(u32(__cvta_generic_to_shared(pbuf+r*32+((4*ch)^(8*(r&3))))),residues+(u64(r)*8+j)*N+i0+4*ch);}
        swc_commit();};
    prefetch(0);
    for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128)bsh[x]=(BPO_MA&8)?x:bfrag[x];
    __syncthreads();
    u32 acc=bsh[lane];
    #pragma unroll
    for(u32 j=0;j<8;++j){
        swc_wait0();__syncwarp();
        #pragma unroll
        for(u32 k=0;k<16;++k)acc^=pbuf[k*32+lane];   // this digit's 16 residue rows (prefetch buffer)
        __syncwarp();
        if(j<7)prefetch(j+1);
        __syncwarp();
    }
    if(dfl&1)for(u32 x=lane;x<K*8u;x+=32u)bpo_discard(residues+u64(x)*N+i0);
    const u32 cc_=is0>>8,h0=is0&255u;
    if(BPO_MA&2){MA_SINK(acc,output+is0,dfl);return;}
    {   __syncwarp();
        #pragma unroll
        for(u32 w=0;w<L;++w)st[lane*L+w]=acc+w;
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
#endif
#endif
#endif
