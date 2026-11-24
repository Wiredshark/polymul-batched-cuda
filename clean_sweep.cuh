// ---- Round 24 (crt2, exit-time combo sweep only; appended to the sweep copy of the source, never to the production one) ----
// CRT restructure lines on 3140af3 + crtx (production PTX unchanged): kernels instantiated from the opt-in bodies of
// batched_crt_pipe.cuh (bcp_body / bcq_body; production module never instantiates them), same signature, grid, block and
// arguments as bpo_crt_lean, all exact (device compare vs the production graph):
//   cx_pipe5  bcp_body<5>: paired-tile IMMA (4 independent accumulator chains), B fragments one LDS.128 per tile, digit j+1
//             tile phase software-pipelined under digit j's carry chains (h row held in registers); 168 regs, 3 blocks/SM
//   cx_pipe7  bcp_body<7>: as pipe5 without the pipeline (production digit order): isolates the pipeline
//   cx_q4a    bcq_body<1>: 4 blocks/SM CRT (128 regs, 23 KB smem): no cp.async residue buffer (A fragments by LDG.64 into
//             registers one digit ahead, MMA rows permuted so a fragment = 2 consecutive coefficients), paired tiles,
//             output staged in two 16-coefficient halves
//   cx_q4b    bcq_body<3>: as q4a with the next-digit fragment loads issued after the tile phase (lower live registers)
#ifndef CL_SWEEP
#define CL_SWEEP 1
#ifdef BTC_F
#define CX_CRT(NAME,BODY,FLV,MINB) extern "C" __global__ void __launch_bounds__(128,MINB) NAME(const u32* __restrict__ residues,\
    u32* __restrict__ output,const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){\
    BODY<FLV>(residues,output,bfrag,N,d,pfa,pfb,pfitems,dfl);}
CX_CRT(cx_pipe5,bcp_body,5u,3)
CX_CRT(cx_pipe7,bcp_body,7u,3)
CX_CRT(cx_q4a,bcq_body,1u,4)
CX_CRT(cx_q4b,bcq_body,3u,4)
#undef CX_CRT
#endif
#endif
