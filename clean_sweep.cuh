// ---- Round 24 (p1ws, exit-time combo sweep + FHERMA_P1WS forced mode only; appended to the sweep copy of the source, never to
// the production one) ----
// Warp-specialized P1 (exact, same output as bpo_first_x): one persistent block per SM (grid = SM count, column-op items
// id = blockIdx.x + gridDim.x t), 8 producer warps + CL_WS_NC consumer warps, dynamic smem CL_P1WS_SMEM (two 40960 B quad slots
// in the production skewed X layout, one 64-row input staging quarter, 4 mbarriers).
//   producers (warps 0-7, af = 64 live registers each, as production): input staging by 64-row quarters (cp.async, evict_first
//     as production) + fragment build for the item, L2 prefetch of the next item; per quad: wait slot empty, IMMA base
//     conversion + Barrett reduction (production formulas) and raw residues (in [0,2p)) stored straight into the slot at the
//     production phase-A addresses, arrive full.
//   consumers (no af): wait full; phase A (production thread mapping: prime tig, rows 64rr+8w+gid: twist, J-DIF, R-DIF len
//     256/128) in place; consumer-only named barrier; phase B = production xrdif6q (one prime, 4 digits per warp, TMA burst);
//     lane 0 waits the bulk reads of the slot, every consumer thread arrives empty.
//   CL_WS HC: Harvey-lazy consumer arithmetic on the quads whose 4 primes are < 2^30 (h30: quads 0-2), Shoup on quad 3
//     (exact: the h30 prime set has 12 primes < 2^30 first; final cross twiddle canonical). HC false: Shoup everywhere.
// CL_WS NC 16 (p1ws16): two consumer teams of 8 warps, team t serves slot t (768 threads: 80-register cap, producers spill).
// setmaxnreg: rejected for the production arch (compute_90 PTX JIT; ptxas: "not supported on .target sm_120", only sm_120a)
// -> uniform __launch_bounds__(32 (8 + NC), 1) cap; with NC 8 every warp gets 128 registers.
#ifndef CL_SWEEP
#define CL_SWEEP 1
#define CL_P1_SIG(NAME) NAME(const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,\
    u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB)
#define CL_P1WS_SMEM 97312u   // 2 x 40960 slots + 64 x 60 x 4 staging + 4 x 8 mbarriers (host copy in combo_sweep.h)
__device__ __forceinline__ void clw_init(u32 a,u32 n){asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"r"(a),"r"(n):"memory");}
__device__ __forceinline__ void clw_arrive(u32 a){asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(a):"memory");}
__device__ __forceinline__ void clw_wait(u32 a,u32 par){u32 ok;
    do{asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0,1,0,P;}":"=r"(ok):"r"(a),"r"(par):"memory");}while(!ok);}
__device__ __forceinline__ void clw_bar(u32 id,u32 n){asm volatile("bar.sync %0, %1;"::"r"(id),"r"(n):"memory");}
#if defined(__CUDA_ARCH__)&&__CUDA_ARCH__>=900
// consumer quad: phase A in place (prime 4u+tig, rows 64rr+rl), consumer barrier, phase B (xrdif6q)
template<bool H,u32 KC=0>   // KC (timing ablation): 1 no phase A, 2 no phase B
__device__ __forceinline__ void clw_quad(u32* const Xs,const u32 u,const u32 c,const u32 op,const u32 cw,const u32 team,const u32 lane,const u64* twist,
    const u64* stage_tw,const u64* ctab,const u32 stride,const XTMap* tmA,const XTMap* tmB,const u32 stm){
    constexpr u32 F=BPO_XF,XD=320u,XS=8u*XD;
    const u32 gid=lane>>2,tig=lane&3u;
    if constexpr(KC!=1u){   const u32 rl=8u*cw+gid,r=4u*u+tig,p=bpr_pr[r];
        u32 p2=0,n2=0;if(H){p2=2u*p;n2=0u-2u*p;asm volatile("":"+r"(p2),"+r"(n2));}   // opaque: one VIADDMNMX per correction
        const u64* const tw=twist+size_t(r)*stride;const u64* const st=stage_tw+size_t(r)*256u;
        u32* const xa=Xs+tig*XS+8u*tig+rl+4u*(rl>>5);
        u32 res[4][8];
        #pragma unroll
        for(u32 rr=0;rr<4;++rr)
            #pragma unroll
            for(u32 j=0;j<8;++j)res[rr][j]=xa[j*XD+72u*rr];
        #pragma unroll
        for(u32 rr=0;rr<4;++rr){const u64 t2=tw[1u+128u+(((64u*rr+rl)*256u+c)>>7)];
            #pragma unroll
            for(u32 j=0;j<8;++j)res[rr][j]=xmshh<H>(res[rr][j],t2,p);
            xjdif<F,H>(res[rr],st,p,p2,n2);}
        const u64 wa=st[rl],wb=st[64u+rl],wc=st[128u+rl];
        #pragma unroll
        for(u32 j=0;j<8;++j){xbfh<H>(res[0][j],res[2][j],wa,p,p2,n2);xbfh<H>(res[1][j],res[3][j],wb,p,p2,n2);xbfh<H>(res[0][j],res[1][j],wc,p,p2,n2);xbfh<H>(res[2][j],res[3][j],wc,p,p2,n2);}
        #pragma unroll
        for(u32 rr=0;rr<4;++rr)
            #pragma unroll
            for(u32 j=0;j<8;++j)xa[j*XD+72u*rr]=res[rr][j];
    }
    clw_bar(2u+team,256u);   // every phase A of this slot (this team) done
    if constexpr(KC!=2u){   const u32 q=cw>>1,r=4u*u+q,p=bpr_pr[r];
        u32 p2=0,n2=0;if(H){p2=2u*p;n2=0u-2u*p;asm volatile("":"+r"(p2),"+r"(n2));}
        xrdif6q<F,H>(Xs+q*XS+4u*(cw&1u)*XD,8u*q,stage_tw+size_t(r)*256u,ctab+(size_t(r)*256u+c)*256u,p,lane,op?tmB:tmA,c,r*8u+4u*(cw&1u),stm,p2,n2);
    }
}
#endif
template<u32 NC,bool HC,u32 K=0>   // K (timing ablations, wrong values): 1 consumers idle, 2 producers idle, 3 producers no prologue,
    // 4 / 5 producers idle and consumers without phase A / phase B
__device__ __forceinline__ void cl_p1ws_body(const u32* a,const u32* b,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,
    u32 N,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const XTMap* tmA,const XTMap* tmB){
#if defined(__CUDA_ARCH__)&&__CUDA_ARCH__>=900
    static_assert(NC==8u||NC==16u,"consumer teams of 8 warps (one phase-A item per thread, one xrdif6q unit per warp); NC 16: team t serves slot t");
    constexpr u32 L=BCRT_L;constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    constexpr u32 XD=320u,XS=8u*XD,SW=4u*XS,XRS=60u,NP=8u;
    static_assert(4u*(2u*SW+64u*XRS)+32u==CL_P1WS_SMEM,"smem layout");
    static_assert(3u+POS[7]/32u+8u<XRS,"fragment windows inside the staged row");
    extern __shared__ __align__(128) u32 clw_sm[];
    u32* const STG=clw_sm+2u*SW;
    const u32 mb=u32(__cvta_generic_to_shared(clw_sm+2u*SW+64u*XRS));   // full[0] +0, full[1] +8, empty[0] +16, empty[1] +24
    const u32 stm=(pfstep>>29)&3u;const bool ldn=(pfstep>>28)&1u;const bool pfon=(pfstep&0x0fffffffu)!=0u;
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,gid=lane>>2,tig=lane&3u;
    const u32 stride=weighted_root_stride(N),G=gridDim.x,u0=r0/4u,u1=r1/4u;
    if(h==0){clw_init(mb,32u*NP);clw_init(mb+8u,32u*NP);clw_init(mb+16u,256u);clw_init(mb+24u,256u);}
    __syncthreads();
    if(warp<NP){   // ---------------- producers
        u32 af[8][2][4];
        const u32 rl=8u*warp+gid;
        u64 pol=0;if(!ldn)asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
        u32 g=0;
        #pragma unroll 1
        for(u32 id=blockIdx.x;id<512u;id+=G){
            const u32 c=id&255u,op=id>>8;
            const u32* const src=(op?b:a)+u64(c)*L;
            const u32 xoff=(c*L)&3u;
            const bool ca=(reinterpret_cast<unsigned long long>(op?b:a)&15ull)==0;
            #pragma unroll
            for(u32 k=0;k<4;++k){   // quarter k = rows 64k..64k+63 -> af[j][k>>1][(k&1)+2xh]
                if constexpr(K==2u||K==3u||K>=4u)continue;
                clw_bar(1u,32u*NP);   // staging free (previous quarter's / item's fragment reads done)
                if(ca){const u32 q=h&15u,nch=(xoff+L+3u)/4u;
                    if(q<nch){const u32* const g0=src-xoff+4u*q+u64((64u*k+(h>>4))*256u)*L;const u32 sa0=u32(__cvta_generic_to_shared(STG+(h>>4)*XRS+4u*q));
                        #pragma unroll
                        for(u32 i=0;i<4;++i){const u32 sa=sa0+i*(16u*XRS*4u);const u32* const gg=g0+i*(16u*256u*L);
                            if(ldn)asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(sa),"l"(gg):"memory");
                            else asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"r"(sa),"l"(gg),"l"(pol):"memory");}}
                    asm volatile("cp.async.commit_group;":::"memory");
                    asm volatile("cp.async.wait_group 0;":::"memory");
                }else{
                    #pragma unroll 4
                    for(u32 x=h;x<64u*L;x+=32u*NP){const u32 rr=x/L,w=x-rr*L;STG[rr*XRS+xoff+w]=__ldcs(src+u64((64u*k+rr)*256u)*L+w);}
                }
                clw_bar(1u,32u*NP);
                const u32* const rpt=STG+rl*XRS+xoff+tig;
                #pragma unroll
                for(u32 j=0;j<8;++j){
                    const u32 wb=POS[j]/32,sh=POS[j]%32;const int D=int(POS[j+1])-int(POS[j]);
                    #pragma unroll
                    for(u32 xh=0;xh<2;++xh){const u32* const rp=rpt+wb+4u*xh;const u32 v=__funnelshift_r(rp[0],rp[1],sh);
                        if(D-32*(3+4*int(xh))>=32)af[j][k>>1][(k&1u)+2u*xh]=v;else af[j][k>>1][(k&1u)+2u*xh]=v&bpr_fmask.v[j][xh][tig];}}
            }
            if(pfon&&id+G<512u){const u32 nx=id+G;const u32* const ps=(nx>=256u?b:a)+u64(nx&255u)*L+u64(h*256u)*L;   // next item's 256 rows -> L2
                asm volatile("prefetch.global.L2 [%0];"::"l"(ps));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+L-1));}
            #pragma unroll 1
            for(u32 u=u0;u<u1;++u,++g){
                const u32 s=g&1u;
                const u32 r=4u*u+tig,p=bpr_pr[r];const u32 msh=(BPO_MIXED&&p<(1u<<30))?3u:2u;const u32 m62=u32(bpr_mu[r]>>msh),qsh=18u-msh;
                const u32 np=bpr_np.v[r];
                clw_wait(mb+16u+8u*s,((g>>1)&1u)^1u);   // slot empty (consumers done with quad g-2)
                u32* const xa=clw_sm+s*SW+tig*XS+8u*tig+rl+4u*(rl>>5);
                #pragma unroll
                for(u32 j=0;j<8;++j){
                    if constexpr(K==2u||K>=4u)continue;
                    const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+u);
                    #pragma unroll
                    for(u32 mt=0;mt<2;++mt){
                        u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
                        pimma16832(c0,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.x,bv.y);
                        pimma16832(c1,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.z,bv.w);
                        #pragma unroll
                        for(u32 hf=0;hf<2;++hf){
                            const u32 y0=c0[2*hf],y1=c0[2*hf+1],y2=c1[2*hf],y3=c1[2*hf+1];
                            const u32 w=y0+(y1<<8),z=y2+(y3<<8);
                            const u32 lo=w+(z<<16),hi=(w>>14)+(z<<2);
                            const u32 q=__umulhi(hi,m62)>>qsh;
                            xa[j*XD+72u*(2u*mt+hf)]=lo+q*np;}
                    }
                }
                clw_arrive(mb+8u*s);
            }
        }
    }else{   // ---------------- consumers
        const u32 cw=(warp-NP)&7u,team=NC==16u?(warp-NP)>>3:0u;
        u32 g=0;
        #pragma unroll 1
        for(u32 id=blockIdx.x;id<512u;id+=G){
            const u32 c=id&255u,op=id>>8;
            #pragma unroll 1
            for(u32 u=u0;u<u1;++u,++g){
                const u32 s=g&1u;
                if(NC==16u&&s!=team)continue;
                clw_wait(mb+8u*s,(g>>1)&1u);
                u32* const Xs=clw_sm+s*SW;
                if constexpr(K==1u){}else
                if(HC&&BPO_NH30!=0&&4u*u+4u<=u32(BPO_NH30))clw_quad<true,(K>=4u?K-3u:0u)>(Xs,u,c,op,cw,team,lane,twist,stage_tw,ctab,stride,tmA,tmB,stm);
                else clw_quad<false,(K>=4u?K-3u:0u)>(Xs,u,c,op,cw,team,lane,twist,stage_tw,ctab,stride,tmA,tmB,stm);
                if(lane==0)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");   // TMA reads of the slot done
                __syncwarp();
                clw_arrive(mb+16u+8u*s);
            }
        }
        if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
    }
#else
    (void)a;(void)b;(void)twist;(void)stage_tw;(void)bpt;(void)N;(void)r0;(void)r1;(void)pfstep;(void)ctab;(void)tmA;(void)tmB;
#endif
}
#define CL_WS(NAME,NC,HC,K) extern "C" __global__ void __launch_bounds__(32*(8+(NC)),1) CL_P1_SIG(NAME){(void)A;(void)B;(void)logR;\
    cl_p1ws_body<(NC),(HC),(K)>(a,b,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB);}
CL_WS(cl_p1ws,8u,true,0u)
CL_WS(cl_p1ws_s,8u,false,0u)
CL_WS(cl_p1ws16,16u,true,0u)
CL_WS(cl_p1ws_kpa,8u,true,4u)
CL_WS(cl_p1ws_kpb,8u,true,5u)
CL_WS(cl_p1ws_kc,8u,true,1u)
CL_WS(cl_p1ws_kp,8u,true,2u)
#endif
