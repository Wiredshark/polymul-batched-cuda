// bpo_p3crt: P3 (J-DIT + R-DIT + t2n, same arithmetic as bpo_last_cm3 with BPO_P3T 260) fused into the tensor CRT
// (same arithmetic as bpo_crt_tc_cm): no residue round trip, one launch instead of P3 launches + CRT.
// Cluster of 2 CTAs (128 threads) per column c: CTA rank k runs P3 for primes 8k..8k+7 into its smem and the CRT
// for rows h in [128k,128k+128), reading the other CTA's residues through DSMEM (rank = the CRT k-slice s).
// 8/JC passes of JC digits j (smem for 3 CTAs/SM): each pass rereads the C column (L2) and redoes the cheap J-DIT.
#if defined(BTC_F)&&defined(BPT_ENABLE)&&defined(__CUDA_ARCH__)&&__CUDA_ARCH__>=900&&(!defined(BPO_LEAN)||BPO_LEAN_P3CRT)
#ifndef BPC_JC
#define BPC_JC 2u
#endif
#define BPC_RS (256u*BPC_JC+8u)   // residue row stride per local prime (words): +8 spreads tig over banks
__device__ __forceinline__ u32 bpc_ld(u32 a){u32 v;asm volatile("ld.shared::cluster.u32 %0,[%1];":"=r"(v):"r"(a):"memory");return v;}
__device__ __forceinline__ void bpc_cluster_sync(){asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;":::"memory");}
extern "C" __global__ void __cluster_dims__(2,1,1) __launch_bounds__(128,3) bpo_p3crt(const u32* __restrict__ C,u32* __restrict__ output,
    const u32* __restrict__ primes,const u64* __restrict__ untwist,const u64* __restrict__ stage_itw,const u64* __restrict__ t2n,
    const u32* __restrict__ bfrag,u32 N,u32 logR,u32 d){
    constexpr u32 K=BCRT_K,SW=BCRT_SW,W=BCRT_W,L=BCRT_L,NW=64,NT=9,SST=36,F=BTC_F,PB=4u/BPC_JC;
    static_assert(K==16&&SW==16,"tensor CRT geometry");
    constexpr u32 POS[8]={BCRT_POS};
    constexpr u32 QW[L]={BCRT_Q},BW[L]={BCRT_BIAS};
    constexpr u32 NPW[SW]={BTC_NP},OFW[SW]={BTC_OFF};
    constexpr u32 RW=8u*BPC_RS,XW=4u*384u,HL=(L+1)/2,SGW=4u*32u*HL;   // residues | X tiles; aliased by the output stage
    static_assert(XW>=4u*288u&&XW>=4u*32u*12u,"P3 tiles and CRT exchange chunks fit in X");
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 sm[(RW+XW)>SGW?(RW+XW):SGW];
    u32* const R=sm;u32* const X=sm+RW;
    u32 rank;asm("mov.u32 %0,%%cluster_ctarank;":"=r"(rank));
    const u32 c=blockIdx.x>>1,tid=threadIdx.x,warp=tid>>5,lane=tid&31,gid=lane>>2,tig=lane&3;
    for(u32 x=tid;x<NT*2*32*2;x+=128)bsh[x]=bfrag[x];
    u32 rbase[2];{const u32 la=u32(__cvta_generic_to_shared(R));
        #pragma unroll
        for(u32 s=0;s<2;++s)asm("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(rbase[s]):"r"(la),"r"(s));}
    const u32 hw0=128u*rank+32u*warp;   // CRT rows of this warp (column-local h)
    u32 acc[NW];
    #pragma unroll
    for(u32 w=0;w<NW;++w)acc[w]=0;
    u32 pend=0;
    #pragma unroll
    for(u32 ps=0;ps<8u/BPC_JC;++ps){
        // ---- P3 for primes 8 rank .. +7, digits j = ps*JC .. +JC-1, into R[rl][jj][h]
        #pragma unroll 1
        for(u32 rb=0;rb<8u;rb+=PB){
            #pragma unroll
            for(u32 pi=0;pi<PB;++pi){
                const u32 prime=8u*rank+rb+pi,p=primes[prime];
                const u64* itw=untwist+u64(prime)*weighted_root_stride(N);const u64* sit=stage_itw+prime*256u;
                #pragma unroll
                for(u32 hf=0;hf<2;++hf){const u32 t=tid+128u*hf;
                    const u32* src=C+u64(prime*8)*N+c*256u+t;
                    u32 yv[8];
                    #pragma unroll
                    for(u32 dd=0;dd<8;++dd)yv[dd]=src[u64(dd)*N];
                    const u32 k1=__brev(t)>>(32-logR);
                    const u32 wv=mul_shoup(u32(bm_root(itw,2u*c*k1,N,p)),itw[1+(c&127u)],p);
                    const u64 m0=itw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
                    const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;
                    const u64 Fw=u64(wv)|(u64(q)<<32);
                    #pragma unroll
                    for(u32 dd=0;dd<8;++dd)yv[dd]=pxmsh(yv[dd],Fw,p);
                    #pragma unroll
                    for(u32 d0=0;d0<8;d0+=2)pzdit1c(yv[d0],yv[d0+1],p);
                    #pragma unroll
                    for(u32 len=4;len<=8;len<<=1){const u32 half=len/2;
                        #pragma unroll
                        for(u32 d0=0;d0<8;d0+=len){
                            #pragma unroll
                            for(u32 jj=0;jj<half;++jj){if(jj==0)pzdit1(yv[d0],yv[d0+half],sit[0],p);else pzdit(yv[d0+jj],yv[d0+jj+half],sit[jj*(256u/len)],p);}}}
                    #pragma unroll
                    for(u32 jj=0;jj<BPC_JC;++jj)X[(pi*BPC_JC+jj)*288u+t+4*(t>>5)]=yv[ps*BPC_JC+jj];
                }
            }
            __syncthreads();
            {   // R-DIT: warp = tile (pi,jj)
                const u32 pi=warp/BPC_JC,jj=warp%BPC_JC,prime=8u*rank+rb+pi,p=primes[prime],j=ps*BPC_JC+jj;
                const u64* sit=stage_itw+prime*256u;
                u32* base=X+warp*288u;u32 v[8];
                {const uint4* p4=reinterpret_cast<const uint4*>(base+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
                 v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
                pzdit_smr<false>(base,v,sit,p,lane);
                const u64* tn=t2n+u64(prime*8+j)*512u+(c>>7)*256u+lane;
                u32* o=R+(rb+pi)*BPC_RS+jj*256u+lane;
                #pragma unroll
                for(u32 m=0;m<8;++m)o[32*m]=pxmsh(v[m],tn[32*m],p);
            }
            __syncthreads();   // X reusable
        }
        bpc_cluster_sync();   // both CTAs' residues of this pass visible
        // ---- CRT digits j of this pass (bpo_crt_tc_cm arithmetic)
        #pragma unroll
        for(u32 jj=0;jj<BPC_JC;++jj){const u32 j=ps*BPC_JC+jj;
            u32 a[2][2][4];
            #pragma unroll
            for(u32 m=0;m<2;++m)
                #pragma unroll
                for(u32 s=0;s<2;++s){
                    const u32 ad=rbase[s]+4u*(tig*BPC_RS+jj*256u+hw0+16*m+gid);
                    a[m][s][0]=bpc_ld(ad);a[m][s][1]=bpc_ld(ad+32u);
                    a[m][s][2]=bpc_ld(ad+16u*BPC_RS);a[m][s][3]=bpc_ld(ad+16u*BPC_RS+32u);
                }
            u32 h[SST];
            {   // per-warp exchange in X (free during the CRT phase), 3 chunks of 3 n-tiles: 32 rows x 12 words (conflict-free)
                u32* sx=X+warp*384u;
                #pragma unroll
                for(u32 g=0;g<3;++g){
                    #pragma unroll
                    for(u32 T=3*g;T<3*g+3;++T){
                        const uint2 b0=*reinterpret_cast<const uint2*>(bsh+((T*2+0)*32+lane)*2);
                        const uint2 b1=*reinterpret_cast<const uint2*>(bsh+((T*2+1)*32+lane)*2);
                        #pragma unroll
                        for(u32 m=0;m<2;++m){
                            u32 cc[4]={0,0,0,0};
                            imma16832(cc,a[m][0][0],a[m][0][1],a[m][0][2],a[m][0][3],b0.x,b0.y);
                            imma16832(cc,a[m][1][0],a[m][1][1],a[m][1][2],a[m][1][3],b1.x,b1.y);
                            sx[(16*m+gid)*12u+4*(T-3*g)+tig]=cc[0]+(cc[1]<<8);
                            sx[(16*m+gid+8)*12u+4*(T-3*g)+tig]=cc[2]+(cc[3]<<8);
                        }
                    }
                    __syncwarp();
                    #pragma unroll
                    for(u32 q=0;q<3;++q){const uint4 v=*reinterpret_cast<const uint4*>(sx+lane*12u+4*q);h[12*g+4*q]=v.x;h[12*g+4*q+1]=v.y;h[12*g+4*q+2]=v.z;h[12*g+4*q+3]=v.w;}
                    __syncwarp();
                }
            }
            const u64 S=u64(h[32])+(u64(h[33])<<16)+(u64(h[34])<<32);
            const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);
            const u32 o=POS[j]/32,sh=POS[j]%32;
            const u32 inj=j?(POS[j-1]/32+SW+1-o):0xffffffffu;
            u64 zc=0,cy=0;u32 prevZ=0;
            #pragma unroll
            for(u32 w=0;w<=SW;++w){
                u32 z=0;
                if(w<SW){const u64 v=u64(h[2*w])+(u64(h[2*w+1])<<16)+u64(alpha)*NPW[w]+OFW[w]+zc;z=u32(v);zc=v>>32;}
                const u32 part=sh?((z<<sh)|(prevZ>>(32-sh))):z;
                prevZ=z;
                if(o+w<NW){const u64 s=u64(acc[o+w])+part+cy+(w==inj?pend:0u);acc[o+w]=u32(s);cy=s>>32;}
            }
            pend=u32(cy);
        }
        bpc_cluster_sync();   // remote reads of this pass done before R is rewritten (and before the output staging)
    }
    {
        constexpr u32 start=POS[7]/32+SW+1;u64 cy=pend;
        #pragma unroll
        for(u32 w=0;w<NW;++w)if(w>=start){const u64 s=u64(acc[w])+cy;acc[w]=u32(s);cy=s>>32;}
    }
    constexpr u32 WO=W/32,WS=W%32;
    #pragma unroll
    for(u32 f=0;f<2;++f){
        u64 cy=0;
        #pragma unroll
        for(u32 w=0;w<NW;++w){
            const u32 a_=w+WO<NW?acc[w+WO]:0,b_=w+WO+1<NW?acc[w+WO+1]:0;
            const u32 hw=WS?((a_>>WS)|(b_<<(32-WS))):a_;
            const u32 low=w<WO?acc[w]:(w==WO&&WS?(acc[w]&((1u<<WS)-1)):0);
            const u64 s=u64(low)+u64(hw)*d+cy;acc[w]=u32(s);cy=s>>32;
        }
    }
    u32 borrow=0;
    #pragma unroll
    for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-QW[w]-borrow;borrow=u32(s>>63);}
    if(!borrow){
        borrow=0;
        #pragma unroll
        for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-QW[w]-borrow;acc[w]=u32(s);borrow=u32(s>>63);}
    }
    borrow=0;
    #pragma unroll
    for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-BW[w]-borrow;acc[w]=u32(s);borrow=u32(s>>63);}
    if(borrow){
        u64 cy=0;
        #pragma unroll
        for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])+QW[w]+cy;acc[w]=u32(s);cy=s>>32;}
    }
    // output rows h = hw0+lane of column c, staged in two word halves (stage aliases R|X after the last cluster sync)
    u32* stg=sm+warp*(32u*HL);
    #pragma unroll
    for(u32 hp=0;hp<2;++hp){constexpr u32 dummy=0;(void)dummy;const u32 w0=hp*HL,nw=hp?L-HL:HL;
        __syncwarp();
        #pragma unroll
        for(u32 w=0;w<HL;++w)if(w<nw)stg[lane*nw+w]=acc[w0+w];
        __syncwarp();
        for(u32 x=lane;x<32*nw;x+=32){const u32 rr=x/nw,w=x-rr*nw;output[u64((hw0+rr)*256u+c)*L+w0+w]=stg[x];}
    }
}
#endif // bpo_p3crt
