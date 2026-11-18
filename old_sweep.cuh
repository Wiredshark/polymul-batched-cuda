// Sweep-only kernels (appended to the sweep copy of the NVRTC source by diag_sweep.h, never to the production module).
// Round 7 (oldideas): locally rejected options re-built on the current kernels for the 128 MB-L2 target.
// * bpo_first_xq (from p1occ 1427a86, the tc3 low-register idea 62dda64 on bpo_first_x): one prime quad per block
//   (grid y = quads; the kernel loops u += gridDim.y), A fragments built per digit from the staged input halves, dense
//   XOR-swizzled 32 KB X, <= 80 registers, 3 blocks/SM. Same values as bpo_first_xb.
// * sw_r16db (P2 double buffer 0890374 on bpo_middle_r16): a block runs two digit tiles j0, j0+1 (grid y = 4); the
//   second tile's A/B loads are issued before the first tile's transforms. Same values as bpo_middle_r16.
#if defined(BPT_ENABLE)&&defined(BPR_DT)
// Flag 131072 (with 1|2|4096|8192|65536 semantics, TMA only): one quad per block (grid y = quads of the launch; the kernel
// loops u += gridDim.y, so any grid y is correct). Input rows staged in two 128-row halves (row stride 60 words) and the
// A fragments built from smem per digit right before their two MMAs: no 64-register af array lives across the quad
// (<= 80 registers). X holds the quad densely (32 column regions of 256 words, 32 KB, rows XOR-swizzled by xsw instead of
// the 25% skew padding), so 3 blocks/SM fit the 100 KB of an sm_120 SM (24 warps instead of 16). Same values as bpo_first_xb.
#define BPO_XQ_SMEM (32*256*4)
// row -> word in its dense 256-word column region of prime q: bits 3-4 ^= (row>>6)^q, bit 2 ^= bit 5. Conflict-free for the
// phase-A stores (lanes: 8 rows x 4 primes), phase-B layout 1 (rows l7+8m+64g) and layout 2 (uint4 pairs of rows 8 lane..+7).
__device__ __forceinline__ u32 xsw(u32 row,u32 q){return row^((((row>>6)^q)&3u)<<3)^(((row>>5)&1u)<<2);}
template<u32 F,u32 NC>
__device__ __forceinline__ void xrdif6z(u32* const base0,const u32 q,const u64* st,const u64* ct,const u32 p,u32 lane,const XTMap* tm,u32 c,const u32 plane0,u32 stm){
    constexpr bool T=(F&1)!=0;
    constexpr u32 XD=256u;
    u32 v[NC][8];
    const u32 l7=lane&7u,g=lane>>3;
    {u32 o1[8];
        #pragma unroll
        for(u32 m=0;m<8;++m)o1[m]=xsw(l7+8*m+64*g,q);
        #pragma unroll
        for(u32 n=0;n<NC;++n)
            #pragma unroll
            for(u32 m=0;m<8;++m)v[n][m]=base0[n*XD+o1[m]];
        u64 w64[4],w32[2];
        #pragma unroll
        for(u32 m=0;m<4;++m)w64[m]=st[192u+l7+8u*m];
        #pragma unroll
        for(u32 qq=0;qq<2;++qq)w32[qq]=st[224u+l7+8u*qq];
        const u64 w16=st[240u+l7];
        #pragma unroll
        for(u32 m=0;m<4;++m)
            #pragma unroll
            for(u32 n=0;n<NC;++n)xbf(v[n][m],v[n][m+4],w64[m],p);   // len 64: j = row&31
        #pragma unroll
        for(u32 qq=0;qq<2;++qq)
            #pragma unroll
            for(u32 n=0;n<NC;++n)
                #pragma unroll
                for(u32 hh=0;hh<2;++hh)xbf(v[n][qq+4*hh],v[n][qq+4*hh+2],w32[qq],p);   // len 32: j = row&15
        #pragma unroll
        for(u32 n=0;n<NC;++n)
            #pragma unroll
            for(u32 m=0;m<8;m+=2)xbf(v[n][m],v[n][m+1],w16,p);   // len 16: j = row&7
        #pragma unroll
        for(u32 n=0;n<NC;++n)
            #pragma unroll
            for(u32 m=0;m<8;++m)base0[n*XD+o1[m]]=v[n][m];
    }
    __syncwarp();
    {const u32 o2=xsw(8*lane,q);   // rows 8 lane..+3 at o2, rows +4..+7 at o2^4
        #pragma unroll
        for(u32 n=0;n<NC;++n){const uint4 x0=*reinterpret_cast<const uint4*>(base0+n*XD+o2),x1=*reinterpret_cast<const uint4*>(base0+n*XD+(o2^4u));
            v[n][0]=x0.x;v[n][1]=x0.y;v[n][2]=x0.z;v[n][3]=x0.w;v[n][4]=x1.x;v[n][5]=x1.y;v[n][6]=x1.z;v[n][7]=x1.w;}}
    {u64 w8[4];
        #pragma unroll
        for(u32 lo=0;lo<4;++lo)w8[lo]=st[248u+lo];
        const u64 w253=st[253u],w252=st[252u],w254=st[254u];
        #pragma unroll
        for(u32 lo=0;lo<4;++lo)
            #pragma unroll
            for(u32 n=0;n<NC;++n){if(lo==0)xbft<T>(v[n][0],v[n][4],w8[0],p);else xbf(v[n][lo],v[n][lo+4],w8[lo],p);}   // len 8 (st[248]=psi^0)
        #pragma unroll
        for(u32 k=0;k<4;++k)
            #pragma unroll
            for(u32 n=0;n<NC;++n){const u32 lo=(k&1)+4*(k>>1);if(lo&1)xbf(v[n][lo],v[n][lo+2],w253,p);else xbft<T>(v[n][lo],v[n][lo+2],w252,p);}
        if(!T){
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2)
            #pragma unroll
            for(u32 n=0;n<NC;++n)xbf(v[n][lo],v[n][lo+1],w254,p);}}
    {const ulonglong2* c2=reinterpret_cast<const ulonglong2*>(ct+8*lane);
        #pragma unroll
        for(u32 m2=0;m2<4;++m2){const ulonglong2 w=__ldg(c2+m2);
            #pragma unroll
            for(u32 n=0;n<NC;++n){const u32 a=v[n][2*m2],b=v[n][2*m2+1];
                if(T){v[n][2*m2]=xmsh(a+b,u64(w.x),p);v[n][2*m2+1]=xmsh(a+p-b,u64(w.y),p);}   // len-2 psi^0 stage fused into the cross twiddle
                else{v[n][2*m2]=xmsh(a,u64(w.x),p);v[n][2*m2+1]=xmsh(b,u64(w.y),p);}}}}
#if BPO_X_HAS_TMA
    __syncwarp();
    #pragma unroll
    for(u32 n=0;n<NC;++n){uint4* const d0=reinterpret_cast<uint4*>(base0+n*XD+8*lane);   // dense slot order (TMA box)
        d0[0]=make_uint4(v[n][0],v[n][1],v[n][2],v[n][3]);d0[1]=make_uint4(v[n][4],v[n][5],v[n][6],v[n][7]);}
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncwarp();
    if(lane==0&&BPO_XA!=8&&BPO_XA!=11&&BPO_XA!=12){
        u64 pol=0;const bool hint=(F&2048)&&stm!=1u;
        if(hint){if(stm==2u)asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));}
        #pragma unroll
        for(u32 n=0;n<NC;++n){const u32 sa=u32(__cvta_generic_to_shared(base0+n*XD));
            if(hint)asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%1,%2,%3}],[%4],%5;"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane0+n),"r"(sa),"l"(pol):"memory");
            else asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane0+n),"r"(sa):"memory");}
        asm volatile("cp.async.bulk.commit_group;":::"memory");}
#endif
}
template<u32 F>
__device__ __forceinline__ void bpo_first_xq(const u32* a,const u32* b,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const XTMap* tmA,const XTMap* tmB,u32* X){
    constexpr u32 L=BCRT_L;
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    constexpr u32 XD=256u,XS=8*256,XRS=60;
    static_assert(128*XRS<=BPO_XQ_SMEM/4&&3u+POS[7]/32u+8u<XRS,"flag 131072 smem layout");
    const u32 stm=(pfstep>>29)&3u;const bool ldn=(pfstep>>28)&1u;pfstep&=0x0fffffffu;
    const u32 id=blockIdx.x,c=id&255u,op=id>>8;
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,gid=lane>>2,tig=lane&3;
    const u32 cc=BPO_XA==11?0u:c;   // ablation 11: compute proxy (column-0 input, L2-resident; no stores)
    const u32* const src=(op?b:a)+u64(cc)*L;
    const u32 xoff=(cc*L)&3u;
    const bool ca=(reinterpret_cast<unsigned long long>(op?b:a)&15ull)==0;
    if(blockIdx.y==0&&pfstep&&id+pfstep<512u){
        const u32 pf=id+pfstep;const u32* ps=(pf>=256u?b:a)+u64(pf&255u)*L+u64(h*256u)*L;
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+L-1));
    }
    const u32 stride=weighted_root_stride(N);
    const u32 rl=8*warp+gid;
    // fragment word masks: words t = tig+4 (x>>1) of a digit of width D: full, partial or 0
    auto fmask=[&](int D)->u32{const int wd=D-32*int(tig)-128;return wd>=32?~0u:(wd<=0?0u:((1u<<wd)-1u));};
    #pragma unroll 1
    for(u32 u=r0/4+blockIdx.y;u<r1/4;u+=gridDim.y){
        const bool last=u+gridDim.y>=r1/4;   // last quad of this column in the launch: the input is not re-read
        const u32 r=4*u+tig,p=bpr_pr[r];const u32 m62=u32(bpr_mu[r]>>2);
        const u32 np=bpr_np.v[r];
        const u64* tw=twist+size_t(r)*stride;const u64* st=stage_tw+size_t(r)*256;
        u32 res[4][8];   // [row 2mt+hf][digit], unreduced in [0,2p)
        #pragma unroll
        for(u32 mt=0;mt<2;++mt){
            __syncthreads();   // previous half's fragment reads (mt 1) / previous quad's phase-B and TMA reads of X (mt 0) done
            const u32* const hs=src+u64(mt*128u*256u)*L;
            if(BPO_XA==12){}else   // ablation 12: no input staging (compute proxy minus the input loads)
            if(ca){   // all 16 B chunks of the half in flight at once
                const u32 q=h&15u,nch=(xoff+L+3u)/4u;
                u64 pol=0;if(!ldn&&last)asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
                if(q<nch){
                    const u32 sa0=u32(__cvta_generic_to_shared(X+(h>>4)*XRS+4u*q));const u32* const gq=hs-xoff+4u*q+u64((h>>4)*256u)*L;
                    #pragma unroll
                    for(u32 i=0;i<8;++i){
                        const u32 sa=sa0+i*(16u*XRS*4u);const u32* g=gq+i*(16u*256u*L);
                        if(ldn||!last)asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(sa),"l"(g):"memory");
                        else asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"r"(sa),"l"(g),"l"(pol):"memory");}}
                asm volatile("cp.async.commit_group;":::"memory");
                asm volatile("cp.async.wait_group 0;":::"memory");
            }else
            #pragma unroll 4
            for(u32 x=h;x<128u*L;x+=256u){const u32 rr=x/L,w=x-rr*L;X[rr*XRS+xoff+w]=__ldg(hs+u64(rr*256u)*L+w);}
            __syncthreads();
            const u32* const rpt=X+rl*XRS+xoff+tig;
            #pragma unroll
            for(u32 j=0;j<8;++j){
                const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+u);
                const u32 wb=POS[j]/32,sh=POS[j]%32;
                const int D=int(POS[j+1])-int(POS[j]);
                u32 af[4];
                #pragma unroll
                for(u32 x=0;x<4;++x){
                    const u32* rp=rpt+64*(x&1)*XRS+wb+4*(x>>1);
                    const u32 v=__funnelshift_r(rp[0],rp[1],sh);
                    af[x]=(D-32*(3+4*int(x>>1))>=32)?v:(v&fmask(D));
                }
                u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
                pimma16832(c0,af[0],af[1],af[2],af[3],bv.x,bv.y);
                pimma16832(c1,af[0],af[1],af[2],af[3],bv.z,bv.w);
                #pragma unroll
                for(u32 hf=0;hf<2;++hf){
                    const u32 y0=c0[2*hf],y1=c0[2*hf+1],y2=c1[2*hf],y3=c1[2*hf+1];   // each < 2^21
                    const u32 w=y0+(y1<<8),z=y2+(y3<<8);   // x = w + 2^16 z: hi = floor(x/2^14) exactly, r = lo + q(-p)
                    const u32 lo=w+(z<<16),hi=(w>>14)+(z<<2);
                    const u32 qv=__umulhi(hi,m62)>>16;
                    res[2*mt+hf][j]=lo+qv*np;
                }
            }
        }
        #pragma unroll
        for(u32 rr=0;rr<4;++rr){   // rows 64rr+rl: twist, J-DIF, R-DIF stages len 256 (rr^2) and 128 (rr^1) in registers
            const u64 t2=tw[1+128u+(((64*rr+rl)*256u+c)>>7)];
            #pragma unroll
            for(u32 j=0;j<8;++j)res[rr][j]=xmsh(res[rr][j],t2,p);
            xjdif<F>(res[rr],st,p);
        }
        {const u64 wa=st[rl],wb=st[64u+rl],wc=st[128u+rl];
        #pragma unroll
        for(u32 j=0;j<8;++j){xbf(res[0][j],res[2][j],wa,p);xbf(res[1][j],res[3][j],wb,p);xbf(res[0][j],res[1][j],wc,p);xbf(res[2][j],res[3][j],wc,p);}}
        __syncthreads();   // fragment reads of the staged input done
        #pragma unroll
        for(u32 rr=0;rr<4;++rr){u32* const xa=X+tig*XS+xsw(64*rr+rl,tig);
            #pragma unroll
            for(u32 j=0;j<8;++j)xa[j*XD]=res[rr][j];}
        __syncthreads();
        if(BPO_XA!=3){   // warp w: prime q = w>>1, digits 4(w&1)..+3 as NC-column groups sharing twiddles
            constexpr u32 NC=BPO_X_NC;
            const u32 q=warp>>1,r2=4*u+q,p2=bpr_pr[r2];
            const u64* const st2=stage_tw+size_t(r2)*256;const u64* const ct=ctab+(size_t(r2)*256u+c)*256u;
            #pragma unroll
            for(u32 g=0;g<4/NC;++g){const u32 j=4*(warp&1u)+NC*g;
                xrdif6z<F,NC>(X+q*XS+j*XD,q,st2,ct,p2,lane,op?tmB:tmA,c,r2*8+j,stm);}
        }
#if BPO_X_HAS_TMA
        if(lane==0&&!last)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");   // X reusable
#endif
    }
#if BPO_X_HAS_TMA
    if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
#endif
}
#ifndef BPO_XQ_MINB
#define BPO_XQ_MINB 3
#endif
#define SWXQ(NAME,F,MB) extern "C" __global__ void __launch_bounds__(256,MB) NAME(const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){__shared__ __align__(1024) u32 XQ[BPO_XQ_SMEM/4];bpo_first_xq<(F)>(a,b,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,XQ);}
#if BPO_X_HAS_TMA
SWXQ(sw_xq,BPO_XF,BPO_XQ_MINB)
SWXQ(sw_xq2,BPO_XF,2)
#endif
#endif // BPT_ENABLE
extern "C" __global__ void __launch_bounds__(128) sw_r16db(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 N,u32 untw){
    __shared__ u32 TA[8*292];
    const u32 g=blockIdx.x,prime=blockIdx.z,p=primes[prime];
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5,s=lane&7u,c2=4u*warp+(lane>>3);
    const u64* tf=r16+u64(prime)*512u;const u64* ti=tf+256u;
    u32 na[16],nb[16];
    auto ld=[&](u32 j){const u64 base=u64(prime*8+j)*N;const u32* a0=A+base+g*2048u+c2*8u+s;const u32* b0=B+base+g*2048u+c2*8u+s;
        #pragma unroll
        for(u32 c1=0;c1<16;++c1){if(untw&4){na[c1]=__ldcs(a0+128u*c1);nb[c1]=__ldcs(b0+128u*c1);}else{na[c1]=a0[128u*c1];nb[c1]=b0[128u*c1];}}};
    ld(2u*blockIdx.y);
    #pragma unroll 1
    for(u32 t=0;t<2;++t){
        const u32 j=2u*blockIdx.y+t;const u64 base=u64(prime*8+j)*N;
        u32 a[16],b[16];
        #pragma unroll
        for(u32 i=0;i<16;++i){a[i]=na[i];b[i]=nb[i];}
        if(t==0)ld(j+1u);   // second tile's loads in flight during the first tile's transforms
        r16_dif16(a,tf,p);r16_dif16(b,tf,p);
        u32* const ta=TA+s*292u;
        #pragma unroll
        for(u32 i=1;i<16;++i){const u64 w=__ldg(tf+((c2*r16_brev4(i))&255u));a[i]=pxmsh(a[i],w,p);b[i]=pxmsh(b[i],w,p);}
        if(t)__syncthreads();   // previous tile's inverse reads of TA done
        #pragma unroll
        for(u32 i=0;i<16;++i)ta[17*i+c2]=a[i];
        __syncthreads();
        if(untw&8)bpo_discard((threadIdx.x<64u?A:B)+base+g*2048u+(threadIdx.x&63u)*32u);
        #pragma unroll
        for(u32 i=0;i<16;++i)a[i]=ta[17*c2+i];
        __syncthreads();
        #pragma unroll
        for(u32 i=0;i<16;++i)ta[17*i+c2]=b[i];
        __syncthreads();
        #pragma unroll
        for(u32 i=0;i<16;++i)b[i]=ta[17*c2+i];
        r16_dif16(a,tf,p);r16_dif16(b,tf,p);
        {const u32 pinv=pxinv32(p);
            #pragma unroll
            for(u32 i=0;i<16;++i)a[i]=pxmont(a[i],b[i],p,pinv);}
        r16_dit16<true>(a,ti,p);
        {const u32 k1=r16_brev4(c2);
            #pragma unroll
            for(u32 i=1;i<16;++i)a[i]=pxmsh(a[i],__ldg(ti+((i*k1)&255u)),p);
            a[0]=pzred(a[0],p);}
        __syncthreads();
        #pragma unroll
        for(u32 i=0;i<16;++i)ta[17*c2+i]=a[i];
        __syncthreads();
        #pragma unroll
        for(u32 i=0;i<16;++i)a[i]=ta[17*i+c2];
        r16_dit16<true>(a,ti,p);
        u32* o=C+base+c2*256u+g*8u+s;
        #pragma unroll
        for(u32 c1=0;c1<16;++c1)o[4096u*c1]=a[c1];
    }
}
