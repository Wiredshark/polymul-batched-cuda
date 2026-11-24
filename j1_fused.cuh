// J=1 fused role-dispatch kernels (FHERMA_J1_FUSE=1).  One persistent block of 1024 threads (32 warps, 64 registers)
// per SM: warps [0, T) run the tensor role, warps [T, 32) the integer role; the roles never synchronise with each other
// (named barriers only inside a role: 1 = tensor role, 2 + g = integer group g).  Each role walks its own work items
// (item = block + j * grid) and its warps exit when done; a role with a null pointer is idle (its warps exit at once).
//   j1_fuse_x : tensor = forward conversion of a and b of product i+1 (fwd body of j1_conv_tc.cuh; tables refilled per
//               (operand, prime group)); integer = row of product i (warp items, prime-major) then colinv of product i
//               (4-warp groups; a colinv item of prime k waits for the 128 row items of k: per-prime counters `done`,
//               zeroed before the launch; every block finishes its row items before its first colinv item, so the waits
//               cannot deadlock; colinv loads with ld.global.cg because other SMs wrote the data in this launch).
//   j1_fuse_y : tensor = back conversion of product i-1 (back body, exactly 16 tensor warps); integer = colfwd of a and b
//               of product i (4-warp groups).
// Dynamic shared memory: X: J1F_SMEM (tensor table area, J1F_QG quads) + J1X_I * 2688 (row: 2176 B tile + 512 B
// twiddles per integer warp; a colinv group's 8.5 KB tile lies inside its own 4 warps' row areas).
// Y: J1B_SMEM + 4 x 8704 (colfwd group tiles); Y2 (back2 tensor role): J1C_SMEM + 4 x 8704.  Optional clock64 records
// rec[block][warp] = (start, end).
// Prime ranges (group mode, FHERMA_J1_GROUP): the tensor forward role converts prime groups [g0, g0 + ngr) (groups of
// 4 J1F_QG primes); the integer roles work on primes [p0, p0 + np); residue pointers are bases for global prime
// indices (the host shifts a group buffer's base by -p0 rows); X's colinv writes Yo (whole-set mode: Yo = A).
// j1_fuse_y2: tensor role = back2 range body (wt-j1conv 5f0a0cc; block b serves range b & 3; table slice resident in
// shared memory, filled per launch); j1_back2_fin runs as a separate small launch after it.
#ifndef J1_FUSED_CUH
#define J1_FUSED_CUH
#ifndef J1X_T
#define J1X_T 16                       // tensor warps of X (compile-time split; integer warps = 32 - J1X_T, a multiple of 4)
#endif
#define J1X_I (32 - J1X_T)
#ifndef J1X_F
#define J1X_F 3                        // forward body of X's tensor role: 3 = j1_fwd3 (wt-j1conv bc485ca), 1 = j1_fwd (4e0ea38)
#endif
#define J1X_ROWB 2688u
#define J1X_SMEM (J1F_SMEM + J1X_I * 2688)
#define J1Y_SMEM (J1B_SMEM + 4 * 8704)
#define J1Y2_SMEM (J1C_SMEM + 4 * 8704)
__device__ __forceinline__ u32 j1x_acquire(const u32* p){u32 v;asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ void j1x_rec(unsigned long long* rec,u32 w,u32 lane,unsigned long long t0){
    if(rec&&lane==0){rec[(size_t(blockIdx.x)*32u+w)*2u]=t0;rec[(size_t(blockIdx.x)*32u+w)*2u+1u]=clock64();}}
extern "C" __global__ void __launch_bounds__(1024,1) j1_fuse_x(
    const u32* __restrict__ xa,const u32* __restrict__ xb,u32* fra,u32* frb,const uint4* __restrict__ tabA,const uint4* __restrict__ tabB,
    const uint2* __restrict__ pm,u32 g0,u32 ngr,u32 p0,u32 np,u32* Yo,
    u32* A,const u32* Bv,const u32* primes,const u32* pinvs,const u32* thi,const u64* tlo,const u64* tw1,u32* done,
    unsigned long long* rec){
    extern __shared__ __align__(16) unsigned char j1_sm[];
    const u32 w=threadIdx.x>>5,lane=threadIdx.x&31u;const unsigned long long t0=clock64();
    if(w<J1X_T){
        if(xa){
            for(u32 ph=0;ph<2u*ngr;++ph){const u32 op=ph/ngr,grp=g0+ph%ngr;
                if(ph)j1_bar(1,32u*J1X_T);   // previous phase's table reads done
#if J1X_F==3
                j1_fwd3_fill(j1_sm,op?tabB:tabA,pm,grp,threadIdx.x,32u*J1X_T);
                j1_bar(1,32u*J1X_T);
                j1_fwd3_body(j1_sm,grp,blockIdx.x,gridDim.x,w,J1X_T,op?xb:xa,op?frb:fra,65536u);}}
#else
                j1_fwd_fill(j1_sm,op?tabB:tabA,pm,grp,threadIdx.x,32u*J1X_T);
                j1_bar(1,32u*J1X_T);
                j1_fwd_body(j1_sm,grp,blockIdx.x,gridDim.x,w,J1X_T,op?xb:xa,op?frb:fra,65536u);}}
#endif
    }else if(A){
        const u32 iw=w-J1X_T;unsigned char* is=j1_sm+J1F_SMEM;
        {   u32* S=reinterpret_cast<u32*>(is+iw*J1X_ROWB);u64* TW=reinterpret_cast<u64*>(is+iw*J1X_ROWB+2176u);
            for(u32 g=blockIdx.x*J1X_I+iw;g<np*128u;g+=gridDim.x*J1X_I){const u32 prime=p0+(g>>7),rp=g&127u;
                j1_row_body(rp,prime,lane,S,TW,A,Bv,A,primes,pinvs,thi,tlo);
                if(lane==0){__threadfence();atomicAdd(done+prime,1u);}}}
        const u32 grp=iw>>2,ng=J1X_I/4u,tid=(iw&3u)*32u+lane,bar=2u+grp;
        u32* S2=reinterpret_cast<u32*>(is+grp*4u*J1X_ROWB);
        j1_bar128(bar);   // the group's 4 warps are done with their row areas
        for(u32 it=blockIdx.x*ng+grp;it<np*32u;it+=gridDim.x*ng){const u32 prime=p0+(it>>5),colblk=it&31u;
            if(tid==0){while(j1x_acquire(done+prime)<128u)__nanosleep(64);}
            j1_bar128(bar);
            j1_colinv_body<false,true>(colblk,prime,tid,bar,S2,A,Yo,primes,tw1);}
    }
    j1x_rec(rec,w,lane,t0);
}
// integer role of Y: colfwd of a and b on primes [p0, p0 + np) in place, 4 groups of 4 warps (warps 16..31)
__device__ __forceinline__ void j1y_colfwd(unsigned char* is,u32 w,u32 lane,u32* fA,u32* fB,u32 p0,u32 np,const u32* primes,const u64* tw1){
    const u32 iw=w-16u,grp=iw>>2,tid=(iw&3u)*32u+lane,bar=2u+grp;
    u32* S=reinterpret_cast<u32*>(is+grp*8704u);
    for(u32 it=blockIdx.x*4u+grp;it<2u*np*32u;it+=gridDim.x*4u){const u32 op=it&1u,r=it>>1,prime=p0+(r>>5),colblk=r&31u;
        u32* X=op?fB:fA;
        j1_colfwd_body<false>(colblk,prime,tid,bar,S,X,X,primes,tw1);}
}
extern "C" __global__ void __launch_bounds__(1024,1) j1_fuse_y(
    const u32* __restrict__ y,u32* __restrict__ out,const uint4* __restrict__ tm,const uint2* __restrict__ tf,
    u32* fA,u32* fB,u32 p0,u32 np,const u32* primes,const u64* tw1,unsigned long long* rec){
    extern __shared__ __align__(16) unsigned char j1_sm[];
    const u32 w=threadIdx.x>>5,lane=threadIdx.x&31u;const unsigned long long t0=clock64();
    if(w<16u){
        if(y)j1_back_body(j1_sm,blockIdx.x,gridDim.x,w,threadIdx.x,1u,y,out,tm,tf,65536u);
    }else if(fA)j1y_colfwd(j1_sm+J1B_SMEM,w,lane,fA,fB,p0,np,primes,tw1);
    j1x_rec(rec,w,lane,t0);
}
extern "C" __global__ void __launch_bounds__(1024,1) j1_fuse_y2(
    const u32* __restrict__ y,u32* __restrict__ out,u32* __restrict__ side,const uint4* __restrict__ tab2,
    u32* fA,u32* fB,u32 p0,u32 np,const u32* primes,const u64* tw1,unsigned long long* rec){
    extern __shared__ __align__(16) unsigned char j1_sm[];
    const u32 w=threadIdx.x>>5,lane=threadIdx.x&31u;const unsigned long long t0=clock64();
    if(w<16u){
        if(y){const u32 r=blockIdx.x&3u,bx=blockIdx.x>>2,nbx=(gridDim.x-r+3u)>>2;
            j1_back2_fill(j1_sm,tab2,r,threadIdx.x,512u);
            j1_bar(1,512u);
            j1_back2_body(j1_sm,reinterpret_cast<u32*>(j1_sm+J1C_TAB_BYTES+w*J1C_STAGE_BYTES),r,bx,nbx,w,16u,y,out,side,65536u);}
    }else if(fA)j1y_colfwd(j1_sm+J1C_SMEM,w,lane,fA,fB,p0,np,primes,tw1);
    j1x_rec(rec,w,lane,t0);
}
// Y3: tensor role = back3 range body (wt-j1conv bc485ca; table tab3, side rows of 80 B; its stage areas lie in the first
// 64 KB of the block's shared window: tensor area first, integer tiles after it); j1_back3_fin as a separate launch.
extern "C" __global__ void __launch_bounds__(1024,1) j1_fuse_y3(
    const u32* __restrict__ y,u32* __restrict__ out,u32* __restrict__ side,const uint4* __restrict__ tab3,
    u32* fA,u32* fB,u32 p0,u32 np,const u32* primes,const u64* tw1,unsigned long long* rec){
    extern __shared__ __align__(16) unsigned char j1_sm[];
    const u32 w=threadIdx.x>>5,lane=threadIdx.x&31u;const unsigned long long t0=clock64();
    if(w<16u){
        if(y){const u32 r=blockIdx.x&3u,bx=blockIdx.x>>2,nbx=(gridDim.x-r+3u)>>2;
            j1_back2_fill(j1_sm,tab3,r,threadIdx.x,512u);
            j1_bar(1,512u);
            j1_back3_body(j1_sm,reinterpret_cast<u32*>(j1_sm+J1C_TAB_BYTES+w*J1C_STAGE_BYTES),r,bx,nbx,w,16u,y,out,side,65536u);}
    }else if(fA)j1y_colfwd(j1_sm+J1C_SMEM,w,lane,fA,fB,p0,np,primes,tw1);
    j1x_rec(rec,w,lane,t0);
}
#endif
