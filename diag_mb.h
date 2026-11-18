#pragma once
// Exit-time instruction / butterfly micro-benchmarks (from tools/sm120_mb): one NVRTC program, kernels mb0..mb22
// (per-instruction thread-ops/clk/SM) and bf0 (current lazy-Shoup DIF butterfly), bf1 (Harvey lazy, p<2^30).
static const char* const kDiagMbSource=R"MB(typedef unsigned u32; typedef unsigned long long u64;
#define MBITER 4096
#define HEAD extern "C" __global__ void
#define PRO u32 a0=threadIdx.x^s,a1=a0*3+1,a2=a0*5+2,a3=a0*7+3,a4=a0+9,a5=a0+11,a6=a0+13,a7=a0+17;u32 m=s|1,c=s*7+5;__syncthreads();u64 t0=clock64();
#define EPI u64 t1=clock64();out[blockIdx.x*blockDim.x+threadIdx.x]=a0^a1^a2^a3^a4^a5^a6^a7;if(threadIdx.x==0)cyc[blockIdx.x]=t1-t0;
HEAD mb0(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){a0=a0*m+c;a1=a1*m+c;a2=a2*m+c;a3=a3*m+c;a4=a4*m+c;a5=a5*m+c;a6=a6*m+c;a7=a7*m+c;} EPI}
HEAD mb1(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){a0=__umulhi(a0,m)^a1;a1=__umulhi(a1,m)^a2;a2=__umulhi(a2,m)^a3;a3=__umulhi(a3,m)^a4;a4=__umulhi(a4,m)^a5;a5=__umulhi(a5,m)^a6;a6=__umulhi(a6,m)^a7;a7=__umulhi(a7,m)^a0;} EPI}
HEAD mb2(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){a0=a0+a1+c;a1=a1+a2+c;a2=a2+a3+c;a3=a3+a4+c;a4=a4+a5+c;a5=a5+a6+c;a6=a6+a7+c;a7=a7+a0+c;} EPI}
#define SH(x) {u32 q=__umulhi(x,m); u32 r=x*c-q*s; x=min(r,r-s);}
HEAD mb3(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){SH(a0)SH(a1)SH(a2)SH(a3)SH(a4)SH(a5)SH(a6)SH(a7)} EPI}
HEAD mb4(u32*out,u64*cyc,u32 s){PRO
 float f0=__uint_as_float(a0),f1=__uint_as_float(a1),f2=__uint_as_float(a2),f3=__uint_as_float(a3);
#pragma unroll 4
 for(int i=0;i<MBITER;i++){f0=fmaf(f0,1.0001f,0.5f);f1=fmaf(f1,1.0001f,0.5f);f2=fmaf(f2,1.0001f,0.5f);f3=fmaf(f3,1.0001f,0.5f);}
 a0=__float_as_uint(f0);a1=__float_as_uint(f1);a2=__float_as_uint(f2);a3=__float_as_uint(f3); EPI}
HEAD mb5(u32*out,u64*cyc,u32 s){PRO
 int d0=0,d1=0,d2=0,d3=0,e0=0,e1=0,e2=0,e3=0,g0=0,g1=0,g2=0,g3=0,h0=0,h1=0,h2=0,h3=0;
#pragma unroll 4
 for(int i=0;i<MBITER;i++){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5));
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+r"(e0),"+r"(e1),"+r"(e2),"+r"(e3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a6),"r"(a7));
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+r"(g0),"+r"(g1),"+r"(g2),"+r"(g3):"r"(a4),"r"(a5),"r"(a6),"r"(a7),"r"(a0),"r"(a1));
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+r"(h0),"+r"(h1),"+r"(h2),"+r"(h3):"r"(a4),"r"(a5),"r"(a6),"r"(a7),"r"(a2),"r"(a3));}
 a0=d0^d1^d2^d3^e0^e1^e2^e3^g0^g1^g2^g3^h0^h1^h2^h3; EPI}
HEAD mb6(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){a0=min(a0+a1,a0-c);a1=min(a1+a2,a1-c);a2=min(a2+a3,a2-c);a3=min(a3+a4,a3-c);a4=min(a4+a5,a4-c);a5=min(a5+a6,a5-c);a6=min(a6+a7,a6-c);a7=min(a7+a0,a7-c);} EPI}
HEAD mb7(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){u64 w0=(u64)a0*m,w1=(u64)a1*m,w2=(u64)a2*m,w3=(u64)a3*m; a0=(u32)(w0>>32)^(u32)w0;a1=(u32)(w1>>32)^(u32)w1;a2=(u32)(w2>>32)^(u32)w2;a3=(u32)(w3>>32)^(u32)w3;} EPI}
HEAD mb8(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){a0=a0*m+c;a1=a1*m+c;a2=a2*m+c;a3=a3*m+c; a4=(a4^a5)&c;a5=(a5|a6)^m;a6=(a6^a7)+c;a7=(a7^a4)|m;} EPI}
#define L3(x,y) asm volatile("lop3.b32 %0,%0,%1,%2,0x96;":"+r"(x):"r"(y),"r"(c));
HEAD mb9(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){L3(a0,a1)L3(a1,a2)L3(a2,a3)L3(a3,a4)L3(a4,a5)L3(a5,a6)L3(a6,a7)L3(a7,a0)} EPI}
#define AD(x,y) asm volatile("add.u32 %0,%0,%1;":"+r"(x):"r"(y));
HEAD mb10(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){AD(a0,a1)AD(a1,a2)AD(a2,a3)AD(a3,a4)AD(a4,a5)AD(a5,a6)AD(a6,a7)AD(a7,a0)} EPI}
#define MN(x,y) asm volatile("min.u32 %0,%0,%1;":"+r"(x):"r"(y));
HEAD mb11(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){MN(a0,a1)MN(a1,a2)MN(a2,a3)MN(a3,a4)MN(a4,a5)MN(a5,a6)MN(a6,a7)MN(a7,a0)} EPI}
#define FF(x) asm volatile("fma.rn.f32 %0,%0,%1,%1;":"+f"(x):"f"(fc));
HEAD mb12(u32*out,u64*cyc,u32 s){PRO float fc=1.0001f;
 float f0=a0,f1=a1,f2=a2,f3=a3,f4=a4,f5=a5,f6=a6,f7=a7;
#pragma unroll 4
 for(int i=0;i<MBITER;i++){FF(f0)FF(f1)FF(f2)FF(f3)FF(f4)FF(f5)FF(f6)FF(f7)}
 a0=f0+f1+f2+f3+f4+f5+f6+f7; EPI}
HEAD mb13(u32*out,u64*cyc,u32 s){PRO float fc=1.0001f;
 float f0=a0,f1=a1,f2=a2,f3=a3;
#pragma unroll 4
 for(int i=0;i<MBITER;i++){FF(f0)L3(a0,a1)FF(f1)L3(a1,a2)FF(f2)L3(a2,a3)FF(f3)L3(a3,a0)}
 a4=f0+f1+f2+f3; EPI}
HEAD mb14(u32*out,u64*cyc,u32 s){PRO u32 z=s*13+7;
#pragma unroll 4
 for(int i=0;i<MBITER;i++){a0=min(a0+m,z);a1=min(a1+m,z);a2=min(a2+m,z);a3=min(a3+m,z);a4=min(a4+m,z);a5=min(a5+m,z);a6=min(a6+m,z);a7=min(a7+m,z);z^=a0;} EPI}
HEAD mb15(u32*out,u64*cyc,u32 s){PRO u32 z=s*13+7;
#pragma unroll 4
 for(int i=0;i<MBITER;i++){
 #define AM(x) {u32 t; asm volatile("add.u32 %0,%1,%2;":"=r"(t):"r"(x),"r"(m)); asm volatile("min.u32 %0,%1,%2;":"=r"(x):"r"(t),"r"(z));}
 AM(a0)AM(a1)AM(a2)AM(a3)AM(a4)AM(a5)AM(a6)AM(a7)} EPI}
HEAD mb16(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){ // xadd: s=a+b; min(s,s-p)
 #define XA(x,y) {u32 t=x+y; x=min(t,t-c);}
 XA(a0,a1)XA(a1,a2)XA(a2,a3)XA(a3,a4)XA(a4,a5)XA(a5,a6)XA(a6,a7)XA(a7,a0)} EPI}
#define IM(x) asm volatile("mad.lo.u32 %0,%0,%1,%2;":"+r"(x):"r"(m),"r"(c));
HEAD mb17(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){IM(a0)MN(a1,a2)IM(a2)MN(a3,a4)IM(a4)MN(a5,a6)IM(a6)MN(a7,a0)} EPI}
HEAD mb18(u32*out,u64*cyc,u32 s){PRO float fc=1.0001f; float f0=a1,f1=a3,f2=a5,f3=a7;
#pragma unroll 4
 for(int i=0;i<MBITER;i++){IM(a0)FF(f0)IM(a2)FF(f1)IM(a4)FF(f2)IM(a6)FF(f3)} a1=f0+f1+f2+f3; EPI}
#define AI(x) asm volatile("add.u32 %0,%0,12345;":"+r"(x));
HEAD mb19(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){AI(a0)AI(a1)AI(a2)AI(a3)AI(a4)AI(a5)AI(a6)AI(a7)} EPI}
#define SHL(x) asm volatile("shl.b32 %0,%0,3;":"+r"(x));
HEAD mb20(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){SHL(a0)SHL(a1)SHL(a2)SHL(a3)SHL(a4)SHL(a5)SHL(a6)SHL(a7)} EPI}
#define SETP(x,y) asm volatile("{.reg .pred q; setp.lt.u32 q,%0,%1; selp.u32 %0,%0,%1,q;}":"+r"(x):"r"(y));
HEAD mb21(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){SETP(a0,a1)SETP(a1,a2)SETP(a2,a3)SETP(a3,a4)SETP(a4,a5)SETP(a5,a6)SETP(a6,a7)SETP(a7,a0)} EPI}
HEAD mb22(u32*out,u64*cyc,u32 s){PRO
#pragma unroll 4
 for(int i=0;i<MBITER;i++){a0=__byte_perm(a0,a1,0x5410);a1=__byte_perm(a1,a2,0x5410);a2=__byte_perm(a2,a3,0x5410);a3=__byte_perm(a3,a4,0x5410);a4=__byte_perm(a4,a5,0x5410);a5=__byte_perm(a5,a6,0x5410);a6=__byte_perm(a6,a7,0x5410);a7=__byte_perm(a7,a0,0x5410);} EPI}

typedef unsigned u32; typedef unsigned long long u64;
#define BFITER 2048
__device__ __forceinline__ u32 xadd(u32 a,u32 b,u32 p){const u32 s=a+b;return min(s,s-p);}
__device__ __forceinline__ u32 xmsh(u32 x,u32 w,u32 wq,u32 p){const u32 q=__umulhi(x,wq);const u32 r=x*w-q*p;return min(r,r-p);}
__device__ __forceinline__ u32 xmshl(u32 x,u32 w,u32 wq,u32 p){const u32 q=__umulhi(x,wq);return x*w-q*p;}
// k0: full DIF butterfly (current); k1: Harvey lazy (p<2^30): x'=min(s,s-2p) , y'=shoup lazy(x+2p-y)
template<int V> __device__ void bfbody(u32* out,u64* cyc,u32 s){
  u32 v[8];for(int i=0;i<8;i++)v[i]=threadIdx.x*(i+3)+s;
  const u32 p=(V==1)?1073479681u:2147221505u, w=s*77+3, wq=s*91+5;
  __syncthreads();u64 t0=clock64();
#pragma unroll 2
  for(int it=0;it<BFITER;it++){
#pragma unroll
    for(int m=0;m<4;m++){u32 x=v[m],y=v[m+4];
      if(V==0){v[m]=xadd(x,y,p);v[m+4]=xmsh(x+p-y,w,wq,p);}
      else {v[m]=xadd(x,y,2*p);v[m+4]=xmshl(x+2*p-y,w,wq,p);}}
#pragma unroll
    for(int m=0;m<8;m+=2){u32 x=v[m],y=v[m+1];
      if(V==0){v[m]=xadd(x,y,p);v[m+1]=xmsh(x+p-y,w+m,wq,p);}
      else {v[m]=xadd(x,y,2*p);v[m+1]=xmshl(x+2*p-y,w+m,wq,p);}}
  }
  u64 t1=clock64();u32 a=0;for(int i=0;i<8;i++)a^=v[i];out[blockIdx.x*blockDim.x+threadIdx.x]=a;if(threadIdx.x==0)cyc[blockIdx.x]=t1-t0;}
extern "C" __global__ void bf0(u32*o,u64*c,u32 s){bfbody<0>(o,c,s);}
extern "C" __global__ void bf1(u32*o,u64*c,u32 s){bfbody<1>(o,c,s);}
)MB";
