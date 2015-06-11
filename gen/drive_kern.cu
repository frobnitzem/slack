#include <stdio.h>
#include <algorithm>
#include "cuda_bits.h"

//#include "cuPrintf.cuh"

extern "C" {
#include "tdot.h"
}

#if defined(TEST)


__global__ void tdot_kernel1(struct DotInfo *info,
                             const float *A, const float *B, float *C);

#define NTHR_MAX 65535
#define BLK_MAX 65535

/* Calling test kernel. */
// info is local, matrix ptrs are on CUDA device
extern "C"
void tdot1(struct DotInfo *info, const float *A, const float *B, float *C,
	   cudaStream_t stream) {
    int blk = info->clen/info->Cstride[0];
    int thr = info->Cstride[0];
    struct DotInfo *d_info;

    if(thr > NTHR_MAX || blk > BLK_MAX) {
        printf("output is too large for CUDA.\n");
        return;
    }

    gpuErrchk(cudaMalloc((void **)&d_info, info->len));
    //gpuErrchk(cudaMemcpy(d_info, info, info->len, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyAsync(d_info, info, info->len, cudaMemcpyHostToDevice, stream));

    //cudaPrintfInit();

    //dim3 threads( 16, 4 );
    //dim3 grid( (m - 1)/64 + 1, (n - 1)/16 + 1 );
    //printf("%d Blocks\n", info->clen);
    tdot_kernel1 <<< blk, thr, 0, stream >>> ( d_info, A, B, C );

    //cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();
    gpuErrchk(cudaStreamSynchronize(stream));

    gpuErrchk(cudaFree(d_info));
}

#elif defined(MAGMA)

extern "C"
void magmablas_sgemm_T_T_64_16_16_16_4_special(
            float *C, const float *A, const float *B,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            float alpha, float beta, cudaStream_t stream );
extern "C"
void tdot1(struct DotInfo *p, const float *A, const float *B, float *C,
	   cudaStream_t stream) {
    int sc0 = p->clen/p->Cstride[0];
    int sc1 = p->Cstride[0];
    if(p->nc != 2 || p->n != 1) {
        printf("Wrong shape!\n");
        return;
    }

    magmablas_sgemm_T_T_64_16_16_16_4_special(C, A, B,
            sc0, sc1, p->scontr[0],
            p->Astride[2], p->Bstride[1], p->Cstride[0],
            p->alpha, p->beta, stream);
}

#elif defined(TX)

#define KERN_name(tx,ty,wx,wy,n) tdot_kernel_T ## tx ## _ ## ty ## _W ## wx ## _ ## wy ## _N ## n
#define KERN(tx,ty,wx,wy,n) KERN_name(tx,ty,wx,wy,n)

#define NAME KERN(TX,TY,WX,WY,NEST)

// ex. tdot_kernel_T1_64_W16_1_N16
__global__ void NAME(
        int sa_stride0, int sa_stride1, int sa_stride2,
        int sb_stride0, int sb_stride1, int sb_stride2,
        int sc0, int sc1, int sc2,
                    const float alpha, const float *A, const float *B,
                    const float beta, float *C);

// Note that this only works if A,B,C have full block-sizes for now.
// info is local, matrix ptrs are on CUDA device
extern "C"
void tdot1(struct DotInfo *p, const float *A, const float *B, float *C,
	   cudaStream_t stream) {
    int sc0 = p->clen/p->Cstride[0];
    int sc1 = p->Cstride[0];
    if(p->nc != 2 || p->n != 1) {
        printf("Wrong shape!\n");
        return;
    }
    if(WX != 1 && p->Bstride[0] == 0) {
        printf("Needless copy along 0 direction.\n");
        return;
    }
    if(WY != 1 && p->Bstride[1] == 0) {
        printf("Needless copy along 1 direction.\n");
        return;
    }
    if(p->scontr[0]%NEST != 0) {
        printf("Contracted index is not a multiple of NEST=%d.\n", NEST);
        return;
    }
    if(sc0 % (TX*WX) != 0) {
        printf("Output index 0 not a multiple of %d*%d.\n", TX,WX);
        return;
    }
    if(sc1 % TY != 0) {
        printf("Output index 1 not a multiple of %d*%d.\n", TY,WY);
        return;
    }
    printf("%d x %d kernel launch.\n", p->clen/(TX*WX*TY*WY), TX*TY);
    NAME <<<
	p->clen/(TX*WX*TY*WY), TX*TY, 0, stream
    >>> (
      p->Astride[0], p->Astride[1], p->Astride[2],
      p->Bstride[0], p->Bstride[1], p->Bstride[2],
      sc0, sc1, p->scontr[0],
      p->alpha, A, B,
      p->beta,  C
    );
}

#elif defined(NAME)

//#define KERN_name(thr,work,nest) tdot_kernel_T ## thr ## _W ## work ## _N ## nest
//#define KERN(thr,work,nest) KERN_name(thr,work,nest)

//#define NAME KERN(THR,WORK,NEST)
#define BLK (32*8*32)

extern "C"
__global__ void NAME(
        int a0, int a1, int a2, int a3, int a4, int a5,
        int b0, int b1, int b2, int b3, int b4, int b5,
        int sc0, int sc1, int sc2, int sc3, int sc4, int sc5,
                    const float alpha, const float *A, const float *B,
                    const float beta, float *C);

extern "C"
void tdot1(struct DotInfo *p, const float *A, const float *B, float *C,
	   cudaStream_t stream) {
    int sc0 = p->clen / p->Cstride[0];
    int sc1 = p->Cstride[0] / p->Cstride[1];
    int sc2 = p->Cstride[1] / p->Cstride[2];
    int sc3 = p->Cstride[2];

    //tdot_kernel_T8_1_32_1_W1_1_1_32_N8_8
    printf("%d x %d kernel launch.\n", p->clen/BLK, BLK);
    NAME <<<
	p->clen/BLK, BLK, 0, stream
    >>> (
      p->Astride[0], p->Astride[1], p->Astride[2], p->Astride[3], p->Astride[4], p->Astride[5],
      p->Bstride[0], p->Bstride[1], p->Bstride[2], p->Bstride[3], p->Bstride[4], p->Bstride[5],
      sc0, sc1, sc2, sc3, p->scontr[0], p->scontr[1],
      p->alpha, A, B,
      p->beta,  C
    );
}

#else
//#define NAME tdot8_12_3T2_3A1_2B2_0
//#define NAME tdot32_32_16T8_8A0_2B1_2
//#define NAME tdot32_32_1_1_16T4_4_1_1A0_2_4B1_3_4
//#define NAME tdot16_1_1_16_4_4T4_1_1_4A4_2_0_5B1_5_4_3
#define NAME tdot24_4_4_24_4_4T4_4_4_4A4_2_0_5B1_5_4_3

extern "C" {
void
NAME(int sC0, int sC1, int sC2, int sC3, int sC4, int sC5,
    float alpha, const float* __restrict__ A, const float* __restrict__ B,
    float  beta,       float* __restrict__ C, cudaStream_t stream);

void tdot1(struct DotInfo *p, const float *A, const float *B, float *C,
	   cudaStream_t stream) {
    int sc0 = p->clen       / p->Cstride[0];
    int sc1 = p->Cstride[0] / p->Cstride[1];
    int sc2 = p->Cstride[1] / p->Cstride[2];
    int sc3 = p->Cstride[2];

    //printf("C shape = %d x %d x %d\n", sc0, sc1, p->scontr[0]);
    printf("C shape = (%d, %d, %d, %d) x (%d, %d)\n",
                sc0, sc1, sc2, sc3, p->scontr[0], p->scontr[1]);
    NAME(sc0, sc1, sc2, sc3, p->scontr[0], p->scontr[1],
                p->alpha, A, B, p->beta, C, stream);
}
}

#endif
