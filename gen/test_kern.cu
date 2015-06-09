#include <algorithm>
//#include "cuPrintf.cu"
#include "tdot.h"

// Note: nc = na + nb - 2*n

#define BINARY_OP(A,B) ((A)*(B))

/* Compute 1 output element of C = alpha A . B + beta C
 * (reference for omnibus dimensions)
 * - each thread must sum over all contracted indices, and this
 *   loop is not nested either
 */
__global__ void tdot_kernel1(struct DotInfo *info,
                             const float *A, const float *B, float *C) {
    int i, j, h, l, J;
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    const int n = info->n;
    const int nc = info->nc;
    float acc = 0.0;
    const float *A0 = A; const float *B0 = B;

    LINK_INFO(info, n, nc);
    const int inner = n > 0 ? info->stride[0]*info->scontr[0] : 1;

    // Use output index to compute starting input indices.
    C += k;
    for(i=0; i<nc; i++) {
        j = k / info->Cstride[i];
        k %= info->Cstride[i];
        A += info->Astride[i]*j; // Astride = perm(pa, indprod(Ashape))
        B += info->Bstride[i]*j; // similar for B [unset dims = 0]
    }

    for(J=0; J<inner; J++) {
        l = J; // full recomputation
        j = k = 0;
        for(i=0; i<n; i++) { // loop over contracted indices to figure offset
            h = l/info->stride[i]; l %= info->stride[i];
            j += info->Astride[nc+i]*h;
            k += info->Bstride[nc+i]*h;
        }

        acc += BINARY_OP(A[j], B[k]);
        //cuPrintf("A[%d] = %f * B[%d] = %f -> %f\n", j,A[j], k,B[k], acc);
    }
    //cuPrintf("%f %f -> %f (inner = %d)\n", A[j], B[k], acc, inner);
    *C = info->alpha*acc + info->beta*(*C);
}

