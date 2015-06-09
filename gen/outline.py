# This is a scratch file used to create the initial outline
# of the tdot routine following "magmablas/gemm_stencil.cuh"
#
# It's an intermediary step before gen_tdot2.py.
#
#    __shared__ FloatingPoint_t sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
#   __shared__ FloatingPoint_t sB[BLK_N][BLK_K+1];      // +1 always required

#   // Registers for the innermost loop
#   FloatingPoint_t rC[THR_N][THR_M];
#   FloatingPoint_t rA[THR_M];
#   FloatingPoint_t rB[THR_N];

#NN
#FloatingPoint_t ra[BLK_K/DIM_YA][BLK_M/DIM_XA];
#FloatingPoint_t rb[BLK_N/DIM_YB][BLK_K/DIM_XB];

#thread_shape = (DIM_Y, DIM_X)
#a-copy = (N/DIM_XA, DIM_XA)
#b-copy = (N/DIM_XB, DIM_XB)

#NN -- inner step along major of A, minor of B
#const FloatingPoint_t *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
#const FloatingPoint_t *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;

#BLK_M = THR_M*DIM_X
#BLK_N = THR_N*DIM_Y

#A : (BLK_K, BLK_M)
#B : (BLK_N, BLK_K)
#C : (THR_N, THR_M) -- work done by each thread

# Zero rC[THR_N][THR_M]
# sA[idyA:idyA+BLK_K:DIM_YA, idxA:idxA+BLK_M:DIM_XA]
#    = A[:BLK_K:DIM_YA, :BLK_M:DIM_XA]
# sB[idyB:idyB+BLK_N:DIM_YB, idxB:idxB+BLK_K:DIM_XB]
#    = B[:BLK_N:DIM_YB, :BLK_K:DIM_XB]
#
# sync
# Loop over kk = 0:K-BLK_K:BLK_K
#    offs_dA += BLK_K*LDA
#    offs_dB += BLK_K
#
#    Start loads from cuda memory
#    ra[:BLK_K/DIM_YA, :BLK_M/DIM_XA] = A[:BLK_K:DIM_YA, :BLK_M:DIM_XA]
#    rb[:BLK_N/DIM_YB, :BLK_K/DIM_XB] = B[:BLK_N:DIM_YB, :BLK_K:DIM_XB]
#    loop over k = 0:BLK_K and compute GER
#      rA[:THR_M] = sA[k][idx:idx+THR_M*DIM_X:DIM_X]
#      rB[:THR_N] = sB[idy:idy+THR_N*DIM_Y:DIM_Y][k]
#      loop over n = :THR_N and m = :THR_M
#        rC[n][m] += rA[m]*rB[n] // fma
#    sync
#    sA[idyA:idyA+BLK_K:DIM_YA][idxA:idxA+BLK_M:DIM_XA] 
#          = ra[:BLK_K/DIM_YA][BLK_M/DIM_XA]
#    sB[idyB:idyB+BLK_N:DIM_YB][idxB:idxB+BLK_K:DIM_XB]
#          = rb[:BLK_N/DIM_YB][:BLK_K/DIM_XB]
#    sync
# Do last multiplication
# offsC = (bly*BLK_N + idy)*LDC + blx*BLK_M + idx
# C[:THR_N*DIM_Y:DIM_Y][:THR_M*DIM_X:DIM_X] = alpha*regC + beta*C[...]

template = """void %(name)s(int sa0, int sa1, int sb0, int sb1) {
    const int tn = threadIdx.x;
    int k;
    %(decl)s
    %(ind)s
    // Preload sA/sB
    %(preload)s
    __syncthreads();
    // Central Load-Accumulate-Swap Loop
    %(inner_loops)s
        %(inner)s
        __syncthreads();
    %(end_inner_loops)s

    %(store)s
}
"""

multiply = """
        #pragma unroll
        for(k = 0; k<%(K)d; k++) {
            // Load tA and tB from shmem
            #pragma unroll
            for(m=0; m<%(M)d; m++) {
                tA[m] = sA[k][m*DIM_X+idx];
            }
            #pragma unroll
            for(n=0; n<%(N)d; n++) {
                tB[n] = sB[k][n*DIM_Y+idy];
            }
            
            // Compute
            #pragma unroll
            for(n=0; n<%(N)d; n++) {
                #pragma unroll
                for(m=0; m<%(M)d; m++) {
                    __fma(tA[m], tB[n], rC[n][m]);
                }
            }
        }
        __syncthreads();
"""

T = 6
blk = [4,4,3]
pa = [1,2]
pb = [2,0]

idxA, decA, preA, cpyA = plan_copy("A", pa, blk, T, nc, n)
idxB, decB, preB, cpyB = plan_copy("B", pb, blk, T, nc, n)
code = template % {
        'decl' : decA+decB,
        'ind' : get_indices(indA, indB),
        'preload' : preA+preB,
        'inner_loop' : "",
        'end_inner_loop' : "",
        'inner' : cpyA(cpyB(multiply)),
}

