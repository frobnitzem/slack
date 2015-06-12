# This is a scratch file used to create the initial outline
# of the tdot routine following "magmablas/gemm_stencil.cuh"
#
# It's an intermediary step before gen_tdot2.py.
#
#    __shared__ FloatingPoint_t sA[BLK_K][BLK_A+1];      // +1 only required if A is transposed
#   __shared__ FloatingPoint_t sB[BLK_B][BLK_K+1];      // +1 always required

#   // Registers for the innermost loop
#   FloatingPoint_t rC[THR_B][THR_A];
#   FloatingPoint_t rA[THR_A];
#   FloatingPoint_t rB[THR_B];

#NN
#FloatingPoint_t ra[BLK_K/THR_YA][BLK_A/THR_XA];
#FloatingPoint_t rb[BLK_B/THR_YB][BLK_K/THR_XB];

#thread_shape = (THR_Y, THR_X)
#a-copy = (N/THR_XA, THR_XA)
#b-copy = (N/THR_XB, THR_XB)

#NN -- inner step along major of A, minor of B
#const FloatingPoint_t *offs_dA = A + blx*BLK_A     + idyA*LDA + idxA;
#const FloatingPoint_t *offs_dB = B + bly*BLK_B*LDB + idyB*LDB + idxB;

#BLK_A = THR_A*THR_X
#BLK_B = THR_B*THR_Y

# A : (BLK_K, BLK_A)
# B : (BLK_B, BLK_K)
#rC : (WORK_B, WORK_A) -- work done by each thread

# -- Advance A, B, C to beginning of thread i/o locations
# A += ixA[txA, tyA+i*BLK_A]
# B += ixB[txB, tyB+j*BLK_B]
# C += ixC[tx+i*BLK_A,ty+j*BLK_B]
#
# sA[txA:BLK_K:THR_XA][tyA:BLK_A:THR_YA] = A[:BLK_K:THR_XA][:BLK_A:THR_YA]
# sB[txB:BLK_K:THR_XB][tyB:BLK_B:THR_YB] = B[:BLK_K:THR_XB][:BLK_B:THR_YB]
#
# sync
# for kk = BLK_K:K:BLK_K
#    A += ixA[0,BLK_K]
#    B += ixB[0,BLK_K]
#
#    -- start loads from gpu global memory
#    rA = A[:BLK_K:THR_XA][:BLK_A:THR_YA]
#    rB = B[:BLK_K:THR_XB][:BLK_B:THR_YB]
#    for k = 0:BLK_K -- compute GER
#      tA = sA[k][tx:BLK_A:THR_A]
#      tB = sB[k][ty:BLK_B:THR_B]
#      for n,m = 0:WORK_A, 0:WORK_B
#        rC[n][m] += tA[n]*tB[m] // fma
#      end for
#    end for
#    sync
#    sA[txA:BLK_K:THR_XA][tyA:BLK_A:THR_YA] = rA
#    sB[txB:BLK_K:THR_XB][tyB:BLK_B:THR_YB] = rB
#    sync
# end for
#
# -- do last multiplication
# tA = sA[k][tx:BLK_A:THR_A]
# tB = sB[k][ty:BLK_B:THR_B]
# for n,m = 0:WORK_A, 0:WORK_B
#     rC[n][m] += tA[n]*tB[m] // fma
# end for
# C[:BLK_A:THR_A][:WORK_B:THR_B] = alpha*rC + beta*C[...]

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
                tA[m] = sA[k][m*THR_X+idx];
            }
            #pragma unroll
            for(n=0; n<%(N)d; n++) {
                tB[n] = sB[k][n*THR_Y+idy];
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

