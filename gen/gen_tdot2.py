import sys
from generate import *
from index import *

template = """/* Compute an output block of C = alpha A . B + beta C
 * This code is auto-generated from gen_tdot2.py
 *
 * 1. Each thread must sum over all contracted indices.
 * 2. Outer/inner loops are explicitly nested.
 * 3. Kernel launch should be %(tA)d x %(tB)d,
 *       corresponding to number of threads working on A and B.
 *    Hint: use tdot%(name)s
 *
 * 4. This routine only works for the A/B permutations listed (A... B...).
 *    Because of this, it only needs the shapes of C and of contracted
 *    indices.
 */
__global__ void
tdot_k%(name)s(%(Cshape)s,
                float alpha, const float* __restrict__ A,
                             const float* __restrict__ B,
                float  beta,       float* __restrict__ C) {
    const int tx = threadIdx.x; const int ty = threadIdx.y;
    const int tn = tx*blockDim.y + ty;
    int x = blockIdx.x; int y = blockIdx.y;
    const int idx = %(idx)s; const int idy = %(idy)s;
    float tA[%(wA)d]; float tB[%(wB)d];
    float rC[%(wA)d][%(wB)d] = {%(Czero)s};
    %(decl)s
    %(compute_strides)s
    %(ind)s

    // Preload sA/sB %(preload)s
    __syncthreads();

    // Central Load-Accumulate-Swap Loop
    %(accum_loops)s
        %(inner)s
        __syncthreads();
    %(end_accum_loops)s

    %(last_accum)s

    %(store)s
}

extern "C"
void
tdot%(name)s(%(Cshape)s,
                float alpha, const float* __restrict__ A,
                             const float* __restrict__ B,
                float  beta,       float* __restrict__ C,
                cudaStream_t stream) {
    dim3 grid( (%(Aprod)s + %(bA)2d - 1)/%(bA)2d,
               (%(Bprod)s + %(bB)2d - 1)/%(bB)2d );
    dim3 threads(%(tA)d, %(tB)d);

    tdot_k%(name)s <<<
        grid, threads, 0, stream >>> (
      %(Cshape_var)s,
      alpha, A, B,
      beta,  C
    );
}
"""

# Plan a copy of nest_blk x work_blk from inp to out.
# This routine returns a shape for computing starting indices,
# an initial copy block, and
# a function that sandwiches a block
# of code using the copied values
#
# in-between dev -> register and register -> shared
# copies.
# 
# i.e. idx, cpy, f such that
# idx = (i0,i1,i2,...) 
#
# cpy = """#
#   sA[][] = A[][]
#"""
#  -- use sync here
# 
# f(code) = """
#   ra[][] = A[][]
#   code using sA
#   -- should end with sync
#   sA[][] = ra[][]
#"""
#  -- use sync here
#
# returns (index, decls, preload, f)
# plan_copy("A", "sA", "rA", (4,4,2,3), 96)
#  Note: magma uses ra instead of rA
# example is for perm = [j,n,l,k,m], nc = 4
# shape = [a,b,c,d,e,f]
def plan_copy_2D(A, perm, shape, T, out, nws=4):
    nc = out.n
    n = len(shape)-nc
    ws = "\n" + " "*nws
    M = 1
    N = 1
    pre = A.name
    sh = [shape[i] for i in perm] # shape of block from A [b,e,d,c,f]
    perm2 = [] # permutation from A to logical sA indices [j,n,l,k,m] (n,m on left)
    for i in perm:
        if i < nc:
            M *= shape[i]
            perm2.append(i+n)
        else:
            N *= shape[i]
            perm2.append(i-nc)

    try:
        index, copy = divide_work(sh, T) # strided loads from A -> rA / sA
    except ValueError as e:
        print "Error planning copy: " + str(e)
        raise ValueError, "number of threads must divide prod(blk shape) for each array"
    copy = TensIdx(copy) # [b1,e1,d1,c1,f1]

    # strided writes from rA -> sA
    s_idx = permut(perm, index, n+nc, 1) # [1,b0,c0,d0,e0,f0]
    s_cpy = permut(perm, copy.shape, n+nc, 1) # [1,b1,c1,d1,e1,f1]
    # put inner indices first
    s_idx = s_idx[nc:] + s_idx[:nc] # [e0,f0,1,b0,c0,d0]
    s_cpy = s_cpy[nc:] + s_cpy[:nc] # [e1,f1,1,b1,c1,d1]
    s_sh = [a*b for a,b in zip(s_idx, s_cpy)] # [e,f,1,b,c,d]

    d  = ws + "// Allocating %s thread block to %s r%s/s%s <- %s loads ea."%(
                                    str(index), str(copy.shape),pre,pre,pre)
    d += ws + "//       and %s%s thread block to %s%s s%s <- r%s writes ea."%(
          str(s_idx[:n]),str(s_idx[n:]), str(s_cpy[:n]),str(s_cpy[n:]),pre,pre)
    d += ws + "float r%s[%d];"%(pre, len(copy)) # same index order as A
    d += ws + "__shared__ float s%s[%d][%d];"%(pre, N, M) # order: (contr, uncontr)

    last = prod(s_idx[n:])
    d += ws + "const int tx%s = "%pre + re_index("(tn/%d)"%last, \
                                    s_idx[:n], s_sh[:n]) + ";"
    d += ws + "const int ty%s = "%pre + re_index("(tn%%%d)"%last, \
                                    s_idx[n:], s_sh[n:]) + ";"

    load = ws + "%s += %s;"%(pre, re_index_stride("tn", s_idx, perm2, pre))

    # immediate transpose
    for i,I in enumerate(copy):
        II = [a*b for a,b in zip(I,index)] # strided copy
                                           # [j*b0,n*e0,l*d0,k*c0,m*f0]
        JJ = permut(perm2, II, n+nc, 0) # absolute indices in sA order
                                        # [n*e0,m*f0, 0,j*b0,k*c0,l*d0]
        Ji = lin_index(JJ[:n], s_sh[:n])
        Jo = lin_index(JJ[n:], s_sh[n:])
        load += ws + "s%s[tx%s+%2d][ty%s+%2d] = %s[%s];"%(pre,
                        pre,Ji,   pre,Jo,   pre, A[II])

    # transpose on second copy (sA <- rA)
    def f(code, nws=8):
        ws = "\n" + " "*nws
        s = ""
        # BLK_K = thr_in*work_in, BLK_M = thr_out_A*work_out_A
        # DIM_YA = thr_in       , DIM_XA = thr_out_A
        # ra[:BLK_K/DIM_YA, :BLK_M/DIM_XA] = A[:BLK_K:DIM_YA, :BLK_M:DIM_XA]
        for i,I in enumerate(copy):
            II = [a*b for a,b in zip(I,index)] # strided copy
                                               # [j*b0,n*e0,l*d0,k*c0,m*f0]
            s += ws + "r%s[%2d] = %s[%s];"%(pre,i,pre,A[II])
        s += "\n" + code + "\n"
        # sA[tyA:tyA+BLK_K:DIM_YA][txA:txA+BLK_M:DIM_XA]
        #        = ra[:BLK_K/DIM_YA][BLK_M/DIM_XA]
        for J in loop_inds(s_cpy):
            #I = [J[a] for a in perm2] # registers are A-copy centric
            #i = lin_index(I, copy)    # [j,n,l,k,m]
            i = copy[[J[a] for a in perm2]]
            JJ = [a*b for a,b in zip(J,s_idx)] # absolute indices in sA order
                                               # [n*e0,m*f0, 0,j*b0,k*c0,l*d0]
            Ji = lin_index(JJ[:n], s_sh[:n]) # (n*e0)*f + m*f0
            Jo = lin_index(JJ[n:], s_sh[n:]) # ((j*b0)*c + k*c0)*d + l*d0
            s += ws + "s%s[tx%s+%2d][ty%s+%2d] = r%s[%2d];"%(pre,
                            pre,Ji, pre,Jo, pre, i)
        return s

    return d, load, f

bname = lambda blk: "_".join(map(str, blk))

# Running example:
# sC = [i,j,k,l] + [n,m]
# pa = [j,n,l,k,m] (1,4,3,2,5)
# pb = [i,m,n]     (0,5,4)
#
# thr_blk = [a1,b1,c1,d1]
# work_blk = [a0,b0,c0,d0,e,f]
def gen_tdot(thr_blk, work_blk, pa, pb):
    nc = len(thr_blk) # 4
    T = prod(thr_blk)
    if (len(pa)+len(pb)-nc)%2 == 1:
        print "Invalid output shape!"

    n = (len(pa)+len(pb)-nc)/2
    assert len(work_blk) == n+nc
    ws = "\n" + " "*4

    # [a,b,c,d,e,f]
    blk = [a*b for a,b in zip(thr_blk, work_blk)] + work_blk[nc:]
    outA = [i for i in pa if i < nc] # output inds of A [j,l,k]
    outA.sort() # [j,k,l]
    outB = [i for i in pb if i < nc] # output inds of B [i]
    outB.sort() # [i]
    ipA = [pa.index(i) for i in outA] # (0,3,2)
    ipB = [pb.index(i) for i in outB] # (0,)

    A = TensIdx_sym("A", ["sC%d"%i for i in pa])
    B = TensIdx_sym("B", ["sC%d"%i for i in pb])
    C = TensIdx_sym("C", ["sC%d"%i for i in range(nc)])

    thrA  = [thr_blk[i] for i in outA] # [b0,c0,d0]
    workA = TensIdx([work_blk[i] for i in outA]) # [b1,c1,d1]
    blkA  = TensIdx([blk[i] for i in outA]) # [b,c,d]
    thrB  = [thr_blk[i] for i in outB] # [a0]
    workB = TensIdx([work_blk[i] for i in outB]) # [a1]
    blkB  = TensIdx([blk[i] for i in outB]) # [a]
    #thrC  = TensIdx(thr_blk[:nc])
    workC = TensIdx(work_blk[:nc])
    #blkC  = TensIdx(blk[:nc])

    decA, preA, cpyA = plan_copy_2D(A, pa, blk, T, C)
    decB, preB, cpyB = plan_copy_2D(B, pb, blk, T, C)

    # padded output shapes (0 <= blockIdx.x[i] < padA.shape[i])
    #                      (0 <= blockIdx.y[i] < padB.shape[i])
    # ~map(round, [sA0, sA3, sA2])
    padA = TensIdx_sym("padA", ["((%s+%d-1)/%d)"%(C.shape[i], b, b) \
                                    for i,b in zip(outA,blkA.shape)])
    # ~map(round, [sB0])
    padB = TensIdx_sym("padB", ["((%s+%d-1)/%d)"%(C.shape[i], b, b) \
                                    for i,b in zip(outB,blkB.shape)])

    def skip():
        # x starts as blockIdx.x, tx as threadIdx.x
        # y starts as blockIdx.y, ty as threadIdx.y
        # per-thread work is strided 0:blk[i]:thr[i]
        indA = compute_idx_shape("n", [
                  ("x",  padA.shape, blkA.shape)], padA.n)
        indB = compute_idx_shape("n", [
                  ("y",  padB.shape, blkB.shape)], padB.n)

        ind_code = ""
        for i in range(len(indA)-1,-1,-1): # reversed([j,k,l])
            ind_code +=  "%s%sA += n*sA_stride%d;"%(indA[i],ws,ipA[i])
            ind_code += ws + "C += n*sC_stride%d;"%(outA[i])
        for i in range(len(indB)-1,-1,-1): # reversed([i])
            ind_code +=  "%s%sB += n*sB_stride%d;"%(indB[i],ws,ipB[i])
            ind_code += ws + "C += n*sC_stride%d;"%(outB[i])

        # unimplemented, but could be worked into a better solution...
        #coffA = indices("tx", permut(outA, thrA, nc, 1))
        #coffB = indices("ty", permut(outB, thrB, nc, 1))
        #ind_code += ' + '.join("(%s+%s)*sC_stride%d"%(a,b,i) \
        #                for i,a,b in zip(range(nc),coffA, coffB))

        # Ok, but doesn't use strides or combine neatly.
        ind_code += ws + "C += %s;"%re_index("tx", \
                permut(outA, thrA, nc, 1), C.shape) # [1, b0, c0, d0]
        ind_code += ws + "C += %s;"%re_index("ty", \
                permut(outB, thrB, nc, 1), C.shape) # [a0, 1, 1, 1]
        return ind_code

    def accum_loops():
        lp =  "A += %d*sA_stride%d;"%(blk[-1], pa.index(nc+n-1)) \
           + " B += %d*sB_stride%d;"%(blk[-1], pb.index(nc+n-1))
        for i in range(n):
          lp += ws + "for(; J%(i)d < sC%(j)d; J%(i)d+=%(nest)d, A+=%(nest)d*sA_stride%(ja)d, B+=%(nest)d*sB_stride%(jb)d) {" % {
                  'i': i,
                  'j': i+nc,
                  'ja' : pa.index(i+nc),
                  'jb' : pb.index(i+nc),
                  'nest' : work_blk[i+nc],
                }
        return lp

    def end_accum_loops():
        s = ""
        for i in range(n-1, 0, -1):
            s += "} A -= J%d*sA_stride%d;"%(i, pa.index(i+nc))
            s +=  " B -= J%d*sB_stride%d;"%(i, pb.index(i+nc))
            s += " J%d = 0;"%i
            s += ws
        if n > 0:
            s += "}"
        return s

    def accum(nws=8, sync=True):
        ws = "\n" + " "*(nws+4)
        s = indent(nws-8, """
        #pragma unroll
        for(k = 0; k<%d; k++) {"""%(prod(work_blk[nc:])))
        for i,I in enumerate(workA):
            II = [a*b for a,b in zip(I, thrA)]
            s += ws + "tA[%d] = sA[k][%2d+idx];"%(i,blkA[II])
        for j,J in enumerate(workB):
            JJ = [a*b for a,b in zip(J, thrB)]
            s += ws + "tB[%d] = sB[k][%2d+idy];"%(j,blkB[JJ])
        s += indent(nws-8, """
            #pragma unroll
            for(n=0; n<%d; n++) {
                #pragma unroll
                for(m=0; m<%d; m++) {
                    rC[n][m] += tA[n]*tB[m];
                }
            }
        }"""%(len(workA), len(workB)))
        if sync:
            s += "\n" + " "*nws + "__syncthreads();"
        return s

    def store():
        s = ""
        for i,I in enumerate(workC): # [i,j,k,l]
            II = [a*b for a,b in zip(I, thr_blk)] # [i*a0,j*b0,k*c0,l*d0]
            Ai = [I[j] for j in outA] # [j,k,l]
            Bi = [I[j] for j in outB] # [i]
            s += ws + "C[%s] = alpha*rC[%d][%d] + beta*C[%s];"%(
                       C[II], workA[Ai], workB[Bi], C[II])
        return s

    dec_loops = "int " + ", ".join( ["J%d = 0"%i for i in range(n-1)] \
                                   +["J%d = %d"%(n-1,blk[-1])]        )
    return template % {
            'name' :   bname(blk)  + "T" + bname(thr_blk) \
                     + "A" + bname(pa) + "B" + bname(pb),
            'Aprod' : "*".join([C.shape[i] for i in outA]),
            'Bprod' : "*".join([C.shape[i] for i in outB]),
            'Cshape' : ", ".join(["int sC%d"%i for i in range(nc+n)]),
            'Cshape_var' : ", ".join(["sC%d"%i for i in range(nc+n)]),
            'tA' : prod(thrA), 'tB' : prod(thrB),
            'wA' : len(workA), 'wB' : len(workB),
            'bA' : len(blkA),  'bB' : len(blkB),
            'Czero' : ', '.join('{' + ', '.join("0." for b in range(workB.sz)) \
                                + '}' \
                                for a in range(workA.sz)),
            'idx' : re_index("tx", thrA, blkA.shape),
            'idy' : re_index("ty", thrB, blkB.shape),
            'decl' : dec_loops + ", n, m, k;" + decA+decB,
            'compute_strides' :  A.strides() \
                               + B.strides() \
                               + C.strides(),
            'ind' : skip(),
            'preload' : preA + preB,
            'accum_loops' : accum_loops(),
            'end_accum_loops' : end_accum_loops(),
            'inner' : cpyA(cpyB(accum())),
            'last_accum' : accum(nws=4, sync=False),
            'store' : store(),
    }

def test():
    thr_blk = [2,3]
    work_blk = [4,4,3]
    pa = [1,2]
    pb = [2,0]
    print gen_tdot(thr_blk, work_blk, pa, pb)

if __name__ == "__main__":
    #test()
    #exit(0)
    argv = sys.argv
    if len(argv) < 5:
        print "Usage: %s <work_blk> <thread_blk> <pa> <pb>"%argv[0]
        sys.exit(1)
    work_blk = map(int,argv[1].split(","))
    thr_blk  = map(int,argv[2].split(","))
    pa       = map(int,argv[3].split(","))
    pb       = map(int,argv[4].split(","))
    print gen_tdot(thr_blk, work_blk, pa, pb)

