#!/usr/bin/env python

import sys
from generate import *

# Generated code template
template = """
// Accumulate a work_blk sized chunk from pre-fetched tensor B into acc (C).
%(axpy)s

/* Compute an output block of C = alpha A . B + beta C
 * This code is auto-generated from a template.
 *
 * 1. Each thread must sum over all contracted indices.
 * 2. Outer/inner loops are explicitly blocked.
 * 3. Template code generally follows each comment.
 *
 * The inputs are:
 *
 * sa_stride  : Vec Int (%(nc)d+%(n)d) -- strides of A,
 *              permuted to (%(nc)d uncontracted + %(n)d contracted)
 * sb_stride  : Vec Int (%(nc)d+%(n)d) -- strides of B, permuted as in A
 *
 *   Note that unless you want a direct product, one of sa_stride[i]
 * or sb_stride[i] should be 0 for every output index (0 <= i < %(nc)d).
 *
 *   Also, work_blk determines what is pre-fetched from B.
 * If sb_stride[i] is 0, work_blk should be 1.  Otherwise,
 * stepping along i will have no effect on B, pre-fetching
 * the same number multiple times.
 * 
 * sc : Vec Int (%(nc)d+%(n)d) -- shape of C, extended on right by
 *                        shape of contraction inds
 *
 */
__global__ void tdot_kernel_%(name)s(%(args)s,
                    const float alpha, const float *A, const float *B,
                    const float beta, float *C) {
    // Declare contraction outer loop indices.
    %(loopdefs)s;
    int i, h, j = threadIdx.x, k = blockIdx.x;
    // Declare output buffer.
    float acc[%(acc_sz)d] = {%(acc_zeros)s};
    // Plan copy into prefetch buffer (nest_blk x work_blk).
    %(plan_copy)s

    // Compute uncontracted strides.
%(compute_nc_strides)s
    // Compute padded output strides.
%(compute_padc_strides)s
%(compute_start)s

    // Outer accumulator loops %(accum_loop)s
        // Prefetch a work_blk x nest_blk sub-block from B. %(prefetch)s
        __syncthreads();

        // Accumulate results into acc %(accum)s
        __syncthreads();
    %(end_accum_loop)s

    // Final C = alpha A . B + beta C %(store)s
}
"""

def plan_copy(thread_shape, work_blk, nest_blk, nws = 4):
    ws = "\n" + " "*nws
    nc = len(thread_shape)
    n = len(nest_blk)

    # split up nest_blk x work_blk into thread_shape ops
    tlen = prod(thread_shape)
    nlen = prod(nest_blk)
    wlen = prod(work_blk)

    # Combine shapes
    shape = [i for i in nest_blk] + [j for j in work_blk]
    try:
        index, copy = divide_work(shape, tlen)
    except ValueError as e:
        print "Error planning copy: " + str(e)
        raise ValueError, "prod(thread_shape) must divide prod(nest_blk*work_blk)"

    s = "__shared__ float pre[%d][%d];"%(nlen, wlen)

    niter = nlen*wlen/tlen
    s += ws + "// Allocating %s%s thread block to %s%s copies ea."%(
            str(index[:n]), str(index[n:]), str(copy[:n]), str(copy[n:]))
    s += ws + "int x = threadIdx.x / %d; const int tx = x*%d;"%(
            prod(index[n:]), prod(copy[:n]))
    s += ws + "int y = threadIdx.x %% %d; const int ty = y*%d;"%(
            prod(index[n:]), prod(copy[n:]))
    return index, copy, s
         
# Fills out pre[nest_blk][work_blk] due to A (or B) indices,
#  whichever holds nest_dim.
#
# in parallel using the provided thread_shape
#       Bb[tx][ty+0 ] = B[s1];
#       Bb[tx][ty+4 ] = B[s2];
#       Bb[tx][ty+8 ] = B[s3];
#       Bb[tx][ty+12] = B[s4];
def prefetch(thread_shape, work_blk, nest_blk, copy_blk):
    ws = "\n" + " "*8
    n = len(nest_blk)
    nc = len(thread_shape)

    s = ""
    for j,J in enumerate(loop_inds(copy_blk[:n])):
      for i,I in enumerate(loop_inds(copy_blk[n:])):
        s += ws + "pre[tx+%2d][ty+%2d] = B[%s];"%(
                            j, i, tens_index("sb_stride", I+J))
    return s

#   *C = alpha*acc + beta*(*C);
def store(work_blk, nws = 4):
    ws = "\n" + " "*nws
    s = ""
    for i,I in enumerate(loop_inds(work_blk)):
        ind = tens_index("sc_stride", I)
        s += ws + "C[%s] = alpha*acc[%2d] +"%(ind,i) \
           + " beta*C[%s];"%(ind)
        #  + ws + "  beta*C[%s];"%(ind)
    return s

def axpy(work_blk):
    s = """static __device__ void axpy(
    float alpha, const float * __restrict__ b,
                       float * __restrict__ c ) {"""
    for i in range(prod(work_blk)):
        s += "\n    c[%d] += alpha*b[%d];"%(i, i)
    return s + "\n}"

# This is a code generator to create nested routines for
# tensor dot products.  This automates writing index lookup
# functions for specialized tensor shapes.
# Output tile size per thread block = thread_shape * work_blk
# thread_dim : Vec Int nc -- Logical dimension of thread block.
#
# work_blk : Vec int nc -- Number of outputs along each dimension
#                            that each worker thread computes.
#
# nest_blk : Vec Int n -- Nested inner loop block sizes
#                       for accumulating contraction.
#
#    Note that to generate optimal code, prod(work_blk)*prod(nest_blk) should be
#  a multiple of prod(thread_shape).  A block of this size is cached
#  in parallel in the inner-loop.
#
def gen_tdot(thread_shape, work_blk, nest_blk):
    nc = len(thread_shape)
    if len(work_blk) != nc:
        raise ValueError, "nc differs between thread_shape and work_blk!"
    n = len(nest_blk)

    name =  "T" + "_".join(map(str, thread_shape)) \
         + "_W" + "_".join(map(str, work_blk)) \
         + "_N" + "_".join(map(str, nest_blk))

    cind, copy_blk, plan = plan_copy(thread_shape, work_blk, nest_blk)

    def args():
        s = "\n"
        for pre, k in [("sa_stride", nc+n), ("sb_stride", nc+n), ("sc", nc+n)]:
            s += "       "
            s += "".join([" int %s%d,"%(pre,i) for i in range(k)]) + "\n"
        return s[:-2]

    # On entry: j, k, x, y = (threadIdx.x, blockIdx.x, tx, ty)
    def compute_start():
        out_shape = [i*j for i,j in zip(thread_shape, work_blk)]
        out_stride = get_strides(out_shape, False)
        s = "    // Compute starting indices for %s*%s output tile."%(\
                    str(thread_shape), str(work_blk))
        ws = "\n    "
        idx =  [ ("j", get_strides(thread_shape), work_blk),
                 ("k", ["(sc_pstride%d/%d)"%(i,out_stride[i]) for i in range(nc)], \
                            out_shape)
               ]
        idx2 = [ ("y", get_strides(cind[n:]), copy_blk[n:]) ]

        i = 0
        for c,d in zip(compute_idx("i", idx, nc), compute_idx("h", idx2, nc)):
            s += c+d+"""
    A += sa_stride%(i)d*i;
    B += sb_stride%(i)d*(i+h);
    C += sc_stride%(i)d*i;\n""" % { 'i': i }
            i += 1

        # Final (inner) offsets for prefetch of B
        for d in compute_idx("h", \
                        [("x", get_strides(cind[:n]), copy_blk[:n])], n):
            s += d + "\n    B += sb_stride%d*h;"%i
            i += 1
        return s

# Old Code:
#   def compute_start():
#       for i in range(nc):
#           wstr = prod(thread_shape[i+1:])
#           ostr = prod(work_blk[i+1:])*wstr
#           subs = {
#               'i': i,
#               'wb': work_blk[i],
#               'ts': thread_shape[i],
#               'ostr': ostr,
#               'wstr': wstr,
#           }
#           if thread_shape[i] != 1:
#               s += ws + "i = ((k / (sc_pstride%(i)d/%(ostr)d))*%(ts)d"%subs
#               s +=      " + (j / %(wstr)d)) * %(wb)d;"%subs
#               if i < nc-1:
#                   s += ws + "j %%= %(wstr)d;"%subs
#           else:
#               s += ws + "i = (k / (sc_pstride%(i)d/%(ostr)d)) * %(wb)d;"%subs
#           if i < nc-1:
#               s += ws + "k %%= sc_pstride%(i)d / %(ostr)d;"%subs
#           s += """
#   A += sa_stride%(i)d*i;
#   B += sb_stride%(i)d*i;
#   C += sc_stride%(i)d*i;\n""" % subs

    def accum_loops():
        lp = ""
        for i in range(n):
          lp += "\n    for(J%(i)d=0; J%(i)d < (sc%(j)d+%(nest)d-1)/%(nest)d; J%(i)d++,A+=%(nest)d*sa_stride%(j)d,B+=%(nest)d*sb_stride%(j)d) {" % {
                  'i': i,
                  'j': i+nc,
                  'nest' : nest_blk[i],
                }
        return lp

    #       // accumulate into sub-blocks of C
    #       A += 4 * lda;
    #       saxpy( Ab[0], &Bb[0][0], Cb );  Ab[0] = A[0*lda];
    #       saxpy( Ab[1], &Bb[1][0], Cb );  Ab[1] = A[1*lda];
    #       saxpy( Ab[2], &Bb[2][0], Cb );  Ab[2] = A[2*lda];
    #       saxpy( Ab[3], &Bb[3][0], Cb );  Ab[3] = A[3*lda];
    #           
    #       ...
    #       A += 4 * lda;
    #       saxpy( Ab[0], &Bb[12][0], Cb );
    #       saxpy( Ab[1], &Bb[13][0], Cb );
    #       saxpy( Ab[2], &Bb[14][0], Cb );
    #       saxpy( Ab[3], &Bb[15][0], Cb );
    #           
    #       B += 16;
    #       // generally,
    #       acc[i,j] += A[i*oda + k*lda] * B[j*odb + k*ldb]
    #       B[j,k] are prefetched
    #    => redesign:
    #       i is const (subsets from A are distributed over threads or ext.)
    #       j is the dimension of acc (work_blk refers to subsets from B)
    #       k in [0, nest_blk) is explicitly written out
    def accum():
        # Loop over nested accumulated dims.
        # Note: we can "look-ahead" for next few A[] as well, e.g.
        # s = "\n        float Ap[4] = {A[0], A[lda], A[2*lda], A[3*lda]};
        # but this only gains ~ 3% in speed.
        s = ""
        for i,I in enumerate(loop_inds(nest_blk)):
            s += "\n        axpy(A[%s], &pre[%2d][0], acc);"%(
                    tens_index("sa_stride", I, nc), i)
        return s
    # axpy works on work_blk sized blocks

    def end_accum_loops():
        s = ""
        ws = "\n    "
        for i in range(n-1, 0, -1):
            s += "} A -= J%d*%d*sa_stride%d;"%(i,nest_blk[i], i+nc)
            s +=  " B -= J%d*%d*sb_stride%d;"%(i,nest_blk[i], i+nc)
            s += ws
        if n > 0:
            s += "}" + ws
        return s

    return template % {
            'axpy' : axpy(work_blk),
            'name' : name,
            'args' : args(),
            'loopdefs' : mk_inds("J", range(n)),
            'acc_sz' : prod(work_blk),
            'acc_zeros' : reduce(lambda a,b: a + "0.,", range(prod(work_blk)), ""),
            'plan_copy' : plan,
            'compute_nc_strides' : strides("sc", nc),
            'compute_padc_strides' : pad_strides("sc", \
                        [i*j for i,j in zip(thread_shape, work_blk)]),
# contracted strides no longer needed.
#            'compute_n_strides' : strides("sc", n, nc),
            'compute_start' : compute_start(),
            'accum_loop' : accum_loops(),
            'end_accum_loop' : end_accum_loops(),
            'prefetch' : prefetch(thread_shape, work_blk, nest_blk, copy_blk),
            'accum' : accum(),
            'store' : store(work_blk),
            'nc' : nc,
            'n'  : n,
           }

# An alternate convention (when the permutation is known) is:
# pa = [2,0], pb = [1,2], nest = (16,64,16)
# -- output dims from A, prefetch dims from B wlog
def get_shapes(pa, pb, nest, nc):
    thread_shape = [1]*nc
    work_blk = [1]*nc

    pa_name = map(str, pa)
    pb_name = map(str, pb)
    name = []
    for i in range(nc):
        if nest[i] == 1:
            try:
                pa_name[pa.index(i)] = str(i)
            except IndexError:
                pass
            try:
                pb_name[pb.index(i)] = str(i)
            except IndexError:
                pass
            continue
        name.append(str(nest[i]))

        if i in pa:
            pa_name[pa.index(i)] = str(i)
            thread_shape[i] = nest[i]
        else:
            pb_name[pb.index(i)] = str(i)
            work_blk[i] = nest[i]

    pa_name = "".join(pa_name)
    pb_name = "".join(pb_name)
    print "tdot%d_A"%nc + pa_name + "_B" + pb_name + "_N" + "_".join(name)
    thread_shape = tuple(thread_shape)
    work_blk = tuple(work_blk)
    nest_blk = nest[nc:]

    return thread_shape, work_blk, nest_blk

# work_blk controls prefetch from B, and so
# can only be != 1 on B dims
def test_gen():
    #thread_shape = (1, 1, 16)
    #work_blk = (1, 1, 1)

    #   Note that thread_shape references output dimensions,
    # and therefore has to be 1 where work_blk is != 1.
    # Otherwise, using all threads to pre-fetch will mix
    # different blocks of B.
    #  If thread_shape != 1, work_blk has to be 1.
    thread_shape = (1, 64)
    work_blk = (16, 1)

    #thread_shape = (1, 2, 4)
    #work_blk = (1, 16, 4)
    #nest_blk = (16,)
    thread_shape = (1,2)
    work_blk = (4,1)
    nest_blk = (1,)

    #thread_shape = (1,1,1,1)
    #work_blk = (1,3,1,3)
    #nest_blk = (2,2)
    print gen_tdot(thread_shape, work_blk, nest_blk)

def test_plan():
    out_shape = [i*j for i,j in zip(thread_shape, work_blk)]
    out_stride = get_strides(out_shape, False)
    nc = len(thread_shape)
    n = len(nest_blk)

    cind, copy, plan = plan_copy(thread_shape, work_blk, nest_blk)

    print plan

    idx =  [ ("k", ["(sc_pstride%d/%d)"%(i,out_stride[i]) for i in range(nc)], \
                        out_shape),
             ("j", get_strides(thread_shape), work_blk)
           ]
    idx2 = [ ("x", get_strides([1]*nc + cind[:n]), [1]*nc + copy[:n]),
             ("y", get_strides(cind[n:]),          copy[n:])
           ]

    for c,d in zip(compute_idx("i", idx, nc+n), compute_idx("h", idx2, nc+n)):
        print c
        print d

if __name__ == "__main__":
    test_gen()
    exit(0)
    argv = sys.argv
    thread_shape = tuple(map(int,argv[1].split(",")))
    work_blk     = tuple(map(int,argv[2].split(",")))
    nest_blk     = tuple(map(int,argv[3].split(",")))
    print gen_tdot(thread_shape, work_blk, nest_blk)

