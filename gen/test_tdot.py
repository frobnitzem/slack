#* Copyright David M. Rogers
#*
#* This source file is released into the public domain.
#* Please use these index tricks and stop re-writing gemm.
#*

from ctypes import CDLL, c_int32, c_int, c_float
import numpy as np
import numpy.ctypeslib as ct
import os
from load_kern import load_kern
from plan import factors, factor

prod = lambda x: reduce(lambda a,b: a*b, x, 1)
mapply = lambda f, m: reduce(lambda a,b: a + f(b), m, [])
array = np.array
zeros = np.zeros
ones = np.ones
rand = np.random.random
where = np.where
log, exp = np.log, np.exp

# for greedy algo.
def argmax(l):
    top = 0
    for i in range(1, len(l)):
        if l[i] >= l[top]: # favor right indices
            top = i
    return top

tdot_generic = None
# Load the generic kernel (if not present).
def load_generic():
    global tdot_generic

    if tdot_generic != None:
        return
    cwd = os.path.dirname(os.path.abspath(__file__))
    dw = CDLL(os.path.join(cwd, "tdot.so"))

    # Shorthand for setting function prototypes
    def decl_fn(a, *args):
            a.argtypes = args[:-1]
            a.restype = args[-1]
    def void_fn(a, *args):
            decl_fn(a, *(args+(None,)))
    def int_fn(a, *args):
            decl_fn(a, *(args+(c_int32,)))
    def dbl_fn(a, *args):
            decl_fn(a, *(args+(c_float,)))
    # Building up data types used
    intarr = ct.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    inds   = ct.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    dblarr = ct.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')

    void_fn(dw.tensdot, c_float, dblarr, c_int32,
                        c_float, dblarr, c_int32, intarr, inds,
                                 dblarr, c_int32, intarr, inds)
    tdot_generic = dw.tensdot

# Use the generic kernel.
def tdot_generic(beta, C, alpha, A, pa, B, pb):
    load_generic()
    tdot_generic(beta, C, len(C.shape),
                 alpha, A, len(A.shape), array(A.shape, dtype=np.int32),
                                         pa.astype(np.uint8),
                        B, len(B.shape), array(B.shape, dtype=np.int32),
                                         pb.astype(np.uint8))

# Lookup table for kernel parameters: (work, thr_blk)
lookup = {(1, 1, (1,) , (1,0)) : ((256,32), (128,)), # axpy
          (1, 1, (1,) , (0,1)) : ((64,128), (64,)),
          (1, 1, (1,0), (1,)) : ((256,32), (128,)),
          (1, 1, (0,1), (1,)) : ((64,128), (64,)),
          (2, 1, (1,2), (0,2)) : ((16,32,64), (16,16)), # gemm
          (2, 1, (1,2), (2,0)) : ((64,16,64), (16,16)),
          (2, 1, (2,1), (0,2)) : ((16,64,64), (16,16)),
          (2, 1, (2,1), (2,0)) : ((64,64,16), (16,16)),
          (1, 2, (0,2,1), (1,2)) : ((32,16,16), (32,)),
          (1, 2, (0,1,2), (2,1)) : ((32,128,128), (32,)),
          (1, 2, (2,0,1), (1,2)) : ((32,128,32), (32,)),
          (1, 2, (1,0,2), (2,1)) : ((32,32,64), (32,)),
          (1, 2, (1,2,0), (1,2)) : ((128,4,16), (64,)),
          (1, 2, (2,1,0), (2,1)) : ((128,8,8), (64,)),
          #(3, 2, (0,3,2,4), (3,1,4)) : ((1,8,32,8,96), (1,4,16)), # cplx
          (3, 2, (0,3,2,4), (3,1,4)) : ((1,128,2,32,2), (1,64,2)),
          (4, 2, (0,1,4,5), (2,3,4,5)) : ((2,8,2,8,64,64), (2,8,2,8)),
          }

# Given a target sum, "T = t1, t2, ..., tn"
# distribute the factors, "F = {f}"
# to match the target as closely as possible.
# return the resulting sum in ea. bin.
# Note: This is close to the subset sum or multiprocessor
# scheduling problems, which are NP-complete.  So, I'm
# just using a greedy packing method here.
# If haggle is set, the sum in bin i will not exceed t[i].
#
# Returns a list of factors sent to ea. bin.
def greedy_distribute(t, f, haggle=False):
    t = [i for i in t] # copy, since we'll destroy it
    fac = [[] for i in t]
    rest = []
    for j in reversed(sorted(f)):
        k = argmax(t)
        if haggle and t[k] < j:
            rest.append(j)
            continue
        t[k] -= j
        fac[k].append(j)
    if haggle:
        return fac, rest
    return fac

# Redistribute N elements using greedy_distribute of N
def redistribute(old, N):
    f = greedy_distribute(old, [log(x) for x in factors(N)])
    return [int(exp(sum(u))+0.5) for u in f]

# Heuristic rules:
# 1. generate a priority for ea. index
#   - rhs = "max_dim" priority
# 2. schedule more operations per thr. blk on higher priority indices
# 3. distribute more threads towards higher-priority indices
# 4. obey constraints: Nthr = 128 or 192 or 256
#                      total registers < 16k per blk
def heuristic(nc, n, pa, pb, log_Nthr=8, log_Nreg=13):
    pri = [i/2 for i in range(nc)] + [0]*n
    max_dim = max(len(pa), len(pb))
    for i,j in enumerate(pa):
        pri[j] += i + max_dim - len(pa)
    for i,j in enumerate(pb):
        pri[j] += i + max_dim - len(pb)
    # Nominally, we would scale up all the priorities by a factor,
    # then declare victory.  However, we have constraints.
    # Higher numbers mean we need more inner-loops.
    # So we have top stop outer indices from
    # stealing all our reg. space.
    sum_out = sum(pri[:nc])
    sum_in  = sum(pri[nc:])
    wt = sum_in - sum_out
    # fix sum_in - sum_out ~ log_Nreg - 2*log_Nthr
    if wt > log_Nreg - 2*log_Nthr:
        log_Nthr = max((log_Nreg - wt)/2, 5)
        # but we need a power of 32 threads

    # specialized greedy_distribute
    blk = [1]*(n+nc)
    thr_blk = [1]*nc
    for j in range(log_Nthr):
        top = argmax(pri[:nc])
        pri[top] -= 1
        thr_blk[top] *= 2
        blk[top] *= 2
    for j in range(log_Nreg-log_Nthr):
        top = argmax(pri)
        pri[top] -= 1
        blk[top] *= 2

    return blk, thr_blk

# Check size restrictions!
def fix_sizes(blk, thr_blk, s, pa, pb):
    print "Fixing sizes for:"
    print blk, thr_blk
    nc = len(thr_blk)
    n = len(blk) - nc

    # Must have blk <= s
    blk = [min(a,b) for a,b in zip(s,blk)]
    thr_blk = [min(a,b) for a,b in zip(s,thr_blk)]
    blk[:nc] = [i - (i%m) for i,m in zip(blk, thr_blk)]

    # need Nthr to divide SA and SB
    sa = prod([blk[i] for i in pa])
    sb = prod([blk[i] for i in pb])
    Nt = prod(thr_blk)
    if sa < Nt or sb < Nt: # Too many threads!
        Nt = min(sa, sb)
        thr_blk = redistribute(thr_blk, Nt)
    print "sa, sb, Nt = "
    print sa, sb, Nt

    if sa % Nt != 0 or sb % Nt != 0: # must re-distribute
        sa -= sa % Nt
        sb -= sb % Nt
        if sa < Nt or sb < Nt: # Too many threads!
            Nt = min(sa, sb)
            thr_blk = redistribute(thr_blk, Nt)
        fa = factor(sa) # fix!
        fb = factor(sb)
        # combined factors (can appear in contraction)
        comb = {}
        for i,n in fa.iteritems():
            if fb.has_key(i):
                m = min(n, fb[i])
                fa[i] -= m
                fb[i] -= m
                comb[i] = m
        print "Re-factoring for:"
        print fa
        print fb
        f3, rest = greedy_distribute(map(log, blk[n:]),
                    mapply(lambda i,n: n*[log(i)], comb.iteritems()),
                    haggle=True)
        print f3, rest
        for j in rest: # return!
            fa[j] += 1
            fb[j] += 1
        # distribute remaining factors
        f1 = greedy_distribute([log(blk[i]) for i in pa],
                        mapply(lambda n: n[1]*[log(n[0])], fa.iteritems()))
        f2 = greedy_distribute([log(blk[i]) for i in pb],
                        mapply(lambda n: n[1]*[log(n[0])], fb.iteritems()))
        print f1, f2
        blk = [0]*nc + map(sum, f3)
        for i,j in enumerate(pa):
            blk[j] += sum(f1[i])
        for i,j in enumerate(pb):
            blk[j] += sum(f2[i])
        blk = [int(exp(x)+0.5) for x in blk] # back to products

    print blk, thr_blk
    return blk, thr_blk

# Generate and load a custom kernel, then carry out dot product.
def tdot(beta, C, alpha, A, pa, B, pb):
    nc = len(C.shape)
    n = (len(A.shape)+len(B.shape)-nc)/2
    s = zeros(nc + n, np.int32)
    for i,j in enumerate(pa):
        s[j] = A.shape[i]
    for i,j in enumerate(pb):
        s[j] = B.shape[i]

    key = (nc, n, tuple(pa), tuple(pb))
    try:
        #raise KeyError
        blk, thr_blk = lookup[(nc, n, tuple(pa), tuple(pb))]
    except KeyError:
        blk, thr_blk = heuristic(nc, n, pa, pb)

    blk, thr_blk = fix_sizes(blk, thr_blk, s, pa, pb)

    tdotk = load_kern(blk, thr_blk, pa, pb)
    if tdotk == None:
        print "Error creating kernel!"
        raise RuntimeError
    else:
        tdotk(beta, C, array(s, dtype=np.int32), alpha, A, B)

# Main testing routine generating matrix instances
# of the given shape and numerically testing.
def test_sz(sa, pa, sb, pb, nc):
    print("  ==  Test", sa, pa, sb, pb, nc, "==")
    A = rand(sa).astype(np.float32)
    #A = ones(sa, dtype=np.float32)
    B = rand(sb).astype(np.float32)
    na = len(sa)
    nb = len(sb)
    n = (na+nb-nc)/2

    sz = [0]*(nc+n)
    sind = [[0]*n, [0]*n]
    for i in range(na):
        sz[pa[i]] = sa[i]
        if pa[i] >= nc:
            sind[0][pa[i]-nc] = i
    for i in range(nb):
        sz[pb[i]] = sb[i]
        if pb[i] >= nc:
            sind[1][pb[i]-nc] = i
    C = zeros(sz[:nc], dtype=np.float32)

    pa = array(pa, dtype=np.int32)
    pb = array(pb, dtype=np.int32)
    tr = inv_perm(pa[where(pa < nc)].tolist() \
         + pb[where(pb < nc)].tolist()) # output permutation from tensdot
    print(sind, tr)

    tdot(0.0, C, 1.0, A, pa, B, pb)
    Cp = np.tensordot(A, B, sind)
    if len(tr) > 0 and tr != range(nc):
        Cp = np.transpose(Cp, tr)

    if not np.allclose(C, Cp):
        print(A)
        print(B)
        print(C)
        print(Cp - C)

	err = abs(C - Cp).max()
        n = np.argmax(np.reshape(abs(C - Cp), -1))
        print("Mismatch by %f at %d!"%(err,n))
        exit(1)
    print("OK.")

def permut(p, x):
    y = []
    for i in range(len(p)):
        y.append(x[p[i]])
    return y

def inv_perm(p):
    x = [0]*len(p)
    for i in range(len(p)):
        x[p[i]] = i
    return x

N = 4 # block-size
n1 = 11*N
n2 = 7*N
n3 = 5*N
n4 = 3*N
n5 = 2*N
n6 = 1*N

n1 = 8*N
n2 = 8*N
n3 = 4*N
n4 = 4*N
n5 = 2*N
n6 = N

test_sz((128,128,2,2), (0,3,2,4), (128,128,2), (3,1,4), 3)
exit(0)

# ger
#test_sz((n1, ),  [0], (n2,), [1], 2)
# dot
#test_sz((n1, ),  [0], (n1,), [0], 0)

# axpy - 4 ordering options
test_sz((n1, ),  [1], (n1,n2), [1,0], 1)
test_sz((n1, ),  [1], (n1,n2), [1,0], 1)
test_sz((n2, ),  [1], (n1,n2), [0,1], 1)
test_sz((n1,n2), [1,0], (n1,),   [1], 1)
test_sz((n1,n2), [0,1], (n2,),   [1], 1)

#n1 = 2
#n2 = 3
#n3 = 5
# gemm - 4 transpose options
test_sz((n1,n3), [2,1], (n1,n2), [2,0], 2)
test_sz((n3,n1), [1,2], (n1,n2), [2,0], 2)
test_sz((n1,n3), [2,1], (n2,n1), [0,2], 2)
test_sz((n3,n1), [1,2], (n2,n1), [0,2], 2)

# nontrivial 2-index contractions
test_sz((n1,n3,n2), [0,2,1], (n2,n3), [1,2], 1)
test_sz((n1,n3,n2), [0,1,2], (n2,n3), [2,1], 1)
test_sz((n3,n1,n2), [2,0,1], (n2,n3), [1,2], 1)
test_sz((n3,n1,n2), [1,0,2], (n2,n3), [2,1], 1)
test_sz((n2,n3,n1), [1,2,0], (n2,n3), [1,2], 1)
test_sz((n2,n3,n1), [2,1,0], (n2,n3), [2,1], 1)

test_sz((n4,n3,n2), [0,3,1], (n2,n3), [2,3], 3)
test_sz((n4,n3,n2), [0,1,3], (n2,n3), [3,2], 3)
test_sz((n3,n4,n2), [3,0,2], (n2,n3), [1,3], 3)
test_sz((n3,n4,n2), [2,0,3], (n2,n3), [3,1], 3)
test_sz((n2,n3,n4), [1,3,0], (n2,n3), [2,3], 3)
test_sz((n2,n3,n4), [3,2,0], (n2,n3), [3,1], 3)

# 2-index contractions, order 4 tensors
test_sz((n1,n2,n3,n4), [0,1,4,5], (n5,n6,n3,n4), [2,3,4,5], 4)
test_sz((n1,n2,n4,n3), [1,0,5,4], (n5,n6,n3,n4), [2,3,4,5], 4)
test_sz((n1,n2,n4,n3), [0,1,5,4], (n5,n6,n3,n4), [2,3,4,5], 4)
test_sz((n1,n2,n4,n3), [0,1,4,5], (n5,n6,n3,n4), [3,2,5,4], 4)
test_sz((n1,n2,n4,n3), [0,1,4,5], (n5,n6,n3,n4), [2,3,5,4], 4)
test_sz((n1,n2,n4,n3), [1,0,4,5], (n5,n6,n3,n4), [3,2,5,4], 4)

