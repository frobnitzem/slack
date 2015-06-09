#* Copyright David M. Rogers
#*
#* This source file is released into the public domain.
#* Please use these index tricks and stop re-writing gemm.
#*

#void tensdot(float alpha, float *A, int na, int *sa, uint8_t *pa,
#                          float *B, int nb, int *sb, uint8_t *pb,
#             float beta,  float *C, int nc)

from ctypes import CDLL, c_int32, c_int, c_float
import numpy as np
import numpy.ctypeslib as ct
import os

array = np.array
zeros = np.zeros
rand = np.random.random
where = np.where

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

void_fn(dw.tensdot, c_float, dblarr, c_int32, intarr, inds,
                                dblarr, c_int32, intarr, inds,
                    c_float, dblarr, c_int32)

def tdot(alpha, A, pa, B, pb, beta, C):
    dw.tensdot(alpha, A, len(A.shape), array(A.shape, dtype=np.int32),
                                       pa.astype(np.uint8),
                      B, len(B.shape), array(B.shape, dtype=np.int32),
                                       pb.astype(np.uint8),
                beta, C, len(C.shape))

def test_sz(sa, pa, sb, pb, nc):
    print "  ==  Test", sa, pa, sb, pb, nc, "=="
    A = rand(sa)
    B = rand(sb)
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
    C = zeros(sz[:nc])

    pa = array(pa, dtype=np.int32)
    pb = array(pb, dtype=np.int32)
    tr = inv_perm(pa[where(pa < nc)].tolist() \
         + pb[where(pb < nc)].tolist()) # output permutation from tensdot
    print sind, tr

    tdot(1.0, A, pa, B, pb, 0.0, C)
    #print A
    #print B
    #print C
    Cp = np.tensordot(A, B, sind)
    if len(tr) > 0 and tr != range(nc):
        Cp = np.transpose(Cp, tr)
    if not np.allclose(C, Cp):
        print "Mismatch!"
        exit(1)
    print "OK."

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

n1 = 11
n2 = 7
n3 = 5
n4 = 3
n5 = 2
n6 = 1
# ger
test_sz((n1, ),  [0], (n2,), [1], 2)
# dot
test_sz((n1, ),  [0], (n1,), [0], 0)

# axpy - 4 ordering options
test_sz((n1, ),  [1], (n1,n2), [1,0], 1)
test_sz((n2, ),  [1], (n1,n2), [0,1], 1)
test_sz((n1,n2), [1,0], (n1,),   [1], 1)
test_sz((n1,n2), [0,1], (n2,),   [1], 1)

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

