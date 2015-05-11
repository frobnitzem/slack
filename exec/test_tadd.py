#* Copyright David M. Rogers
#*
#* This source file is released into the public domain.
#* Please use these index tricks and stop re-writing gemm.
#*

#void tensdot(double alpha, double *A, int na, int *sa, int *pa,
#                           double *B, int nb, int *sb, int *pb,
#             double beta,  double *C, int n)

from ctypes import CDLL, c_int32, c_int, c_double
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
        decl_fn(a, *(args+(c_double,)))
# Building up data types used
intarr = ct.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
dblarr = ct.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')



void_fn(dw.tensadd, c_double, dblarr, c_int32, intarr,
                    c_double, dblarr, intarr)

def tadd(alpha, A, beta, B, perm):
    assert len(A.shape) == len(B.shape) and np.prod(A.shape) == np.prod(B.shape)
    dw.tensadd(alpha, A, len(A.shape), array(A.shape, dtype=np.int32),
               beta,  B, array(perm, dtype=np.int32))

def test_sz(sa, perm):
    print "  ==  Test", sa, perm, "=="
    A = rand(sa)
    B = rand(permut(inv_perm(perm), sa))

    Ap = 1.0*A + 2.0*np.transpose(B, perm)
    tadd(1.0, A, 2.0, B, perm)
    if not np.allclose(A, Ap):
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

test_sz((n1,), [0])
test_sz((n1,n2), [0,1])
test_sz((n1,n2), [1,0])
test_sz((n1,n2,n3), [0,1,2])
test_sz((n1,n2,n3), [0,2,1])
test_sz((n1,n2,n3), [1,0,2])
test_sz((n1,n2,n3), [1,2,0])
test_sz((n1,n2,n3), [2,0,1])
test_sz((n1,n2,n3), [2,1,0])

test_sz((n1,n2,n3,n4), [0,1,2,3])
test_sz((n1,n2,n3,n4), [3,2,1,0])
test_sz((n1,n2,n3,n4), [0,2,1,3])
test_sz((n1,n2,n3,n4), [3,1,2,0])
test_sz((n1,n2,n3,n4), [1,0,3,2])

