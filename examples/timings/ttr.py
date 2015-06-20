#!/usr/bin/env python

from ucgrad import zeros, array, arange, write_matrix
import timeit
#from pylab import plot, show

def best(run, setup, n):
    best = 1e20
    for i in range(n):
        t = timeit.timeit(run, setup, number=1)
        if t < best:
            best = t
    return best


trans = [(1,2,0,3), (0,3,2,1)]

setup = """
from numpy import zeros
N = %d
a = zeros((N, N, N, N))
"""

n = array(range(4, 64, 4) + range(64, 128, 8) + range(128, 256, 16))
#n = arange(4,5)

M = zeros((len(n), 1+len(trans)))
M[:,0] = n

for i, tr in enumerate(trans):
    t = array([best('a.transpose(%s).reshape((N*N,N*N))'%repr(tr), setup%N, 10)
             for N in n])
    print i, t

    #plot(n, t)
    #show()

    M[:,i+1] = t

write_matrix(M, "trans.dat")
