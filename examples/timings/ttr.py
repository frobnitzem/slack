#!/usr/bin/env python

from ucgrad import zeros, array, arange, write_matrix
import timeit
#from pylab import plot, show

trans = [(1,2,0,3), (0,3,2,1)]

setup = """
from numpy import zeros
N = %d
a = zeros((N, N, N, N))
"""

n = array(range(4, 64, 4) + range(64, 128, 8) + range(128, 256, 16))
#n = arange(4,5)

M = zeros((len(n), 2))
M[:,0] = n

for i, tr in enumerate(trans):
    t = array([timeit.timeit('a.transpose(%s).reshape((N*N,N*N))'%repr(tr),
              setup=setup%N, number=10)
             for N in n])
    print i, t

    #plot(n, t)
    #show()

    M[:,1] = t
    write_matrix("trans%d.dat"%(i+1), M)
