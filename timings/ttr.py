#!/usr/bin/env python

from ucgrad import zeros, array, arange, write_matrix
import timeit
#from pylab import plot, show

setup = """
from numpy import zeros
N = %d
a = zeros((N, N, N, N))
"""

n = arange(4, 128, 4)
#n = arange(4,5)

t = array([timeit.timeit('a.transpose((1,2,0,3)).reshape((N*N,N*N))',
              setup=setup%N, number=10)
             for N in n])
print t

#plot(n, t)
#show()

M = zeros((len(n), 2))
M[:,0] = n
M[:,1] = t
write_matrix("trans1.dat", M)
