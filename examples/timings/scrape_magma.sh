#!/bin/sh

# Get the matrix size and magma, cublas times in ms
sed -n -e 's/ *\([0-9]*\)[^(]*(\([^)]*\))[^(]*(\([^)]*\)).*/\1 \2 \3/p' $1.out >$1.raw

# get min and convert to s
python <<<.
from ucgrad import *
x = read_array("$1.raw", (-1,10,3))
y = zeros((len(x), 2))
for i in range(2):
    y[:,i] = x[:,:,i+1].min(1)
write_matrix("$1.dat", y*1e-3)
.
