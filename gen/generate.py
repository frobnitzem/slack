# Helper functions for array code generators.

import sys
from plan import divide_work

prod = lambda x: reduce(lambda a,b: a*b, x, 1)

def permut(p, x, n=None, default=None):
    if n == None:
        y = [default]*len(p)
    else:
        y = [default]*n
    try:
        for xi,j in zip(x,p):
            y[j] = xi
    except IndexError as e:
        raise IndexError, "Invalid permutation: %s is out of range"%str(e)

    return y

def mk_inds(pre, li):
    if(li == []):
        return ""
    inds = "int "
    for i in li[:-1]:
        inds += "%s%d, "%(pre,i)
    inds += "%s%d"%(pre,li[-1])
    return inds

def loop_inds(shape):
    d = len(shape)
    n = [prod(shape[i:]) for i in range(d+1)]
    x = [0]*d
    I = 0
    while 1:
        yield x
        i = d
        while (I+1)%n[i] == 0:
            if i == 0:
                raise StopIteration
            x[i-1] = (x[i-1] + 1)%shape[i-1]
            i -= 1
        I += 1

def get_strides(s, zero=True):
    k = 1
    t = [0]*len(s)
    for i in range(len(s)-1, -1, -1):
        r = s[i]
        if zero and s[i] == 1: # mark inoperative dim with stride = 0
            t[i] = 0
        else:
            t[i] = k
        k *= s[i]
    return t

# Generate code to index a tensor, using strides named
# with prefix 'pre'.
#
# tens_index("sa_stride", (2,1,0), 4)
#  ~> "2*sa_stride4 + 1*sa_stride5"
def tens_index(pre, I, off=0):
    if all(x == 0 for x in I):
        return "0"

    if len(I) == 1:
        return "%2d*%s%d"%(I[0], pre, off)
    return " + ".join("%2d*%s%d"%(x, pre, i + off) \
                      for i,x in enumerate(I) if x != 0)

# Compute linear index corresponding to (i0,i1,...)
# in an array with shape (s0,s1,...) -- note s0 is ignored.
def lin_index(ind, s):
    n = len(ind)
    if n == 0:
        return 0
        raise ValueError, "lin_index called with blank index!"
    x = ind[0]
    for i in range(1,n):
        x = x*s[i] + ind[i]
    return x

# Returns a one-liner to re-index var
# from shape s0 to shape s1 (same lengths).
# Note prod(s0) == prod(s1) and s0 must be [Int].
def re_index(var, s0i, s1i):
    s0 = []
    s1 = []
    wt = [] # accumulated output width
    # Exclude unused dimensions.
    for i,j in zip(s0i, s1i):
        if i != 1:
            wt.append(str(j))
            s0.append(i)
            s1.append("*".join(wt))
            wt = []
        else:
            wt.append(str(j))

    n = len(s0)
    if n == 0:
        return "0"

    ustride = [prod(s0[i:]) for i in range(1, len(s0))]

    def ind(i):
        if ustride[i-1] == 1:
            return 0
        if i == n-1 or ustride[i] == 1:
            return "%s%%%d"%(var,ustride[i-1])
        return "(%s%%%d)/%d"%(var,ustride[i-1],ustride[i])

    if n == 1:
        code = var
    else:
        code = "("*(n-1) + "%s/%d"%(var,ustride[0])
    for i in range(1, n):
        code += ")*%s + "%str(s1[i]) + ind(i)

    if len(wt) != 0: # some 1-s on rhs of s0i
        code = "(" + code + ")*" + "*".join(wt)

    return code

# Out-of-order re-index using strides
# perm is permutation from pre to s0.
def re_index_stride(var, s0, perm, pre):
    n = len(s0)
    if n == 0:
        return "0"

    ustride = [prod(s0[i:]) for i in range(1, n+1)]

    def ind(i):
        if ustride[i] == 1:
            return "%s%%%d"%(var,ustride[i-1])
        return "(%s%%%d)/%d"%(var,ustride[i-1],ustride[i])
    def stride(v, i):
        j = perm.index(i) # Lookup destination index for re_index.
        return "%s*s%s_stride%d"%(v, pre, j)

    if s0[0] != 1:
        if n == 1:
            l = [stride(var, 0)]
        else:
            l = [stride("(%s/%d)"%(var,ustride[0]), 0)]
    else:
        l = []
    l += [stride("(%s)"%ind(i), i) for i in range(1, n) if s0[i] != 1]
    if len(l) == 0:
        return "0"
    return " + ".join(l)

# inverse of lin_index
def get_index(x, s):
    n = len(s)
    I = [0]*n
    for i in range(n-1, -1, -1):
        I[i] = x%s[i]
        x /= s[i]
    if x != 0:
        raise ValueError, "Out-of bounds by %d arrays!"%x
    return I

# Takes a list with elements of the form:
#   (total index name, [decoding strides], [steps])
#  e.g. ("j", get_strides(thread_shape), work_blk)
#       ("k", ["(sc_pstride%d/%d)"%(i,out_stride[i]) for i in range(nc)],\
#                   out_shape)
# and emits code blocks -- i.e. a list of strings to set 'out'
# to successive indices.
#
#   Note that the the strides must be ordered largest to smallest.
# If no strides are present for a given dimension,
# the index is left alone.  The emitted code destroys
# the starting total indices (assumed to start in 'out').
#
#   Note that the code is simplified in the case where
# the index must be 0.  This has to be marked by setting
# the corresponding stride to 0.
#
#  Ex. code:
#    i = (k/(sc_pstride%(i)d/%(ostr)d))*%(os)d + (j/%(wstr)d)*%(wb)d;
#    k %%= sc_pstride%(i)d/%(ostr)d;
#    j %%= %(wstr)d;
def compute_idx(out, idx, n, nws=4):
    ws = "\n"+" "*nws
    blk = []
    for k in range(n):
        l = []
        r = "" # remainder code
        for i,x in enumerate(idx):
            if len(x[1]) <= k:
                continue
            stride = x[1][k]
            step = x[2][k]
            if stride != 0: # no contribution to idx for this dim.
                if stride == 1: # must be last dim
                    if len(x[1]) != k+1 and not all(u == 0 for u in x[1][k+1:]):
                        raise ValueError, "Error! stride 1 in " + \
                                            "%s is not last dim!"%str(x)
                    if step == 1:
                        l.append( "%s"%(x[0]) )
                    else:
                        l.append( "%s*%s"%(x[0], str(step)) )
                elif step == 1:
                    l.append( "%s/%s"%(x[0], str(stride)) )
                else:
                    l.append( "(%s/%s)*%s"%(x[0], str(stride), str(step)) )

                if len(x[1]) > k+1 and stride != 1: # last index
                    r += ws + "%s %%= %s;"%(x[0], str(stride))
        if len(l) == 0:
            s = ws + "%s = 0;"%out
        else:
            s = ws + "%s = "%out + " + ".join(l) + ";"
        s += r
        blk.append(s)
    return blk

# Emit code to compute indices from a shape when looped over
# in reverse order!
# Here, idx = [(tot, Shape, Steps)],
# and all Shape-s and Steps-s must be length n.
def compute_idx_shape(out, idx, n, nws=4):
    ws = "\n"+" "*nws
    blk = []
    #for k in range(n-1,-1,-1):
    for k in range(n):
        l = []
        r = "" # remainder code
        for i,x in enumerate(idx):
            ilen = x[1][k]
            step = x[2][k]
            if ilen != 1:
                if k == 0:
                    if step == 1:
                        l.append( "%s"%(x[0]) )
                    else:
                        l.append( "%s*%s"%(x[0], str(step)) )
                else:
                  if step == 1:
                    l.append( "%s%%%s"%(x[0], str(ilen)) )
                  else:
                    l.append( "(%s%%%s)*%s"%(x[0], str(ilen), str(step)) )
                  r += ws + "%s /= %s;"%(x[0], str(ilen))
        if len(l) == 0:
            s = ws + "%s = 0;"%out
        else:
            s = ws + "%s = "%out + " + ".join(l) + ";"
        s += r
        blk.append(s)
    return blk

# Generate code to compute strides from
# input array shape.
def strides(pre, sh, off=0):
    if type(sh) == type(1): # old behavior
        n = sh
        sh = ["%s%d"%(pre,i) for i in range(n)]
        pre = "%s_stride"%pre
    else:
        n = len(sh)
    if n < 1:
        return ""
    s = "\n    const int %s%d = 1;"%(pre, off+n-1)
    for i in range(n-2,-1,-1):
        s += "\n    const int %(p)s%(i)d = %(shn)s * %(p)s%(ip1)d;" % {
                'p' : pre,
                'i' : off+i,
                'shn' : sh[i+1],
                'ip1' : off+i+1,
             }
    return s

# Generate code to compute strides of a padded array
# from the input array shape.
def pad_strides(pre, sp, off=0):
    n = len(sp)
    if(n < 1):
        return ""
    s = "    const int %s_pstride%d = 1;\n"%(pre, off+n-1)
    for i in range(n-2,-1,-1):
        s += "    const int %(p)s_pstride%(i)d = ((%(p)s%(ip1)d+%(pad)d-1)/%(pad)d) * %(pad)d * %(p)s_pstride%(ip1)d;\n" % {
                'p' : pre,
                'i' : off+i,
                'pad' : sp[i+1],
                'ip1' : off+i+1,
             }
    return s

def indent(nws, x):
    if nws < 0:
      def f(s):
        if len(s) < -nws:
            return s
        return s[-nws:]
    else:
      def f(s):
        #s = s.strip()
        if len(s) == 0:
            return s
        return " "*nws + s
    return "\n".join(map(f, x.split('\n')))

