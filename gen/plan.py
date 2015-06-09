# Plan a parallel copy using n workers into output shape s.
# The algorithm requires prod(s) to be a multiple of n and
# works by matching factors from n with those of s,
# with preference to the left.  This means as many
# workers as possible for the most sig. dimensions,
# each doing as many copies as possible on the least sig. ones.
#
# The output is a pair of shapes, with the same length as s:
#   index_shape -- outer loops, used to decode the worker starting index
#   copy_shape  -- shape copied by each worker
#
#   prod(index_shape) = n
#   index_shape * copy_shape = s

prod = lambda x: reduce(lambda a,b: a*b, x, 1)

def divide_work(s, n):
    sz = prod(s)
    if n > sz:
        raise ValueError, "Have too many workers."
    if sz % n != 0:
        raise ValueError, "Workers don't evenly divide number of copies."

    f = factor(n) # Map (prime factors) (multiplicity)

    index = [1 for i in s]
    copy = [i for i in s]
    for i in range(len(s)):
        for x in factors(s[i]):
            try:
                if f[x] > 0: # parallelize this one
                    copy[i] /= x # fewer copies
                    index[i] *= x # more workers
                    f[x] -= 1
            except KeyError:
                pass
    if any(v != 0 for k,v in f.iteritems()):
        raise ValueError, "Internal Error! Leftover workers (factors = %s)"%(str(f))

    return index, copy

def factors(n):
    j = 2
    while j <= n/2:
        if n%j == 0:
            yield j
            n /= j
        else:
            j += 1
    yield n

def factor(n):
    f = {}
    for x in factors(n):
        try:
            f[x] += 1
        except KeyError:
            f[x] = 1
    return f

def test():
    for n in range(1, 10):
        print n, [i for i in factors(n)]
    print plan_copy((4,4,9), 2*3)

