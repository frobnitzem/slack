from generate import *

# Index over symbolic shape names
# i.e. outer loops and initial offsets
class TensIdx_sym:
    def __init__(self, name, shape):
        self.n = len(shape)
        self.name = name # symbolic name
        self.shape = shape # shape strings

    def __getitem__(self, I): # computes linear index
        if len(I) != self.n:
            raise ValueError, "Wrong index shape: idx=%s, sh = %s"%(
                                    str(I), str(self.shape))

        def txt(a):
            if type(a) == type(1):
                return "%2d"%a
            return "%2s"%(str(a))
        if all(x == 0 for x in I):
            return "0"
        if len(I) == 1:
            return txt(I[0])

        def skipi(i,x):
            if i == self.n-1: # last stride = 1
                return txt(x)
            if x == 1:
                return "s%s_stride%d"%(self.name, i)
            return "%s*s%s_stride%d"%(txt(x), self.name, i)
        return " + ".join(skipi(i,x) for i,x in enumerate(I) if x != 0)

    def strides(self): # code to compute strides
        return strides("s%s_stride"%self.name, self.shape)

# Index over fixed integer shapes (e.g. nested inner indices)
class TensIdx:
    def __init__(self, shape):
        self.n = len(shape)
        self.shape = shape # shape ints
        self.sz = prod(self.shape) # size

    def __len__(self):
        return self.sz

    def __getitem__(self, I):
        if len(I) != self.n:
            raise ValueError, "Wrong index shape: idx=%s, sh = %s"%(
                                    str(I), str(self.shape))
        return lin_index(I, self.shape)

    def __iter__(self): # loop over all indices
        d = len(self.shape)
        n = [prod(self.shape[i:]) for i in range(d+1)]
        x = [0]*d
        I = 0
        while 1:
            yield x
            i = d
            while (I+1)%n[i] == 0:
                if i == 0:
                    raise StopIteration
                x[i-1] = (x[i-1] + 1)%self.shape[i-1]
                i -= 1
            I += 1

    # compute the index from an integer linear index
    def from_lin(self, i):
        return get_index(i, self.shape)

    # generate code to convert from a symbolic index to an integer index
    def to_shape(self, x, sh):
        return re_index(x, self.shape, sh)

def test():
    na = 3
    ai = TensIdx_sym("A", ["sA%d"%i for i in range(na)])
    print ai.strides()
    print ai[1,2,1]

    blk = TensIdx([12,9,5])
    print blk.from_lin(2*9*5+4*5+3)
    print 2*9*5+4*5+3 == blk[2,4,3]
    print blk.to_shape(2*9*5+4*5+3, [100,100,100])

