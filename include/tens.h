// Included by exec.h

#include <stdint.h>

// Tensors are ordered, with contiguous dimension last.
// stride_i = prod_{j > i} shape[j]
typedef struct {
    uint32_t n;      // The logical dimension of the array.
    uint32_t len;    // prod(shape)
    void *x;         // Block of memory holding the array.
    int shape[0];    // A list of n integers, stating the
                     //    length along each dimension
} Tensor;

Tensor *tensor_ctor(int nd, int *shape);
void tensor_dtor(Tensor **t, void *m); // release the block or free the tensor

Tensor *mkTensor(int nd, int *shape, int nref, void *m);
Tensor *toTensor(double *x, int nd, int *shape, void *m);

//Tensor *tensdot(Tensor *a, Tensor *b, int nc, int *ind, int nref, Slice *m);

