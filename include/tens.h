// Included by exec.h

#include <stdint.h>
#include <memspace.h>

// Tensors are ordered, with contiguous dimension last.
// stride_i = prod_{j > i} shape[j]
typedef struct {
    uint32_t n;      // The logical dimension of the array.
    uint32_t len;    // prod(shape)
    double *x;         // Block of memory holding the array.
    int shape[0];    // A list of n integers, stating the
                     //    length along each dimension
} Tensor;

// Implemented in lib/tbb/memspace.cpp
/*
Tensor *tensor_ctor(int nd, int *shape);
void tensor_dtor(Tensor **t, void *m); // release the block or free the tensor

Tensor *mkTensor(int nd, int *shape, int nref, void *m);
Tensor *toTensor(double *x, int nd, int *shape, void *m);
*/

Tensor *tensor_ctor(const int nd, const int *shape);
void tensor_dtor(Tensor **t, MemSpace *mem);
Tensor *mkTensor(const int nd, const int *shape, const int nref, MemSpace *mem);
Tensor *toTensor(double *x, const int nd, const int *shape, MemSpace *mem);
Tensor *newTensor(const int nd, const int *shape,
                  const int nref, MemSpace *mem);
Tensor *uniq_tens(MemSpace *, Tensor *, const int nref);

//Tensor *tensdot(Tensor *a, Tensor *b, int nc, int *ind, int nref, Slice *m);
// These just wrap tensadd, tensdot, DSCAL
void tdot(const double alpha, Tensor *a, const uint8_t *pa,
                              Tensor *b, const uint8_t *pb,
          const double beta,  Tensor *c);
void tadd(const double alpha, Tensor *a,
          const double beta,  Tensor *b, const uint8_t *pb);
void tscale(const double, Tensor *);


// low-level Blas-like routines.
void tensdot(const double alpha,
             double *A, const int na, const int *sa, const uint8_t *pa,
             double *B, const int nb, const int *sb, const uint8_t *pb,
             const double beta,
             double *C, int nc);
void tensadd(const double alpha,
             double *A, const int n, const int *sa,
             const double beta,
             double *B, const uint8_t *perm);

void compose_permut(const int m, uint8_t *x, const int n, const uint8_t *perm);
uint8_t *inv_permut(const int n, const uint8_t *x);

