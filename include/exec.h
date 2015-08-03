#ifndef _EXEC_H
#define _EXEC_H

#include <stdint.h>

// This operates between blocks
//void *exec_ast(Ast *op, int n, void **arg);

#ifdef QUARK
#define exec_ast run_quark
Tensor *run_quark(Ast *a, int nthreads, MemSpace *mem, SMap *named);
int run_quark_n(int n, void **names, int nthreads, MemSpace *mem, SMap *named);
#else
#define exec_ast run_seq
Tensor *run_seq(Ast *a, int nthreads, MemSpace *mem, SMap *named);
#endif

struct Node {
    Ast *op;
    Tensor *val;
    int nref;
    int visited;
};

// returns a Map from (Ast *) to (struct Node *)
Map *zip_ast(Ast *a, struct Node **n, SMap *named);
Map *zip_ast_n(int n, Ast **a, struct Node **node, SMap *named);
struct Node *node_ctor(Ast *a, int nref);

/*
extern void magma_sgemm(
    int transA, int transB,
    int m, int n, int k,
    float alpha,
    const float *dA, int ldda,
    const float *dB, int lddb,
    float beta,
    float *dC, int lddc);

void magma_sger(
    int m, int n,
    float alpha,
    const float *dx, int incx,
    const float *dy, int incy,
    float *dA, int ldda);

void magma_sscal(
    int n,
    float alpha,
    float *dx, int incx);

void magma_saxpy(
    int n,
    float alpha,
    float *dx, int incx,
    float *dy, int incy);
*/

void dlarnv_(const int *, const int *, const int *, double *);

// C-style function interfaces with traditonal Fortran addressing.
// For definitions, see "magma-1.6.1/include/magmablas_s.h"
#define GEMM  magma_sgemm
#define GER   magma_sger
#define DOT   magma_sdot
#define COPY  magma_scopy
#define SCAL  magma_sscal
#define AXPY  magma_saxpy
#define LARNV dlarnv_

#endif
