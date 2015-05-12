#ifndef _EXEC_H
#define _EXEC_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <lib/slice.h>
#include <lib/smap.h>

#include "tens.h"

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

// C-style function interfaces with traditonal Fortran addressing.
// For definitions, see "magma-1.6.1/include/magmablas_s.h"
#define GEMM  magma_sgemm
#define GER   magma_sger
#define DOT   magma_sdot
#define COPY  magma_scopy
#define SCAL  magma_sscal
#define AXPY  magma_saxpy

#endif
