/* Copyright David M. Rogers
 *
 * This source file is released into the public domain.
 *
 * Warning: This tensor addition is for timing comparison purposes only.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BINARY_OP(a,b) ((a) * (b))

int szprod(int *x, int n);
void print_vec(int *x, int n);
void print_nvec(char *, int *x, int n);

/* Computes AXPY-like:
 * A[i] = alpha * A[i] + beta * B[f(i)]
 *
 * n : dimension of A & B
 * sa : shape of A (length n, prod(sa) = len(A))
 *
 * Note A and B must match, so: nb = na = n, and sa[pb[i]] = sb[i].
 *
 * Index manipulations:
 *   given: pb - a permutation from B's indices to output (A's) indices
 *
 *   III. the loops keep track of indices f(i) by using
 *        integer vectors of len n, which increment 
 *        when i passes output dimension barriers,
 *        Ashape[n_a-1], Ashape[a_c-1]*Ashape[n_a-2], ...
 *        f(i) = sum_{j=0,n_b-1} Bind[j]*Bstride[j]
 *
 *   IV. There are obviously lots of things we could do by wrapping
 *       this function, like combine adjacent (matched) indices,
 *       re-order summation indices, and unroll inner loops
 *       into small-sized gemm-s.
 */

void tensadd(const double alpha,
             double *A, const int n, const int *sa,
             const double beta,
             double *B, const uint8_t *pb) {
    if(n > 100 || n < 1) {
        printf("whoa there!\n");
        return;
    }

    double acc;
    int i, k, I, J, K;
    int outer, outer2, Bistep;
    int lind;
    int *Bind  = malloc(sizeof(int)*(3*n)); // n
    int *sb    = Bind + n;                  // n : B shape
    int *bound = Bind + 2*n;                // n : B boundaries

    // Index pass 0 - check
    for(i=0; i<n; i++) {
        if(pb[i] < 0 || pb[i] >= n) {
            printf("Permutation index out of range (%d)\n", pb[i]);
            free(Bind);
            return;
        }
        Bind[i] = -1; // for checking
    }

    // Index pass 1 - assign b shape and zero Bind
    for(i=0; i<n; i++) {
        if(Bind[pb[i]] != -1) {
            printf("Duplicate index (%d -> %d) in B permutation.\n", i, pb[i]);
            free(Bind);
            return;
        }
        Bind[pb[i]] = 0;
        sb[i] = sa[pb[i]];
        bound[i] = sa[i];
        if(pb[i] == n-1)
            lind = i; // index of B mapping to last index of A
    }

    // Form cumulative product of a-shape (boundaries)
    outer = szprod(bound, n-1);
    outer2 = sa[n-1];

    // Use cumulative product of b-shape to speed index computation.
    szprod(sb, n);

    // Inner-loop step value (smaller is better).
    Bistep = sb[lind];
    /*print_nvec("bound =", bound, n-1);
    print_nvec("sb =", sb, n);
    printf("ldim = %d, istep = %d\n", outer2, Bistep);*/

    for(I=0; I<outer; I++) {
        // Track lowest index inner loop separately.
        // (leaving corresponding Bind at 0)
        k = 0;
        for(i=0; i<n; i++)
            k += Bind[i]*sb[i];

        acc = 0.0;
        J = I*outer2;
        for(K=0; K<outer2; K++) {
            //printf("  "); print_vec(Aind, na); printf(" = %d ", j);
            //print_vec(Bind, nb); printf(" = %d\n", k);
            A[J+K] = alpha*A[J+K] + beta*B[k];
            k += Bistep;
        }

        // Ck. what boundaries next I-step hits.
        // Note: this could be optimized by pre-permuting bound, sa.
        for(i=0; i<n; i++) { // loop over inner indices
            Bind[i] = (Bind[i] + !((I+1)%bound[pb[i]]))%sa[pb[i]];
        }
        Bind[lind] = 0;
    }

    free(Bind);
}

