/* Copyright David M. Rogers
 *
 * This source file is released into the public domain.
 * Please use these index tricks and stop re-writing gemm.
 *
 * Warning: This tensor inner product is for timing comparison purposes only.
 *
 */
#include <stdio.h>
#include <stdlib.h>

#define BINARY_OP(a,b) ((a) * (b))

int szprod(int *x, int n);
void print_vec(int *x, int n);
void print_nvec(char *, int *x, int n);

/* Computes xGEMM-like:
 * C[i] = alpha (sum_{ctr} A[f(i)] B[g(i)]) + beta C[i]
 * where C has no overlap in memory space with A or B
 *
 * where the basic step can at least take part in a fused-multiply-add
 * na : dimension of A
 * sa : shape of A (length na, prod(sa) = len(A))
 * pa : P_a below
 *
 * Index manipulations:
 *   given: P_a - a permutation from A's indices to "output" indices
 *          P_b - a permutation from B's indices to "output" indices
 *          n_c - the dimension of the output tensor
 *
 *     I. Contraction happens over the last n "output" dimensions.
 *
 *    II. real output size is Prod_{j | P_a[j] < n} Ashape[j]
 *                       * Prod_{k | P_b[k] < n} Bshape[k]
 *             this is the outer loop
 *        sum size is all else
 *             this is the inner loop
 *
 *   III. the loops keep track of indices f(i), g(i) by using
 *        integer vectors of len n_a and n_b, which increment 
 *        when i passes output dimension barriers,
 *        Cshape[n_c-1], Cshape[n_c-1]*Cshape[n_c-2], ...
 *        f(i) = sum_{j=0,n_a-1} Aind[j]*Astride[j]
 *
 *   IV. There are obviously lots of things we could do by wrapping
 *       this function, like combine adjacent (matched) indices,
 *       re-order summation indices, and unroll inner loops
 *       into small-sized gemm-s.
 */

void tensdot(const double alpha,
             double *A, const int na, const int *sa, const uint8_t *pa,
             double *B, const int nb, const int *sb, const uint8_t *pb,
             const double beta,
             double *C, int nc) {
    if(na > 100 || nb > 100 || nc > na+nb || na < 1 || nb < 1 || nc < 0
            || (na+nb-nc)%2 == 1) {
        printf("whoa there!\n");
        return;
    }

    double acc;
    int i, j, k, I, J, K;
    int outer, inner;
    int n = (na+nb-nc)/2;   // number of contraction indices (nc =na + nb - 2*n)
    int ntot = na + nb - n; // number of "expanded output" indices
                        // run-time index
    int *Aind = malloc(sizeof(int)*(2*(na+nb) + ntot + nc)); // na
                        // reverse permutations
    int *ain  = Aind + na;              // n      : perm from inner to A index
    int *aout = ain + n;                // na - n : perm from outer to A index

                        // run-time index
    int *Bind = Aind + 2*na;            // nb
                        // reverse permutations
    int *bin  = Bind + nb;              // n      : perm from inner to B index
    int *bout = bin  + n;               // na - n : perm from outer to B index

                        // 'I' / 'J' boundary locations (derived from sc)
    int *Aoutb = Bind + 2*nb;     // na - n : I-boundaries for outer A indices
    int *Boutb = Aoutb + na-n;    // nb - n : I-boundaries for outer B indices

    int *sc    = Aoutb + nc;      // ntot : strides in C
    int *inb   = sc + nc;   // n : J-boundaries for inner A/B ind.
                            // (just refers to last part of sc)
                            // last 2 values are ignored if inner
                            // loop is rolled

    // Index pass 0 - check
    for(i=0; i<ntot; i++) sc[i] = -1;

    // Index pass 1 - assign c-size, reverse permutations, and zero Aind/Bind
    j=0; // index to aout / bout
    for(i=0; i<na; i++) {
        if(pa[i] < 0 || pa[i] >= ntot) {
            printf("A index out of range (%d)\n", pa[i]);
            free(Aind);
            return;
        }
        if(sc[pa[i]] != -1) {
            printf("Duplicate index from A (%d)\n", pa[i]);
            free(Aind);
            return;
        }
        sc[pa[i]] = sa[i];

        if(pa[i] < nc) {
            aout[j++] = i;
        } else {
            ain[pa[i]-nc] = i;
        }
        Aind[i] = 0;
    }
    j=0; // index to aout / bout
    for(i=0; i<nb; i++) {
        if(pb[i] < 0 || pb[i] >= ntot) {
            printf("B index out of range (%d)\n", pb[i]);
            free(Aind);
            return;
        }
        if(pb[i] < nc) {
            if(sc[pb[i]] != -1) {
                printf("Duplicate uncontracted index (%d)\n", pb[i]);
                free(Aind);
                return;
            }
            sc[pb[i]] = sb[i];
            bout[j++] = i;
        } else {
            if(sc[pb[i]] != sb[i]) {
                printf("Summation index %d doesn't have same "
                   "size in A (%d) and B (%d)\n", pb[i], sb[i], sc[pb[i]]);
                free(Aind);
                return;
            }
            bin[pb[i]-nc] = i;
        }
        Bind[i] = 0;
    }
    /*
    printf("A : %d, B : %d, output : %d (%d | %d)\n", na, nb, ntot, nc, n);
    printf("A : "); print_vec(aout, na-n);
                   printf(" | "); print_vec(ain, n); printf("\n");
    printf("B : "); print_vec(bout, nb-n);
                   printf(" | "); print_vec(bin, n); printf("\n");
    print_nvec("sc =", sc, ntot);
    */

    // form cumulative produce of c-shape (find boundaries)
    outer = szprod(sc,   nc);
    if(n > 0) // don't count last (K) index in inner boundaries
        inner = szprod(inb,  n-1);

    // Index pass 2 - assign inner and outer boundaries from sc
    for(i=0; i<na-n; i++) {
        Aoutb[i] = sc[pa[aout[i]]];
    }
    for(i=0; i<nb-n; i++) {
        Boutb[i] = sc[pb[bout[i]]];
    }
    //print_nvec("Aoutb =", Aoutb, na-n);
    //print_nvec("Boutb =", Boutb, nb-n);

    if(n > 0) { // rolled inner loop
      int Aistep = 1, Bistep = 1;
      int inner2 = sc[ntot-1]; // inner loop size (larger is better)

      // determine inner-loop step values (smaller is better)
      for(i=ain[n-1]+1; i<na; i++) Aistep *= sa[i];
      for(i=bin[n-1]+1; i<nb; i++) Bistep *= sb[i];

      for(I=0; I<outer; I++) {
        acc = 0.0;
        for(J=0; J<inner; J++) {
            // Track lowest index inner loop separately.
            // (and leave corresponding Aind / Bind at 0)
            j = Aind[0];
            for(i=1; i<na; i++) // Horner
                j = j*sa[i] + Aind[i];
            k = Bind[0];
            for(i=1; i<nb; i++) // Horner
                k = k*sb[i] + Bind[i];

            for(K=0; K<inner2; K++) {
                //printf("  "); print_vec(Aind, na); printf(" = %d ", j);
                //print_vec(Bind, nb); printf(" = %d\n", k);
                acc += BINARY_OP(A[j], B[k]);
                j += Aistep; k += Bistep;
            }

            // ck what boundaries next J-step hits
            for(i=0; i<n-1; i++) { // loop over inner indices
                Aind[ain[i]] = (Aind[ain[i]] + !((J+1)%inb[i]))%sa[ain[i]];
                Bind[bin[i]] = (Bind[bin[i]] + !((J+1)%inb[i]))%sb[bin[i]];
            }
        }
        //printf("Output %d = %f\n", I, acc);
        C[I] = alpha*acc + beta*C[I];

        // ck what boundaries next I-step hits
        for(i=0; i<na-n; i++) { // outer indices assoc. to A
            Aind[aout[i]] = (Aind[aout[i]] + !((I+1)%Aoutb[i]))%sa[aout[i]];
        }
        for(i=0; i<nb-n; i++) { // outer indices assoc. to B
            Bind[bout[i]] = (Bind[bout[i]] + !((I+1)%Boutb[i]))%sb[bout[i]];
        }
      }
    } else { // lots of GER
      for(I=0; I<outer; I++) {
        j = Aind[0];
        for(i=1; i<na; i++) // Horner
            j = j*sa[i] + Aind[i];
        k = Bind[0];
        for(i=1; i<nb; i++) // Horner
            k = k*sb[i] + Bind[i];

        C[I] = alpha*BINARY_OP(A[j], B[k]) + beta*C[I];

        // ck what boundaries next I-step hits
        for(i=0; i<na-n; i++) { // outer indices assoc. to A
            Aind[aout[i]] = (Aind[aout[i]] + !((I+1)%Aoutb[i]))%sa[aout[i]];
        }
        for(i=0; i<nb-n; i++) { // outer indices assoc. to B
            Bind[bout[i]] = (Bind[bout[i]] + !((I+1)%Boutb[i]))%sb[bout[i]];
        }
      }
    }
    free(Aind);
}

// Form strides from lengths of dimensions.
int szprod(int *x, int n) {
    int i, j, k = 1; // last stride is 1 (C-ordering)

    for(i=n-1; i>=0; i--) {
        j = x[i];
        x[i] = k;
        k *= j;
    }
    return k; // total sz.
}

void print_nvec(char *str, int *x, int n) {
    int i;
    if(n == 0) {
        printf("%s ()\n", str);
        return;
    } else {
        printf("%s (", str);
    }

    for(i=0; i<n-1; i++) {
        printf("%d, ", x[i]);
    }
    printf("%d)\n", x[i]);
}

void print_vec(int *x, int n) {
    int i;
    if(n == 0) {
        printf("()");
        return;
    } else {
        printf("(");
    }

    for(i=0; i<n-1; i++) {
        printf("%d, ", x[i]);
    }
    printf("%d)", x[i]);
}
