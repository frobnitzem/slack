/*   A simple Ast encoding tensor math expressions.
 */

#ifndef _AST_H
#define _AST_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <lib/slice.h>
#include <lib/smap.h>

// various structures used during execution
#include "exec.h"

typedef struct Ast_s Ast;

// Tuninit is just for uncopied objecs during GC passes.
enum AstType { Tuninit=0, TRef=1, TBase=2, TTranspose=3,
               TSum=4, TDiff=5, TDot=6, TScale=7, TReduce=8 } ;

enum BaseType { TTens=1 };

/* Ast Types */
struct Pair {
    Ast *a;
    Ast *b;
};
struct Dot {
    Ast *a, *b;
    int n; // number of indices
    uint8_t ind[0]; // ind represents the indices to sum over (n x 2)
};
struct Reduce { // sum over indices of the tensor
    Ast *a;
    int n;
    uint8_t ind[0]; // ind represents the indices to sum over
};
struct Scale {
    Ast *a;
    float s;
};
struct Transpose {
    Ast *a;
    int n; // total number of indices
    uint8_t perm[0];
};
struct Base {
    enum BaseType type;
    union {
        Tensor *t;
    };
};

struct Ast_s {
    enum AstType type;
    uint32_t len;
    union {
        struct Base   base[0];
        struct Pair   pair[0];
        struct Dot    dot[0];
        struct Scale  scale[0];
        struct Reduce reduce[0];
        struct Transpose t[0];
        char   ref[0];
    };
};
#define AST_HDR_SIZE    (sizeof(Ast))
#define BASE_SIZE       (AST_HDR_SIZE + sizeof(struct Base))
#define PAIR_SIZE       (AST_HDR_SIZE + sizeof(struct Pair))
#define DOT_SIZE(n)     (AST_HDR_SIZE + sizeof(struct Dot) + n)
#define REF_SIZE(n)     (AST_HDR_SIZE + n)
#define T_SIZE(n)       (AST_HDR_SIZE + sizeof(struct Transpose) + n)
#define SCALE_SIZE      (sizeof(Ast) + sizeof(struct Scale))
#define REDUCE_SIZE     (sizeof(Ast) + sizeof(struct Reduce))

struct Environ {
    int debuglevel;
};

Ast *mkScale(Ast *, float);
Ast *mkNamed(Ast *ref, Ast *term);
Ast *mkSum(Ast *, Ast *);
Ast *mkDiff(Ast *a, Ast *b); // a - b
Ast *mkDot(Ast *a, Ast *b, Slice ind);
Ast *mkRef(char *name);
Ast *mkTranspose(Ast *, int n, uint8_t *perm);
Ast *mkLit(Tensor *);

int ast_children(Ast *a, Ast ***t);

char *estrdup(char *);

#include "serial.h"

#endif
