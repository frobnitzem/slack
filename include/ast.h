/*   A simple Ast encoding tensor math expressions.
 */

#ifndef _AST_H
#define _AST_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <lib/slice.h>
#include <lib/smap.h>

typedef struct Ast_s Ast;

// Tuninit is just for uncopied objecs during GC passes.
enum AstType { Tuninit=0, TNamed=1, TSum=3, TDiff=4, TDot=5, TRef=6, 
               TCons=7, } ;

/* Ast Types */
struct Pair {
    Ast *a;
    Ast *b;
};
// Name is informational, indices may represent transpose
struct Named {
    Ast *a; // NULL if pure ref.
    char *name; // points somewhere into end of ind
    int n; // number of indices
    uint8_t ind[0];
};
struct Dot {
    Ast *a, *b;
    int n; // number of indices
    uint8_t ind[0]; // ind represents the indices to sum over
};

struct Ast_s {
    enum AstType type;
    uint32_t len;
    union {
        struct Named ref[0];
        struct Pair  pair[0];
        struct Dot   dot[0];
    };
};
#define AST_HDR_SIZE    (sizeof(Ast))
#define PAIR_SIZE       (AST_HDR_SIZE + sizeof(struct Pair))
#define DOT_SIZE(n)     (AST_HDR_SIZE + sizeof(struct Dot) + n)
#define REF_SIZE(n,m) (AST_HDR_SIZE + sizeof(struct Named) + n+m)

struct Environ {
    int debuglevel;
    int nindices; // index serial number - can't exceed 255!
    SMap *ind_map; // index lookup
};

Ast *mkCons(Ast *car, Ast *cdr);
Ast *mkNamed(Ast *ref, Ast *term);
Ast *mkSum(Ast *, Ast *);
Ast *mkDiff(Ast *a, Ast *b); // a - b
Ast *mkDot(Ast *a, Ast *b, Slice ind);
Ast *mkRef(char *name, Slice ind); // slice of uint8_t

int ast_children(Ast *a, Ast ***t);

char *estrdup(char *);

#include "serial.h"

#endif
