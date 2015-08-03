#include "ast.h"
#include <stdarg.h>

struct Lexer_Context {
    void *scanner;   // the scanner state
    FILE *file;      // input stream
    int esc_depth;   // escaping depth
    int nest;        // nested ([{}]) level
    struct Environ *e;
    SMap *ind_map;   // index lookup
    int nindices;    // index serial number - can't exceed 255!
    SMap *ret;       // return value -- map of (name, Ast *)
};

// An active parse holds both an Ast object along with
// a collection of "active" index labels and a scale.
// The latter are consulted to provide permutations
// and alpha / beta whenever a binary operation is encountered.
struct active {
    Ast *a;
    Slice ind;
    double scale;
};

/* Mucking about with explicit indices. */
struct active *mkActive(Ast *a, Slice ind);
void act_dtor(struct active *act);
Ast *ck_transpose(struct active *act, Slice ind);
uint8_t *get_perm(Slice ind, Slice out, int *is_ord);
int partition_inds(Slice *cc_p, Slice *ctr_p, Slice csum, Slice ca, Slice cb);
int ck_duplicate(Slice a);

/* Internal functions */
void slack_lex_ctor(struct Lexer_Context *);
void slack_lex_dtor(struct Lexer_Context *);
int slack_parse(struct Lexer_Context *);

/* uses batch parsing mode when ft == 1 */
SMap *slack_parse_inp(struct Environ *, FILE *f);

