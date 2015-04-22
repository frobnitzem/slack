#include <ast.h>
#include <stdarg.h>

struct Lexer_Context {
    void *scanner;   // the scanner state
    FILE *file;      // input stream
    int esc_depth;   // escaping depth
    int nest;        // nested ([{}]) level
    struct Environ *e;
    Ast *ret;        // return value
};

/* Internal functions */
void tce2_lex_ctor(struct Lexer_Context *this);
void tce2_lex_dtor(struct Lexer_Context *this);
int tce2_parse(struct Lexer_Context *);

/* uses batch parsing mode when ft == 1 */
Ast *tce2_parse_inp(struct Environ *, FILE *f);

