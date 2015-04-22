#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "parser.h"
#include "parser.tab.h"
#include "lexer.h"

Ast *tce2_parse_inp(struct Environ *e, FILE *f) {
    struct Lexer_Context ctxt = {
        .file = f,
        .esc_depth = 0,
        .nest = 0,
        .ret = NULL,
        .e = e,
    };
    Ast *ret;
    int r;

    tce2_lex_ctor(&ctxt);

    tce2_parse(&ctxt); // nonzero on parse error
    ret = ctxt.ret;

    tce2_lex_dtor(&ctxt);

    return ret;
}

