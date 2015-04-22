%pure_parser
%name_prefix="tce2_"
%locations
%error-verbose
%defines
%parse-param { struct Lexer_Context* context }
%lex-param { void* scanner  }

/* declare tokens */
%token <str> CHAR
%token <str> STRING
%token <i> INT
%token <f> FLOAT
%token EOL SUM

%type <sl> indices index indlist
%type <a> statements assign term factor contraction literal

%destructor { free ($$); } STRING

%{
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "parser.h"

#define scanner context->scanner
%}

%union {
	int i;
	double f;
	char *str; // unicode-aware, hence chars too
	Ast *a;
	Slice sl; // slice of indices
}

%%

start: statements { context->ret = $1; }

 /* Tensor contraction expression grammar. */

statements: /* empty */             {$$ = NULL; }
          | statements assign EOL   { $$  = mkCons($1, $2); }
          ;

 /* Named node - repurpose TRef returned by 'literal' */
assign: literal '=' term            { $1->ref->a = $3;
                                      $1->type = TNamed;
                                      $$ = $1;
                                    }
      ;

term: term '+' factor               { $$ = mkSum($1, $3); }
    | term '-' factor               { $$ = mkDiff($1, $3); }
    | factor                        { $$ = $1; }
    ;

factor: contraction                 { $$ = $1; }
      | literal                     { $$ = $1; }
      | '(' term ')'                { $$ = $2; }
      ;

contraction: SUM indices factor factor { $$ = mkDot($3, $4, $2); }
           ;

literal: STRING indices { $$ = mkRef($1, $2); slice_dtor(&$2); }
       | CHAR   indices { $$ = mkRef($1, $2); slice_dtor(&$2); }
       ;

 /* always order subscripts first */
indices: '_' indlist '^' indlist { $$ = slice_append($2, $4->x, $4->n); };
       | '^' indlist '_' indlist { $$ = slice_append($4, $2->x, $2->n); };
       | '_' indlist { $$ = $2; };
       | '^' indlist { $$ = $2; };
       ;

 /* un-nest a bracketed list */
indlist: /* empty */ { $$ = slice_ctor(1, 0, 4); }
       | indlist index { uint8_t ind = $2;
                         $$ = slice_append($1, &ind, 1);
                       }
       | indlist '{' indlist '}' { $$ = slice_append($1, $3->x, $3->n); };
       ;

/* lookup the new index from the context, add if not found */
index: CHAR {  uint8_t ind = (uint8_t) smap_get(context->e->ind_map, $1);
               if(!ind) {
                   if(context->e->nindices >= 255) {
                       tce2_error(yylloc, NULL, "Overflowed 255 available indices!\n");
                       exit(1); // as if that wasn't fatal enough!
                   }
                   ind = ++context->e->nindices;
                   smap_put(context->e->ind_map, $1, (void *)ind);
               }
               $$ = (int) ind;
            }
       ;


%%

