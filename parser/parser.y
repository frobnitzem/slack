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

%type <sl> indices ck_indices ind indlist
%type <a>  assign term factor mfactor contraction literal
%type <ui> index
%type <f>  expr

%destructor { free ($$); } STRING

%start statements

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
        uint8_t ui;
	double f;
	char *str; // unicode-aware, hence chars too
	struct active *a; // active parse - index + Ast
	Slice sl; // slice of indices
}

%%

 /* Tensor contraction expression grammar. */

statements: /* empty */             { }
          | eols                    { }
          | statements assign eols  { }
          ;

eols: eols EOL
    | EOL
    ;

 /* Named node */
assign: literal '=' term            { if(smap_put(context->ret, 
                                                  $1->a->ref,
                                                  ck_transpose($3, $1->ind))
                                         != NULL) {
          fprintf(stderr, "Warning! re-defining %s.\n", $1->a->ref);
                                      }
                                      act_dtor($1);
                                      act_dtor($3);
                                    }
      ;

expr: INT                            { $$ = (float)$1; }
    | FLOAT                          { $$ = $1; }
    ;



 /* Use the order from the first arg. to +/- */
term: term '+' mfactor               { $1->a = mkSum($1->a,
                                                     ck_transpose($3, $1->ind));
                                       $$ = $1;
                                       act_dtor($3);
                                     }
    | term '-' mfactor               { $1->a = mkDiff($1->a,
                                                  ck_transpose($3, $1->ind));
                                       $$ = $1;
                                       act_dtor($3);
                                     }
    | mfactor                        { $$ = $1; }
    ;

mfactor: expr factor { $2->a = mkScale($2->a, $1); $$ = $2; }
       | factor { $$ = $1; }
       ;

factor: contraction                 { $$ = $1; }
      | literal                     { $$ = $1; }
      | '(' term ')'                { $$ = $2; }
      ;

contraction: SUM indices factor factor {
               Slice act; // [index codes]
               Slice ctr; // [(dim,dim)]
               if(partition_inds(&act, &ctr, $2, $3->ind, $4->ind) < 0) {
                   tce2_error(&yylloc, NULL, "Bad contraction!\n");
                   exit(1); // as if that wasn't fatal enough!
               }
               $$ = mkActive(mkDot($3->a, $4->a, ctr), act);
               act_dtor($3); act_dtor($4);
             }
           ;

literal: STRING indices { $$ = mkActive(mkRef($1), $2); }
       | CHAR   indices { $$ = mkActive(mkRef($1), $2); }
       ;

 /* copy-rule to allow end-of-parse checking */
indices: ck_indices { if(ck_duplicate($1)) {
                        tce2_error(&yylloc, NULL, "Duplicate indices!\n");
                        exit(1); // as if that wasn't fatal enough!
                      }
                      $$ = $1;
                    }
       ;

 /* always order superscripts first */
ck_indices: '^' ind '_' ind { $$ = slice_append($2, $4->x, $4->n); }
          | '_' ind '^' ind { $$ = slice_append($4, $2->x, $2->n); }
          | '_' ind { $$ = $2; }
          | '^' ind { $$ = $2; }
          ;

ind: index { $$ = slice_ctor(1, 1, 1); *(uint8_t *)$$->x = $1; }
   | '{' indlist '}' { $$ = $2; }

 /* un-nest a bracketed list */
indlist: /* empty */ { $$ = slice_ctor(1, 0, 4); }
       | indlist index { $$ = slice_append($1, &$2, 1); }
       | indlist '{' indlist '}' { $$ = slice_append($1, $3->x, $3->n); }
       ;

/* lookup the new index from the context, add if not found */
index: CHAR {  uint8_t ind = (uint8_t) smap_get(context->ind_map, $1);
               if(!ind) {
                   if(context->nindices >= 255) {
                       tce2_error(&yylloc, NULL, "Overflowed 255 available indices!\n");
                       exit(1); // as if that wasn't fatal enough!
                   }
                   ind = ++context->nindices;
                   smap_put(context->ind_map, $1, (void *)ind);
               }
               $$ = ind;
            }
       ;


%%

