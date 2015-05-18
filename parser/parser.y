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
%token EOL SUM RAND

%type <sl> indices ck_indices ind indlist intlist
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

expr: INT                            { $$ = (double)$1; }
    | FLOAT                          { $$ = $1; }
    ;



 /* Use the order from the first arg. to +/- */
term: term '+' mfactor               { int ord;
                                       uint8_t *perm= get_perm($3->ind,$1->ind,
                                                                &ord);
                                       if(perm == NULL) {
                   tce2_error(&yylloc, NULL, "Indices in '+' don't match!\n");
                   exit(1);
                                       }
                                       $1->a = simpAdd($1->scale, $1->a,
                                                     $3->scale, $3->a,
                                                     $1->ind->n, perm);
                                       free(perm);
                                       $$ = $1;
                                       $$->scale = 1.0;
                                       act_dtor($3);
                                     }
    | term '-' mfactor               { int ord;
                                       uint8_t *perm= get_perm($3->ind,$1->ind,
                                                                &ord);
                                       if(perm == NULL) {
                   tce2_error(&yylloc, NULL, "Indices in '-' don't match!\n");
                   exit(1);
                                       }
                                       $1->a = simpAdd($1->scale, $1->a,
                                               (-1.)*$3->scale, $3->a,
                                                     $1->ind->n, perm);
                                       free(perm);
                                       $$ = $1;
                                       $$->scale = 1.0;
                                       act_dtor($3);
                                     }
    | mfactor                        { $$ = $1; }
    ;

mfactor: expr factor { $2->scale *= $1; $$ = $2; }
       | factor { $$ = $1; }
       ;

factor: contraction                 { $$ = $1; }
      | literal                     { $$ = $1; }
      | RAND indices '(' intlist ')'        { $$ = mkActive(mkRand($4), $2); }
      | RAND indices '(' intlist INT ')'    { $$ = mkActive(
                    mkRand(slice_append($4, &$5, 1)), $2); }
      | '(' term ')'                { $$ = $2; }
      ;

intlist: INT ','                    { $$ = slice_ctor(sizeof(int), 1, 4);
                                      *(int *)$$->x = $1; }
       | intlist INT ','            { $$ = slice_append($1, &$2, 1); }

contraction: SUM indices factor factor {
               Slice act; // [index codes]
               Slice ctr; // [(dim,dim)]
               if(partition_inds(&act, &ctr, $2, $3->ind, $4->ind) < 0) {
                   tce2_error(&yylloc, NULL, "Bad contraction!\n");
                   exit(1); // as if that wasn't fatal enough!
               }
               $$ = mkActive(mkTensDot($3->scale*$4->scale,
                                       $3->a, $3->ind->n,
                                       $4->a, $4->ind->n, ctr),
                             act);
               slice_dtor(&ctr);
               act_dtor($3); act_dtor($4);
             }
           ;

literal: STRING indices { $$ = mkActive(mkRef($1), $2); }
       | CHAR   indices { $$ = mkActive(mkRef($1), $2); }
       ;

 /* copy-rule to allow end-of-parse checking */
indices: ck_indices { if(ck_duplicate($1)) {
                        tce2_error(&yylloc, NULL, "Duplicate indices!\n");
                        exit(1);
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

