// Included from ast.h

#ifndef _TCE2_SERIAL
#define _TCE2_SERIAL

void ast_to_dot(FILE *f, Ast *a);
int show_ast(char *name, Ast *a, int waitfor);
void write_ast_label(FILE *f, Ast *a);

#endif
