// stuff that almost got the parser to compile with g++-4.8

typedef union YYSTYPE {
	int i;
        uint8_t ui;
	double f;
	char *str; // unicode-aware, hence chars too
	struct active *a; // active parse - index + Ast
	Slice sl; // slice of indices
} YYSTYPE;

struct YYLTYPE;
#define YYSTYPE_IS_DECLARED 1
void slack_error(struct YYLTYPE *, struct Lexer_Context *, const char *s, ...);
int slack_lex(YYSTYPE *, struct YYLTYPE *, void *);

