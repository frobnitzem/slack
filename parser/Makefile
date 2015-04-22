OBJS=lexer.lex.o parser.tab.o parse.o

../lib/parse.a: $(OBJS)
	$(AR) -cr $@ $^

parser.tab.c: parser.y
	#bison --defines=parser.tab.h -d parser.y
	bison -d parser.y

lexer.lex.c: lexer.l parser.tab.c
	flex -d --header-file=lexer.h -o lexer.lex.c lexer.l

clean:
	rm -f $(OBJS) lexer.{h,lex.c} parser.tab.{c,h}
