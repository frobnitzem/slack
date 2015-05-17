OBJS=ast.o tens.o tdot.o tadd.o memspace.o

../lib/exec.a:	$(OBJS)
	$(AR) -cr $@ $^

test: tdot.so test_tdot.py
	python test_tdot.py
	python test_tadd.py

tdot.so: tdot.c tadd.c
	gcc -shared -o $@ $^

clean:
	rm -f $(OBJS)
