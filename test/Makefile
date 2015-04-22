TESTS = test_ast test_gc test_parse test_dot test_eval test_dir test_serial

test.sh:	run_test.sh $(TESTS)
	@( echo "#!/bin/sh" >test.sh; \
	   echo "cd $(PWD)/test" >>test.sh; \
	   echo "TESTS=\"$(TESTS)\"" >>test.sh; \
	   cat run_test.sh >>test.sh; \
	)

test_ast:	test_ast.o $(LIB)
	$(LD) -o $@ $^ $(LDFLAGS)

test_gc:	test_gc.o $(LIB)
	$(LD) -o $@ $^ $(LDFLAGS)

test_parse:	test_parse.o $(LIB)
	$(LD) -o $@ $^ $(LDFLAGS)

test_dot:	test_dot.o $(LIB)
	$(LD) -o $@ $^ $(LDFLAGS)

test_eval:	test_eval.o $(LIB)
	$(LD) -o $@ $^ $(LDFLAGS)

test_dir:	test_dir.o $(LIB)
	$(LD) -o $@ $^ $(LDFLAGS)

test_serial:	test_serial.o $(LIB)
	$(LD) -o $@ $^ $(LDFLAGS)

clean:
	for test in $(TESTS); do \
	    rm -f $$test.o; \
	done;
