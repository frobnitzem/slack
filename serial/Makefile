OBJS = print_dot.o print.o

../lib/serial.a:	$(OBJS)
	$(AR) -cr $@ $^

clean:
	rm -f $(OBJS)
