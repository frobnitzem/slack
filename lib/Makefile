OBJS=map.o smap.o slice.o quark/to_quark.o

lib.a:	$(OBJS)
	$(AR) -cr $@ $^

clean:
	rm -f $(OBJS)
