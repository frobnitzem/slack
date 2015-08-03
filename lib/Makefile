OBJS=map.o smap.o slice.o
ifdef QUARK
OBJS += quark/to_quark.o
endif

lib.a:	$(OBJS)
	$(AR) -cr $@ $^

clean:
	rm -f $(OBJS)
