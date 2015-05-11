# Make your make do the work.
# http://www.gnu.org/software/make/manual/make.html#Advanced
# http://okmij.org/ftp/Computation/Make-functional.txt
#
#   Admittedly, this is sloppy, since it references path names
# inside the make system.  However, it accurately finds
# all the dependencies and recompiles on source changes,
# which was the original idea anyway.

CC=gcc
LD=gcc
#CFLAGS=-ggdb -I$(PWD)/include
CFLAGS=-ggdb -I/sw/include -I$(PWD)/include
#CFLAGS += -DYYDEBUG
#CFLAGS += $(shell pkg-config gnutls --cflags) -DSRCDIR=$(PWD)
#LDFLAGS=-lixp -lpthread -rdynamic -ldl
LDFLAGS=-L/sw/lib -lpthread -ldl -ltbb
#LDFLAGS += $(shell pkg-config gnutls --libs)

BIN_OBJ=$(PWD)/bin/tce2.o
LIB=$(PWD)/lib/libtce2.a
LIBS=$(PWD)/lib/parse.a $(PWD)/lib/exec.a $(PWD)/lib/serial.a $(PWD)/lib/lib.a

DIRS = exec parser serial test lib

export CC LD CFLAGS LDFLAGS LIBS LIB

all: bin/tce2

tests: test/test.sh
	sh test/test.sh

distclean: clean
	rm -f bin/tce2

clean:
	rm -f $(LIBS) $(LIB) $(BIN_OBJ)
	rm -f bin/server.o bin/repl.o
	@( for i in $(DIRS); do \
	    $(MAKE) -C $$i clean; \
	    done \
	)

bin/tce2: bin/tce2.o $(LIB)
	$(LD) -o $@ $^ $(LDFLAGS)

test/test.sh:	test $(LIB)
	$(MAKE) -C test test.sh

$(PWD)/lib/exec.a:	exec
	$(MAKE) -C exec ../lib/exec.a

$(PWD)/lib/parse.a:	parser
	$(MAKE) -C parser ../lib/parse.a

$(PWD)/lib/serial.a:	serial
	$(MAKE) -C serial ../lib/serial.a

$(PWD)/lib/lib.a:	lib
	$(MAKE) -C lib lib.a

$(LIB):		$(LIBS)
	rm -f $@
	libtool -static -o $@ $^
	#ar cqT $@ $^

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $^

