# Make your make do the work.
# http://www.gnu.org/software/make/manual/make.html#Advanced
# http://okmij.org/ftp/Computation/Make-functional.txt
#
#   Admittedly, this is sloppy, since it references path names
# inside the make system.  However, it accurately finds
# all the dependencies and recompiles on source changes,
# which was the original idea anyway.

########### configurable options

CC=icc
LD=icc
CFLAGS=-ggdb -fPIC -I$(PWD)/include
#CFLAGS=-ggdb -I/sw/include -I$(PWD)/include

LDFLAGS=-lpthread

#CFLAGS += -DYYDEBUG

# OSX Accelerate
#LDFLAGS += -framework Accelerate
# Generic BLAS
LDFLAGS += -lblas -llapack

# use QUARK
QUARK=$(HOME)/build/quark-0.9.0
CFLAGS += -I$(QUARK) -DQUARK
LDFLAGS += -L$(QUARK) -lquark
export QUARK

# use STARPU
#CFLAGS += $(shell pkg-config --cflags starpu-1.1) -DSTARPU
#LDFLAGS += $(shell pkg-config --libs starpu-1.1)

############ end config options

BIN_OBJ=$(PWD)/bin/slack.o
LIB=$(PWD)/lib/libslack.a
LIBS=$(PWD)/lib/parse.a $(PWD)/lib/exec.a $(PWD)/lib/serial.a $(PWD)/lib/lib.a

DIRS = exec parser serial test lib

export CC LD CFLAGS LDFLAGS LIBS LIB

all: bin/slack

tests: test/test.sh
	sh test/test.sh

distclean: clean
	rm -f bin/slack

clean:
	rm -f $(LIBS) $(LIB) $(BIN_OBJ)
	rm -f bin/server.o bin/repl.o
	@( for i in $(DIRS); do \
	    $(MAKE) -C $$i clean; \
	    done \
	)

bin/slack: bin/slack.o $(LIB)
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
	#libtool -static -o $@ $^
	ar cqT $@ $^

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $^

%.o: %.cu
	nvcc $(CFLAGS) $< -c $@

