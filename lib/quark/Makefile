QUARK_DIR=$(HOME)/build/quark-0.9.0
CFLAGS=-I$(QUARK_DIR) -I../../include
LDFLAGS=-L$(QUARK_DIR) -lquark
CC=cc

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $^
