CC = gcc
CFLAGS = -O3 -fopenmp $(UFLAGS)

ifneq (, $(findstring -DSIMD, $(UFLAGS)))
	SIMDFLAGS = -mavx2
endif


all:
	@echo "Use make <specific target> instead. Check Makefile for more details"

balancing:
	$(CC) $(CFLAGS) balancing.c

numprocs:
	$(CC) $(CFLAGS) numprocs.c

matrix:
	$(CC) $(CFLAGS) $(SIMDFLAGS) matrix.c

run:
	./a.out

clean:
	rm -f *\.out
	rm -f *\.o
