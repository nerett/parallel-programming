CONTAINER = rocm-24.04
ENVC = distrobox enter $(CONTAINER) --
#!TODO add installation guide

# CC = /opt/rocm/bin/amdclang
CC = gcc
CFLAGS = -O3 $(UFLAGS)
LFLAGS = -lOpenCL
#CFLAGS = -O3 -fopenmp --offload-arch=$(ARCH) $(UFLAGS)

all:
	@echo "Use make <specific target> instead. Check Makefile for more details"

info:
	$(ENVC) $(CC) $(CFLAGS) device-info.c $(LFLAGS)

# matrix:
# 	$(ENVC) $(CC) $(CFLAGS) offload-matrix.c

run:
	$(ENVC) ./a.out

clean:
	rm -f *\.out
	rm -f *\.o
