CONTAINER = rocm-24.04
ENVC = distrobox enter $(CONTAINER) --
#!TODO add installation guide

CC = /opt/rocm/bin/amdclang
TARGET = amdgcn-amd-amdhsa
ARCH = gfx1101
CFLAGS = -O3 -fopenmp -fopenmp-targets=$(TARGET) -Xopenmp-target=$(TARGET) -march=$(ARCH) $(UFLAGS)
#CFLAGS = -O3 -fopenmp --offload-arch=$(ARCH) $(UFLAGS)

all:
	@echo "Use make <specific target> instead. Check Makefile for more details"

test:
	$(ENVC) $(CC) $(CFLAGS) offload-test.c

avg:
	$(ENVC) $(CC) $(CFLAGS) offload-avg.c

run:
	$(ENVC) ./a.out

clean:
	rm *\.out
	rm *\.o
