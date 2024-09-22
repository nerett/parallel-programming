all:
	@echo "Use make <specific target> instead. Check Makefile for more details"

pi:
	mpicc pi.c -lsodium -DLIBSODIUM_ENABLED

communication:
	mpicc communication.c

run:
	mpirun ./a.out

run-100:
	mpirun -np 100 --oversubscribe ./a.out
