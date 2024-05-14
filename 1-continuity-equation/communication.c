#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "mpi.h"

#define COMM_ITERATIONS 50000
#define BUFF_SIZE 1000
#define TAG 9

int main(int argc, char** argv)
{
    int size = 0, rank = 0;
    MPI_Status status = {};

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int32_t* buf = (int32_t*)malloc(BUFF_SIZE * sizeof(int32_t));
    //int* buf = (int*)malloc(BUFF_SIZE * sizeof(int));
    //int32_t* buf = (int32_t*)calloc(BUFF_SIZE, sizeof(int32_t));
    //printf("")

    if (rank == 0) {
        struct timespec tstart={}, tend={};
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        for (int i = 0; i < COMM_ITERATIONS; i++) {
            MPI_Send(buf, BUFF_SIZE, MPI_INT, 1, TAG, MPI_COMM_WORLD);
            MPI_Recv(buf, BUFF_SIZE, MPI_INT, 1, TAG, MPI_COMM_WORLD, &status);
        }

        clock_gettime(CLOCK_MONOTONIC, &tend);
        double dt = ((tend.tv_sec - tstart.tv_sec)*1e9 + (tend.tv_nsec - tstart.tv_nsec)) / (COMM_ITERATIONS * 2);
        printf("Communication time is %.1lfns\n", dt);
    }
    else if (rank == 1) {
        for (int i = 0; i < COMM_ITERATIONS; i++) {
            MPI_Recv(buf, BUFF_SIZE, MPI_INT, 0, TAG, MPI_COMM_WORLD, &status);
            MPI_Send(buf, BUFF_SIZE, MPI_INT, 0, TAG, MPI_COMM_WORLD);
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}