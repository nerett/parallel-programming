#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "mpi.h"

#ifndef LIBSODIUM_ENABLED
    #include <time.h>
#endif
#ifdef LIBSODIUM_ENABLED
    #include <sodium.h>
#endif

#define PI 3.1415926535897932384626433832795028841971
#define SHOOT_ITERATIONS 1000000
#define TAG 5

unsigned long long shoot_circle_segment(unsigned long long iterations) //Monte Carlo method
{
    const unsigned long long R = RAND_MAX;
    const unsigned long long R2 = R*R;
    unsigned long long x = 0, y = 0;
    unsigned long long counter = 0;
    
    #ifndef LIBSODIUM_ENABLED
        srand(time(0));
    #endif
    #ifdef LIBSODIUM_ENABLED
        if (sodium_init() < 0) {
            fprintf(stderr, "Unable to init libsodium!\n");
            return 1;
        }
    #endif

    for (unsigned long long  i = 0; i < iterations; i++) { //shoot at the inscribed circle segment (x>0, y>0)
        #ifndef LIBSODIUM_ENABLED
            x = rand();
            y = rand();
        #endif
        #ifdef LIBSODIUM_ENABLED
            x = randombytes_uniform(RAND_MAX);
            y = randombytes_uniform(RAND_MAX);
        #endif

        if (x*x + y*y < R*R)
            counter++;
    }

    return counter;
}

int main(int argc, char** argv)
{
    int size = 0, rank = 0;
    unsigned long long total_hits = 0;
    MPI_Status status = {};

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    unsigned long long total_iterations = size * SHOOT_ITERATIONS;
    unsigned long long n_hits = shoot_circle_segment(SHOOT_ITERATIONS);

    if (rank) {
        MPI_Send(&n_hits, 1, MPI_UNSIGNED_LONG_LONG, 0, TAG, MPI_COMM_WORLD);
    }
    else { //main node with rank = 0
        unsigned long long buff = 0;
        for (int i = 0; i < size - 1; i++) {
            if (MPI_Recv(&buff, 1, MPI_UNSIGNED_LONG_LONG, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
                fprintf(stderr, "Unable to receive MPI message!\n");
                return 1;
            }

            total_hits += buff;
        }

        total_hits += n_hits;
        long double pi = 4 * ((long double)total_hits) / ((long double)(total_iterations));

    #ifdef LIBSODIUM_ENABLED
            printf("Used libsodium to generate random shots\n");
    #endif
        printf("PI = %Lf\n", pi);
        printf("Calculation residual is %Lf\n", pi - PI);
    }

    MPI_Finalize();
    return 0;
}