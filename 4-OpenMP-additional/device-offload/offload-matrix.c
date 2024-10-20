#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/mman.h>
#include <omp.h>

#ifndef MATRIX_DIM
    #define MATRIX_DIM 8192
#endif
#ifndef MATRIX_MUL_BS
    #define MATRIX_MUL_BS 2
#endif

enum {
        NS_PER_SECOND = 1000000000,
        MAGIC_KEY = 0xDEAD10CC,
        MATRIX_ELEM_MAX = 100
    };

long* create_matrix(size_t dim)
{
    const int prot_flags = PROT_READ|PROT_WRITE;
    const int map_flags = MAP_PRIVATE|MAP_ANON; // MAP_POPULATE
    void* ptr = mmap(NULL, sizeof(long)*dim*dim, prot_flags, map_flags, -1, 0);
    if(ptr == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }

    return (long*)ptr;
}

void delete_matrix(long* matrix, size_t dim)
{
    munmap(matrix, sizeof(long)*dim*dim);
}

void init_matrix(long* matrix, size_t dim, unsigned int seed)
{
    srand(seed);

    for (size_t i = 0; i < dim*dim; ++i) {
        matrix[i] = rand() % MATRIX_ELEM_MAX;
    }
}

void transpose_matrix(long* A, long* T, size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            T[j*dim + i] = A[i*dim + j];
        }
    }
}

unsigned int hash_matrix(long* matrix, size_t dim)
{
    // unsigned int hash = 0;
    // for (size_t i = 0; i < dim*dim; ++i) {
    //     hash += (i % dim) * (matrix[i] ^ MAGIC_KEY);
    // }

    unsigned int hash = 0;
    for (size_t i = 0; i < dim*dim; ++i) {
        hash += (i % dim) * ((long)matrix[i] ^ MAGIC_KEY);
    }

    return hash;
}

void print_matrix(long* matrix, size_t dim)
{
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            printf("%ld ", matrix[i*dim + j]);
        }
        printf("\n");
    }
}

void target_mul_matrix(long* A, long* B, long* C, size_t dim)
{
    size_t msize = dim * dim;
    #pragma omp target teams distribute parallel for collapse(2) map(tofrom: dim, A[:msize], B[:msize], C[:msize])
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            long sum = 0;
            for (size_t k = 0; k < dim; ++k) {
                sum += A[i*dim + k] * B[k*dim + j];
            }
            C[i*dim + j] = sum;
        }
    }
}

#pragma omp declare target
    void target_mul_matrix(long* A, long* B, long* C, size_t dim);
#pragma omp end declare target

int main()
{
    long* A = create_matrix(MATRIX_DIM);
    long* B = create_matrix(MATRIX_DIM);
    long* BT = create_matrix(MATRIX_DIM);
    long* C = create_matrix(MATRIX_DIM);

    init_matrix(A, MATRIX_DIM, 0xA);
    init_matrix(B, MATRIX_DIM, 0xB);
    transpose_matrix(B, BT, MATRIX_DIM);

    size_t dim = MATRIX_DIM;
    const size_t msize = dim*dim;

    double start = omp_get_wtime();

    target_mul_matrix(A, B, C, MATRIX_DIM);

    // #pragma omp target teams distribute parallel for collapse(2) map(tofrom: dim, A[:msize], B[:msize], C[:msize])
    // for (size_t i = 0; i < dim; ++i) {
    //     for (size_t j = 0; j < dim; ++j) {
    //         long sum = 0;
    //         for (size_t k = 0; k < dim; ++k) {
    //             sum += A[i*dim + k] * B[k*dim + j];
    //         }
    //         C[i*dim + j] = sum;
    //     }
    // }

    // for (size_t i = 0; i < dim; i += MATRIX_MUL_BS) {
    //     for (size_t j = 0; j < dim; j += MATRIX_MUL_BS) {
    //         for (size_t k = 0; k < dim; k += MATRIX_MUL_BS) {

    //             for (size_t i2 = 0; i2 < MATRIX_MUL_BS; ++i2) {

    //                 long* rA = &A[i*dim + k];
    //                 long* rB = &B[k*dim + j];
    //                 long* rC = &C[i*dim + j];
    //                 for (size_t k2 = 0; k2 < MATRIX_MUL_BS; ++k2) {
    //                     for (size_t j2 = 0; j2 < MATRIX_MUL_BS; ++j2) {
    //                         rC[j2] += rA[k2] * rB[j2];
    //                     }

    //                     rB += dim;
    //                 }

    //                 rC += dim;
    //                 rA += dim;
    //             }
    //         }
    //     }
    // }

    // for (size_t i = 0; i < dim; i += MATRIX_MUL_BS) {
    //     for (size_t j = 0; j < dim; j += MATRIX_MUL_BS) {
    //         for (size_t k = 0; k < dim; k += MATRIX_MUL_BS) {
    //             C[i*dim + j] += A[i*dim + k] * B[k*dim + j];
    //         }
    //     }
    // }

    double end = omp_get_wtime();

    printf("\n");
    printf("Calculation time: %lf\n", end - start);


    printf("hash(A) = %x\n", hash_matrix(A, MATRIX_DIM));
    printf("hash(B) = %x\n", hash_matrix(B, MATRIX_DIM));
    printf("hash(C) = %x\n", hash_matrix(C, MATRIX_DIM));

    // printf("A:\n");
    // print_matrix(A, MATRIX_DIM);
    // printf("B:\n");
    // print_matrix(B, MATRIX_DIM);
    // printf("C:\n");
    // print_matrix(C, MATRIX_DIM);
}
