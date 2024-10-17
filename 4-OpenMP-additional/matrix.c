#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/mman.h>
#include <immintrin.h>
#include <omp.h>

#ifndef MATRIX_DIM
    #define MATRIX_DIM 8192
#endif
#ifndef MATRIX_MUL_BS
    #define MATRIX_MUL_BS 512
#endif

#ifdef PARALLEL
    const int enable_omp_parallel = 1;
#else
    const int enable_omp_parallel = 0;
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

void mul_matrix(long* A, long* B, long* C, size_t dim)
{
    #pragma omp parallel for if (enable_omp_parallel)
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    C[i*dim + j] += A[i*dim + k] * B[k*dim + j];
                }
            }
        }
}

void transposed_mul_matrix(long* A, long* B, long* C, size_t dim)
{
    #pragma omp parallel for if (enable_omp_parallel)
        for (int i = 0; i < dim; ++i) {
            for (int k = 0; k < dim; ++k) {
                for (int j = 0; j < dim; ++j) {
                    C[i*dim + j] += A[i*dim + k] * B[k*dim + j];
                }
            }
        }
}

void block_mul_matrix(long* A, long* B, long* C, size_t dim)
{
    size_t bs = MATRIX_MUL_BS;

    #pragma omp parallel for if (enable_omp_parallel)
        for (size_t i = 0; i < dim; i += bs) {
            for (size_t j = 0; j < dim; j += bs) {
                for (size_t k = 0; k < dim; k += bs) {

                    long* rC = &C[i*dim + j];
                    long* rA = &A[i*dim + k];
                    for (size_t i2 = 0; i2 < bs; ++i2) {

                        long* rB = &B[k*dim + j];
                        for (size_t k2 = 0; k2 < bs; ++k2) {
                            for (size_t j2 = 0; j2 < bs; ++j2) {
                                rC[j2] += rA[k2] * rB[j2];
                            }

                            rB += dim;
                        }

                        rC += dim;
                        rA += dim;
                    }
                }
            }
        }
}

#ifdef SIMD
void simd_mul_matrix(long* A, long* BT, long* C, size_t dim)
{
    #pragma omp parallel for if (enable_omp_parallel)
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {

                __m256i c_vec = _mm256_setzero_si256();
                for (size_t k = 0; k < dim; k += 4) {
                    __m256i a_vec = _mm256_loadu_si256((__m256i*)&A[i*dim + k]);
                    __m256i b_vec = _mm256_loadu_si256((__m256i*)&BT[j*dim + k]);

                    __m256i mul_result = _mm256_mul_epi32(a_vec, b_vec);

                    c_vec = _mm256_add_epi64(c_vec, mul_result);
                }

                long temp[4];
                _mm256_storeu_si256((__m256i*)temp, c_vec);
                C[i*dim + j] += temp[0] + temp[1] + temp[2] + temp[3];
            }
        }
}
#endif

void print_matrix(long* matrix, size_t dim)
{
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            printf("%ld ", matrix[i*dim + j]);
        }
        printf("\n");
    }
}

unsigned int hash_matrix(long* matrix, size_t dim)
{
    unsigned int hash = 0;
    for (size_t i = 0; i < dim*dim; ++i) {
        hash += (i % dim) * (matrix[i] ^ MAGIC_KEY);
    }

    return hash;
}

void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *dt)
{
    dt->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    dt->tv_sec = t2.tv_sec - t1.tv_sec;
    if (dt->tv_sec > 0 && dt->tv_nsec < 0) {
        dt->tv_nsec += NS_PER_SECOND;
        dt->tv_sec--;
    }
    else if (dt->tv_sec < 0 && dt->tv_nsec > 0) {
        dt->tv_nsec -= NS_PER_SECOND;
        dt->tv_sec++;
    }
}

int main()
{
    printf("Matrix size: %d x %d\n", MATRIX_DIM, MATRIX_DIM);
    printf("Maximum element size: %d\n", MATRIX_ELEM_MAX);
    if (enable_omp_parallel) {
        printf("OpenMP parallelization enabled\n");
    }

    long* A = create_matrix(MATRIX_DIM);
    long* B = create_matrix(MATRIX_DIM);
    long* BT = create_matrix(MATRIX_DIM);
    long* C = create_matrix(MATRIX_DIM);
    // mlockall(MCL_CURRENT | MCL_FUTURE);

    init_matrix(A, MATRIX_DIM, 0xA);
    init_matrix(B, MATRIX_DIM, 0xB);
    transpose_matrix(B, BT, MATRIX_DIM);

    struct timespec start, finish, delta;
    clock_gettime(CLOCK_MONOTONIC, &start);

    #ifdef TRANSPOSE
        printf("Using transposed_mul_matrix()\n");
        transposed_mul_matrix(A, B, C, MATRIX_DIM);
    #elif BLOCK
        printf("Using block_mul_matrix()\n");
        block_mul_matrix(A, B, C, MATRIX_DIM);
    #elif SIMD
        printf("Using simd_mul_matrix()\n");
        simd_mul_matrix(A, BT, C, MATRIX_DIM);
    #else
        printf("Using mul_matrix()\n");
        mul_matrix(A, B, C, MATRIX_DIM);
    #endif

    clock_gettime(CLOCK_MONOTONIC, &finish);
    sub_timespec(start, finish, &delta);

    printf("\n");
    printf("Multiplication time: %d.%.9ld\n", (int)delta.tv_sec, delta.tv_nsec);

    printf("hash(A) = %x\n", hash_matrix(A, MATRIX_DIM));
    printf("hash(B) = %x\n", hash_matrix(B, MATRIX_DIM));
    printf("hash(C) = %x\n", hash_matrix(C, MATRIX_DIM));

    return 0;
}
