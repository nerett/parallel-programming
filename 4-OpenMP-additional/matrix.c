#include "matrix-tools.h"
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
        for (size_t i = 0; i < dim; ++i) {
            for (size_t k = 0; k < dim; ++k) {
                for (size_t j = 0; j < dim; ++j) {
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

    double start = omp_get_wtime();

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

    double end = omp_get_wtime();

    printf("\n");
    printf("Multiplication time: %lf\n", end - start);

    printf("hash(A) = %x\n", hash_matrix(A, MATRIX_DIM));
    printf("hash(B) = %x\n", hash_matrix(B, MATRIX_DIM));
    printf("hash(C) = %x\n", hash_matrix(C, MATRIX_DIM));

    return 0;
}
