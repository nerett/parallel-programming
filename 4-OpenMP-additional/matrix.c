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

inline void _add_matrix(long* A, long* B, long* C, size_t n) {
    #pragma omp parallel for if(enable_omp_parallel)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    }
}

inline void _sub_matrix(long* A, long* B, long* C, size_t n) {
    #pragma omp parallel for if(enable_omp_parallel)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] - B[i * n + j];
        }
    }
}

void _strassen(long* A, long* B, long* C, size_t n) {
    if (n <= 128) {
        transposed_mul_matrix(A, B, C, n);

        return;
    }

    size_t newSize = n / 2;
    long* A11 = (long*)calloc(newSize * newSize, sizeof(long));
    long* A12 = (long*)calloc(newSize * newSize, sizeof(long));
    long* A21 = (long*)calloc(newSize * newSize, sizeof(long));
    long* A22 = (long*)calloc(newSize * newSize, sizeof(long));

    long* B11 = (long*)calloc(newSize * newSize, sizeof(long));
    long* B12 = (long*)calloc(newSize * newSize, sizeof(long));
    long* B21 = (long*)calloc(newSize * newSize, sizeof(long));
    long* B22 = (long*)calloc(newSize * newSize, sizeof(long));

    long* M1 = (long*)calloc(newSize * newSize, sizeof(long));
    long* M2 = (long*)calloc(newSize * newSize, sizeof(long));
    long* M3 = (long*)calloc(newSize * newSize, sizeof(long));
    long* M4 = (long*)calloc(newSize * newSize, sizeof(long));
    long* M5 = (long*)calloc(newSize * newSize, sizeof(long));
    long* M6 = (long*)calloc(newSize * newSize, sizeof(long));
    long* M7 = (long*)calloc(newSize * newSize, sizeof(long));

    // Divide matrices into quadrants
    #pragma omp parallel for collapse(2) if(enable_omp_parallel)
    for (size_t i = 0; i < newSize; i++) {
        for (size_t j = 0; j < newSize; j++) {
            A11[i * newSize + j] = A[i * n + j];
            A12[i * newSize + j] = A[i * n + j + newSize];
            A21[i * newSize + j] = A[(i + newSize) * n + j];
            A22[i * newSize + j] = A[(i + newSize) * n + j + newSize];

            B11[i * newSize + j] = B[i * n + j];
            B12[i * newSize + j] = B[i * n + j + newSize];
            B21[i * newSize + j] = B[(i + newSize) * n + j];
            B22[i * newSize + j] = B[(i + newSize) * n + j + newSize];
        }
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task shared(M1) if(enable_omp_parallel)
            {
                long* temp1 = (long*)calloc(newSize * newSize, sizeof(long));
                long* temp2 = (long*)calloc(newSize * newSize, sizeof(long));

                _add_matrix(A11, A22, temp1, newSize);
                _add_matrix(B11, B22, temp2, newSize);
                _strassen(temp1, temp2, M1, newSize);

                free(temp1);
                free(temp2);
            }

            #pragma omp task shared(M2) if(enable_omp_parallel)
            {
                long* temp1 = (long*)calloc(newSize * newSize, sizeof(long));

                _add_matrix(A21, A22, temp1, newSize);
                _strassen(temp1, B11, M2, newSize);

                free(temp1);
            }

            #pragma omp task shared(M3) if(enable_omp_parallel)
            {
                long* temp2 = (long*)calloc(newSize * newSize, sizeof(long));

                _sub_matrix(B12, B22, temp2, newSize);
                _strassen(A11, temp2, M3, newSize);

                free(temp2);
            }

            #pragma omp task shared(M4) if(enable_omp_parallel)
            {
                long* temp2 = (long*)calloc(newSize * newSize, sizeof(long));

                _sub_matrix(B21, B11, temp2, newSize);
                _strassen(A22, temp2, M4, newSize);

                free(temp2);
            }

            #pragma omp task shared(M5) if(enable_omp_parallel)
            {
                long* temp1 = (long*)calloc(newSize * newSize, sizeof(long));

                _add_matrix(A11, A12, temp1, newSize);
                _strassen(temp1, B22, M5, newSize);

                free(temp1);
            }

            #pragma omp task shared(M6) if(enable_omp_parallel)
            {
                long* temp1 = (long*)calloc(newSize * newSize, sizeof(long));
                long* temp2 = (long*)calloc(newSize * newSize, sizeof(long));

                _sub_matrix(A21, A11, temp1, newSize);
                _add_matrix(B11, B12, temp2, newSize);
                _strassen(temp1, temp2, M6, newSize);

                free(temp1);
                free(temp2);
            }

            #pragma omp task shared(M7) if(enable_omp_parallel)
            {
                long* temp1 = (long*)calloc(newSize * newSize, sizeof(long));
                long* temp2 = (long*)calloc(newSize * newSize, sizeof(long));

                _sub_matrix(A12, A22, temp1, newSize);
                _add_matrix(B21, B22, temp2, newSize);
                _strassen(temp1, temp2, M7, newSize);

                free(temp1);
                free(temp2);
            }

            #pragma omp taskwait
        }
    }

    // Combine results into C
    #pragma omp parallel for if(enable_omp_parallel)
    for (size_t i = 0; i < newSize; i++) {
        for (size_t j = 0; j < newSize; j++) {
            C[i * n + j] += M1[i * newSize + j] + M4[i * newSize + j] - M5[i * newSize + j] + M7[i * newSize + j];
            C[i * n + j + newSize] += M3[i * newSize + j] + M5[i * newSize + j];
            C[(i + newSize) * n + j] += M2[i * newSize + j] + M4[i * newSize + j];
            C[(i + newSize) * n + j + newSize] += M1[i * newSize + j] - M2[i * newSize + j] + M3[i * newSize + j] + M6[i * newSize + j];
        }
    }

    free(A11);
    free(A12);
    free(A21);
    free(A22);

    free(B11);
    free(B12);
    free(B21);
    free(B22);

    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);
    free(M7);
}


void fast_mul_matrix(long* A, long* B, long* C, size_t dim) {
    _strassen(A, B, C, dim);
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
    #elif FAST
        printf("Using fast_mul_matrix()\n");
        fast_mul_matrix(A, B, C, MATRIX_DIM);
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
