#include "matrix-tools.h"
#include <immintrin.h>
#include <omp.h>

#ifndef MATRIX_DIM
    #define MATRIX_DIM 8192
#endif
#ifndef MATRIX_MUL_BS
    #define MATRIX_MUL_BS 512
#endif
#ifndef MATRIX_FASTMUL_THRESHHOLD
    #define MATRIX_FASTMUL_THRESHHOLD 128
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

#ifdef AVX
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

#ifdef AVX
inline void __avx_add_matrix(long* A, long* B, long* C, size_t dim)
{
    size_t size = dim * dim;
    size_t i = 0;

    for (; i + 4 <= size; i += 4) {
        __m256i a_vec = _mm256_loadu_si256((__m256i*)&A[i]);
        __m256i b_vec = _mm256_loadu_si256((__m256i*)&B[i]);

        __m256i c_vec = _mm256_add_epi64(a_vec, b_vec);

        _mm256_storeu_si256((__m256i*)&C[i], c_vec);
    }

    // Process remaining elements that do not fit in a 256-bit register
    for (; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

inline void __avx_sub_matrix(long* A, long* B, long* C, size_t dim)
{
    size_t size = dim * dim;
    size_t i = 0;

    for (; i + 4 <= size; i += 4) {
        __m256i a_vec = _mm256_loadu_si256((__m256i*)&A[i]);
        __m256i b_vec = _mm256_loadu_si256((__m256i*)&B[i]);

        __m256i c_vec = _mm256_sub_epi64(a_vec, b_vec);

        _mm256_storeu_si256((__m256i*)&C[i], c_vec);
    }

    // Process remaining elements that do not fit in a 256-bit register
    for (; i < size; i++) {
        C[i] = A[i] - B[i];
    }
}
#endif

void _add_matrix(long* A, long* B, long* C, size_t dim)
{
    #ifdef AVX
        __avx_add_matrix(A, B, C, dim);
    #else
        #pragma omp parallel for if(enable_omp_parallel)
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                C[i*dim + j] = A[i*dim + j] + B[i*dim + j];
            }
        }
    #endif
}

void _sub_matrix(long* A, long* B, long* C, size_t dim)
{
    #ifdef AVX
        __avx_sub_matrix(A, B, C, dim);
    #else
        #pragma omp parallel for if(enable_omp_parallel)
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                C[i*dim + j] = A[i*dim + j] - B[i*dim + j];
            }
        }
    #endif
}

void _strassen(long* A, long* B, long* C, size_t dim)
{
    if (dim <= MATRIX_FASTMUL_THRESHHOLD) {
        transposed_mul_matrix(A, B, C, dim);

        return;
    }

    size_t new_dim = dim / 2;
    long* A11 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* A12 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* A21 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* A22 = (long*)calloc(new_dim * new_dim, sizeof(long));

    long* B11 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* B12 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* B21 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* B22 = (long*)calloc(new_dim * new_dim, sizeof(long));

    long* M1 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* M2 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* M3 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* M4 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* M5 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* M6 = (long*)calloc(new_dim * new_dim, sizeof(long));
    long* M7 = (long*)calloc(new_dim * new_dim, sizeof(long));

    // Divide matrices into quadrants
    #pragma omp parallel for collapse(2) if(enable_omp_parallel)
    for (size_t i = 0; i < new_dim; i++) {
        for (size_t j = 0; j < new_dim; j++) {
            A11[i*new_dim + j] = A[i*dim + j];
            A12[i*new_dim + j] = A[i*dim + j + new_dim];
            A21[i*new_dim + j] = A[(i + new_dim)*dim + j];
            A22[i*new_dim + j] = A[(i + new_dim)*dim + j + new_dim];

            B11[i*new_dim + j] = B[i*dim + j];
            B12[i*new_dim + j] = B[i*dim + j + new_dim];
            B21[i*new_dim + j] = B[(i + new_dim)*dim + j];
            B22[i*new_dim + j] = B[(i + new_dim)*dim + j + new_dim];
        }
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task shared(M1) if(enable_omp_parallel)
            {
                long* temp1 = (long*)calloc(new_dim * new_dim, sizeof(long));
                long* temp2 = (long*)calloc(new_dim * new_dim, sizeof(long));

                _add_matrix(A11, A22, temp1, new_dim);
                _add_matrix(B11, B22, temp2, new_dim);
                _strassen(temp1, temp2, M1, new_dim);

                free(temp1);
                free(temp2);
            }

            #pragma omp task shared(M2) if(enable_omp_parallel)
            {
                long* temp1 = (long*)calloc(new_dim * new_dim, sizeof(long));

                _add_matrix(A21, A22, temp1, new_dim);
                _strassen(temp1, B11, M2, new_dim);

                free(temp1);
            }

            #pragma omp task shared(M3) if(enable_omp_parallel)
            {
                long* temp2 = (long*)calloc(new_dim * new_dim, sizeof(long));

                _sub_matrix(B12, B22, temp2, new_dim);
                _strassen(A11, temp2, M3, new_dim);

                free(temp2);
            }

            #pragma omp task shared(M4) if(enable_omp_parallel)
            {
                long* temp2 = (long*)calloc(new_dim * new_dim, sizeof(long));

                _sub_matrix(B21, B11, temp2, new_dim);
                _strassen(A22, temp2, M4, new_dim);

                free(temp2);
            }

            #pragma omp task shared(M5) if(enable_omp_parallel)
            {
                long* temp1 = (long*)calloc(new_dim * new_dim, sizeof(long));

                _add_matrix(A11, A12, temp1, new_dim);
                _strassen(temp1, B22, M5, new_dim);

                free(temp1);
            }

            #pragma omp task shared(M6) if(enable_omp_parallel)
            {
                long* temp1 = (long*)calloc(new_dim * new_dim, sizeof(long));
                long* temp2 = (long*)calloc(new_dim * new_dim, sizeof(long));

                _sub_matrix(A21, A11, temp1, new_dim);
                _add_matrix(B11, B12, temp2, new_dim);
                _strassen(temp1, temp2, M6, new_dim);

                free(temp1);
                free(temp2);
            }

            #pragma omp task shared(M7) if(enable_omp_parallel)
            {
                long* temp1 = (long*)calloc(new_dim * new_dim, sizeof(long));
                long* temp2 = (long*)calloc(new_dim * new_dim, sizeof(long));

                _sub_matrix(A12, A22, temp1, new_dim);
                _add_matrix(B21, B22, temp2, new_dim);
                _strassen(temp1, temp2, M7, new_dim);

                free(temp1);
                free(temp2);
            }

            #pragma omp taskwait
        }
    }

    // Combine results into C
    #pragma omp parallel for if(enable_omp_parallel)
    for (size_t i = 0; i < new_dim; i++) {
        for (size_t j = 0; j < new_dim; j++) {
            C[i*dim + j] += M1[i*new_dim + j] + M4[i*new_dim + j] - M5[i*new_dim + j] + M7[i*new_dim + j];
            C[i*dim + j + new_dim] += M3[i*new_dim + j] + M5[i*new_dim + j];
            C[(i + new_dim)*dim + j] += M2[i*new_dim + j] + M4[i*new_dim + j];
            C[(i + new_dim)*dim + j + new_dim] += M1[i*new_dim + j] - M2[i*new_dim + j] + M3[i*new_dim + j] + M6[i*new_dim + j];
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

void fast_mul_matrix(long* A, long* B, long* C, size_t dim)
{
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
