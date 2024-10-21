#include "../matrix-tools.h"
#include <omp.h>

#ifndef MATRIX_DIM
    #define MATRIX_DIM 8192
#endif
#ifndef MATRIX_MUL_BS
    #define MATRIX_MUL_BS 2
#endif

void target_mul_matrix(long* A, long* B, long* C, size_t dim)
{
    #pragma omp target teams distribute parallel for collapse(2)
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

void target_block_mul_matrix(long* A, long* B, long* C, size_t dim)
{
    size_t bs = MATRIX_MUL_BS;

    #pragma omp target teams distribute parallel for collapse(2)
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

int main()
{
    printf("Matrix size: %d x %d\n", MATRIX_DIM, MATRIX_DIM);
    printf("Maximum element size: %d\n", MATRIX_ELEM_MAX);

    long* A = create_matrix(MATRIX_DIM);
    long* B = create_matrix(MATRIX_DIM);
    long* C = create_matrix(MATRIX_DIM);

    init_matrix(A, MATRIX_DIM, 0xA);
    init_matrix(B, MATRIX_DIM, 0xB);

    const size_t msize = MATRIX_DIM*MATRIX_DIM;
    double start = 0, end = 0;

    #pragma omp target data map(to: A[:msize], B[:msize])
    #pragma omp target data map(tofrom: C[:msize])
    {
        start = omp_get_wtime();

        #ifdef BLOCK
            printf("Using target_block_mul_matrix()\n");
            target_block_mul_matrix(A, B, C, MATRIX_DIM);
        #else
            printf("Using target_mul_matrix()\n");
            target_mul_matrix(A, B, C, MATRIX_DIM);
        #endif

        end = omp_get_wtime();
    }

    printf("\n");
    printf("Calculation time: %lf\n", end - start);

    printf("hash(A) = %x\n", hash_matrix(A, MATRIX_DIM));
    printf("hash(B) = %x\n", hash_matrix(B, MATRIX_DIM));
    printf("hash(C) = %x\n", hash_matrix(C, MATRIX_DIM));
}
