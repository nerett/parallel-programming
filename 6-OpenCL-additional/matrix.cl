__kernel void mul_matrix(__global long* A, __global long* B, __global long* C, int dim)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    long result = 0;
    for (int k = 0; k < dim; k++) {
        result += A[row*dim + k] * B[k*dim + col];
    }

    C[row*dim + col] = result;
}

__kernel void simd_mul_matrix(__global long* A, __global long* BT, __global long* C, int dim) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    long4 a_vec, b_vec, prod_vec;
    long result = 0;

    for (int k = 0; k < dim; k += 4) {
        a_vec = vload4(0, &A[row * dim + k]);
        b_vec = vload4(0, &BT[col * dim + k]);

        prod_vec = a_vec * b_vec;

        result += prod_vec.s0 + prod_vec.s1 + prod_vec.s2 + prod_vec.s3;
    }

    C[row*dim + col] = result;
}
