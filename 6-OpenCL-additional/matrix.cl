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
