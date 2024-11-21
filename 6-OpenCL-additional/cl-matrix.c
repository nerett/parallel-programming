#include <stdio.h>
#include <stdlib.h>

#include "../4-OpenMP-additional/matrix-tools.h"
#include "cl-tools.h"

#define PROGRAM_FILE "matrix.cl"
#define KERNEL_FUNC "simd_mul_matrix"

#ifndef MATRIX_DIM
    #define MATRIX_DIM 16384
#endif
#ifndef DEVICE_LOCAL_SIZE
    #define DEVICE_LOCAL_SIZE 16
#endif


int main()
{
    printf("Matrix size: %d x %d\n", MATRIX_DIM, MATRIX_DIM);
    printf("Maximum element size: %d\n", MATRIX_ELEM_MAX);

    long* A = create_matrix(MATRIX_DIM);
    long* B = create_matrix(MATRIX_DIM);
    long* BT = create_matrix(MATRIX_DIM);
    long* C = create_matrix(MATRIX_DIM);

    init_matrix(A, MATRIX_DIM, 0xA);
    init_matrix(B, MATRIX_DIM, 0xB);
    transpose_matrix(B, BT, MATRIX_DIM);

    cl_int err = CL_SUCCESS;
    unsigned int cl_version = 0;

    cl_device_id device = create_device(&cl_version);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err != CL_SUCCESS) {
        perror("clCreateContext");
        exit(EXIT_FAILURE);
    }

    cl_program program = build_program(context, device, PROGRAM_FILE);

    cl_mem device_A = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, MATRIX_DIM * MATRIX_DIM * sizeof(long), A, &err);
    cl_mem device_B = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, MATRIX_DIM * MATRIX_DIM * sizeof(long), BT, &err);
    cl_mem device_C = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, MATRIX_DIM * MATRIX_DIM * sizeof(long), C, &err);
    if(err != CL_SUCCESS) {
        perror("clCreateBuffer");
        exit(EXIT_FAILURE);
    };

    cl_command_queue queue = create_queue(context, device, cl_version);

    cl_kernel kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if(err != CL_SUCCESS) {
        perror("clCreateKernel");
        exit(EXIT_FAILURE);
    };

    int dim = MATRIX_DIM;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_C);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &dim);
    if(err != CL_SUCCESS) {
        perror("clSetKernelArg");
        exit(EXIT_FAILURE);
    }

    printf("Running %s() on device\n", KERNEL_FUNC);

    size_t global_size[2] = {MATRIX_DIM, MATRIX_DIM};
    size_t local_size[2] = {DEVICE_LOCAL_SIZE, DEVICE_LOCAL_SIZE}; //!TODO CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
    cl_event event = {0};

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &event);
    if(err != CL_SUCCESS) {
        perror("clEnqueueNDRangeKernel");
        exit(EXIT_FAILURE);
    }

    err = clEnqueueReadBuffer(queue, device_C, CL_TRUE, 0, MATRIX_DIM * MATRIX_DIM * sizeof(long), C, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        perror("clEnqueueReadBuffer");
        exit(EXIT_FAILURE);
    }

    clFinish(queue);

    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

    printf("\n");
    printf("Multiplication time: %lf\n", ((double)end - (double)start)/1e9);

    printf("hash(A) = %x\n", hash_matrix(A, MATRIX_DIM));
    printf("hash(B) = %x\n", hash_matrix(B, MATRIX_DIM));
    printf("hash(C) = %x\n", hash_matrix(C, MATRIX_DIM));

    clReleaseKernel(kernel);
    clReleaseMemObject(device_A);
    clReleaseMemObject(device_B);
    clReleaseMemObject(device_C);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}
