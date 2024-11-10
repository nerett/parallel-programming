#include <stdio.h>
#include <stdlib.h>

#include "../4-OpenMP-additional/matrix-tools.h"

#define CL_TARGET_OPENCL_VERSION 200
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define STR_LEN 128

#define PROGRAM_FILE "matrix.cl"
#define KERNEL_FUNC "mul_matrix"

#ifndef MATRIX_DIM
    #define MATRIX_DIM 8192
#endif
#ifndef DEVICE_LOCAL_SIZE
    #define DEVICE_LOCAL_SIZE 16
#endif


cl_device_id create_device()
{
    cl_platform_id platform = {0};
    cl_device_id device = {0};
    cl_int err = 0;

    cl_uint num_platforms = 0;
    cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
    if (status != CL_SUCCESS || num_platforms <= 0) {
        perror("Couldn't find any platforms");
        exit(EXIT_FAILURE);
    }

    cl_platform_id* platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (status != CL_SUCCESS) {
        perror("Couldn't get platforms");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_platforms; ++i) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if(err == CL_SUCCESS) {
            break;
        }
    }

    free(platforms);

    cl_bool device_ok = CL_FALSE;
    clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(device_ok), &device_ok, NULL);
    if (device_ok == CL_TRUE) {

        char name[STR_LEN] = "";
        char vendor[STR_LEN] = "";
        char version[STR_LEN] = "";
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
        clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);

        printf("Found GPU device: %s; %s; %s\n", name, vendor, version);

    } else {
        fprintf(stderr, "No available GPU devices found\n");
        exit(EXIT_FAILURE);
    }

    return device;
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename)
{
    cl_program program = {0};
    FILE *program_handle = NULL;
    char *program_buffer = NULL, *program_log = NULL;
    size_t program_size = 0, log_size = 0;
    cl_int err = 0;

    program_handle = fopen(filename, "r");
    if(program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(EXIT_FAILURE);
    }

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);

    program_buffer = (char*)calloc(program_size + 1, sizeof(char));
    size_t num_read = fread(program_buffer, sizeof(char), program_size, program_handle);
    if(num_read != program_size) {
        fclose(program_handle);

        perror("Couldn't read the program file");
        exit(EXIT_FAILURE);
    }

    fclose(program_handle);

    program = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    if(err != CL_SUCCESS) {
        perror("Couldn't create the program");
        exit(EXIT_FAILURE);
    }
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS) {
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*)calloc(log_size + 1, sizeof(char));

        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(EXIT_FAILURE);
    }

    return program;
}

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

    cl_device_id device = create_device();
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err != CL_SUCCESS) {
        perror("Couldn't create a context");
        exit(EXIT_FAILURE);
    }

    cl_program program = build_program(context, device, PROGRAM_FILE);

    cl_mem device_A = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, MATRIX_DIM * MATRIX_DIM * sizeof(long), A, &err);
    cl_mem device_B = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, MATRIX_DIM * MATRIX_DIM * sizeof(long), B, &err);
    cl_mem device_C = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, MATRIX_DIM * MATRIX_DIM * sizeof(long), C, &err);
    if(err != CL_SUCCESS) {
        perror("Couldn't create a buffer");
        exit(EXIT_FAILURE);
    };

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err); //!TODO check platform version
    if(err != CL_SUCCESS) {
        perror("Couldn't create a command queue");
        exit(EXIT_FAILURE);
    };

    cl_kernel kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if(err != CL_SUCCESS) {
        perror("Couldn't create a kernel");
        exit(EXIT_FAILURE);
    };

    int dim = MATRIX_DIM;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_C);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &dim);
    if(err != CL_SUCCESS) {
        perror("Couldn't create a kernel argument");
        exit(EXIT_FAILURE);
    }

    size_t global_size[2] = {MATRIX_DIM, MATRIX_DIM};
    size_t local_size[2] = {DEVICE_LOCAL_SIZE, DEVICE_LOCAL_SIZE};

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        perror("Couldn't enqueue the kernel");
        exit(EXIT_FAILURE);
    }

    err = clEnqueueReadBuffer(queue, device_C, CL_TRUE, 0, MATRIX_DIM * MATRIX_DIM * sizeof(long), C, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        perror("Couldn't read the buffer");
        exit(EXIT_FAILURE);
    }

    printf("hash(A) = %x\n", hash_matrix(A, MATRIX_DIM));
    printf("hash(B) = %x\n", hash_matrix(B, MATRIX_DIM));
    printf("hash(C) = %x\n", hash_matrix(C, MATRIX_DIM));

    clFinish(queue);

    clReleaseKernel(kernel);
    clReleaseMemObject(device_A);
    clReleaseMemObject(device_B);
    clReleaseMemObject(device_C);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}
