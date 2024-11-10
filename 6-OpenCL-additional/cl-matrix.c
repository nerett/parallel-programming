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
#define KERNEL_FUNC "simd_mul_matrix"

#ifndef MATRIX_DIM
    #define MATRIX_DIM 16384
#endif
#ifndef DEVICE_LOCAL_SIZE
    #define DEVICE_LOCAL_SIZE 16
#endif


cl_device_id create_device()
{
    cl_device_id device = {0};
    cl_int err = 0;

    cl_uint num_platforms = 0;
    cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
    if (status != CL_SUCCESS || num_platforms <= 0) {
        perror("clGetPlatformIDs");
        exit(EXIT_FAILURE);
    }

    cl_platform_id* platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (status != CL_SUCCESS) {
        perror("clGetPlatformIDs");
        exit(EXIT_FAILURE);
    }

    for (cl_uint i = 0; i < num_platforms; ++i) {
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

cl_program build_program(cl_context context, cl_device_id device, const char* filename)
{
    cl_int err = CL_SUCCESS;

    FILE* program_handle = fopen(filename, "r");
    if(!program_handle) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    fseek(program_handle, 0, SEEK_END);
    size_t program_size = ftell(program_handle);
    rewind(program_handle);

    char* program_buffer = (char*)calloc(program_size + 1, sizeof(char));
    size_t num_read = fread(program_buffer, sizeof(char), program_size, program_handle);
    if(num_read != program_size) {
        perror("fread");

        fclose(program_handle);
        exit(EXIT_FAILURE);
    }

    fclose(program_handle);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &program_size, &err);
    if(err != CL_SUCCESS) {
        perror("clCreateProgramWithSource");
        exit(EXIT_FAILURE);
    }
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* program_log = (char*)calloc(log_size + 1, sizeof(char));

        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
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

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); //!TODO check platform version
    if(err != CL_SUCCESS) {
        perror("clCreateCommandQueue");
        exit(EXIT_FAILURE);
    };

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
    size_t local_size[2] = {DEVICE_LOCAL_SIZE, DEVICE_LOCAL_SIZE};
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
