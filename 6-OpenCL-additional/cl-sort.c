#include <sys/mman.h>
#include "cl-tools.h"
#include <time.h>

#define PROGRAM_FILE "sort.cl"
#define KERNEL_FUNC "bitonic_sort"

#ifndef DEVICE_LOCAL_SIZE
    #define DEVICE_LOCAL_SIZE 128
#endif

#ifndef ARRAY_LENGTH
    #define ARRAY_LENGTH 1 << 30
#endif

const size_t ARR_LEN = ARRAY_LENGTH;

enum {
        ARR_ELEM_MAX = 100
    };

long* create_array(size_t len)
{
    const int prot_flags = PROT_READ|PROT_WRITE;
    const int map_flags = MAP_PRIVATE|MAP_ANON;
    void* ptr = mmap(NULL, sizeof(long)*len, prot_flags, map_flags, -1, 0);
    if(ptr == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }

    return (long*)ptr;
}

void delete_array(long* arr, size_t len)
{
    munmap(arr, sizeof(long)*len);
}

void init_array(long* arr, size_t len, unsigned int seed)
{
    srand(seed);

    for (size_t i = 0; i < len; ++i) {
        arr[i] = rand() % ARR_ELEM_MAX;
    }
}

int is_sorted(long *array, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (array[i - 1] > array[i]) return 0;
    }
    return 1;
}

double get_time() {
    struct timespec ts = {0};
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ts.tv_sec + ts.tv_nsec/1e9;
}

int main()
{
    printf("Array length: %lu\n", ARR_LEN);

    long* array = create_array(ARR_LEN);
    init_array(array, ARR_LEN, 0xA77);

    cl_int err = CL_SUCCESS;
    unsigned int cl_version = 0;

    cl_device_id device = create_device(&cl_version);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err != CL_SUCCESS) {
        perror("clCreateContext");
        exit(EXIT_FAILURE);
    }

    cl_program program = build_program(context, device, PROGRAM_FILE);

    cl_mem device_array = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, ARR_LEN * sizeof(long), array, &err);
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

    printf("Running %s() on device\n", KERNEL_FUNC);

    size_t global_size = ARR_LEN;
    size_t local_size = DEVICE_LOCAL_SIZE;

    double start = get_time();

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_array);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &ARR_LEN);
    if(err != CL_SUCCESS) {
        perror("clSetKernelArg");
        exit(EXIT_FAILURE);
    };

    for (int stage = 2; stage <= ARR_LEN; stage <<= 1) {
        for (int step = stage >> 1; step > 0; step >>= 1) {
            err = clSetKernelArg(kernel, 2, sizeof(int), &stage);
            err |= clSetKernelArg(kernel, 3, sizeof(int), &step);
            if(err != CL_SUCCESS) {
                perror("clSetKernelArg");
                exit(EXIT_FAILURE);
            };

            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
            if(err != CL_SUCCESS) {
                perror("clEnqueueNDRangeKernel");
                exit(EXIT_FAILURE);
            };

            clFinish(queue);
        }
    }

    clFinish(queue);
    double end = get_time();

    err = clEnqueueReadBuffer(queue, device_array, CL_TRUE, 0, ARR_LEN * sizeof(long), array, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        perror("clEnqueueReadBuffer");
        exit(EXIT_FAILURE);
    }

    clFinish(queue);

    printf("\n");
    printf("Calculation time: %lf\n", end - start);

    if (is_sorted(array, ARR_LEN)) {
        printf("Array is sorted.\n");
    } else {
        printf("Array is NOT sorted!\n");
    }

    clReleaseKernel(kernel);
    clReleaseMemObject(device_array);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}
