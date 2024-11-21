#include <stdlib.h>
#include <stdio.h>

/*
    Even though this code targets OpenCL 2.X,
    it can be run on older hardware with the
    help of some runtime version checks
*/

#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define STR_LEN 128

cl_device_id create_device(unsigned int* cl_version)
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

        if (cl_version) {
            unsigned int v1 = 0, v2 = 0, v3 = 0;

            if(sscanf(version, "OpenCL %d.%d.%d", &v1, &v2, &v3) != EOF) {
                *cl_version = v1*100 + v2*10 + v3;
            }
        }

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

cl_command_queue create_queue(cl_context context, cl_device_id device, unsigned int cl_version)
{
    cl_int err = CL_SUCCESS;
    cl_command_queue queue = {0};

    if (cl_version >= 200) {
        const cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    } else {
        printf("OpenCL version is %u, using deprecated clCreateCommandQueue()\n", cl_version);
        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    }
    if(err != CL_SUCCESS) {
        perror("clCreateCommandQueue");
        exit(EXIT_FAILURE);
    };

    return queue;
}
