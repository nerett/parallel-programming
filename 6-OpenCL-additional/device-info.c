#include <stdio.h>

//#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#define STR_LEN 128

int main()
{
    printf("OpenCL info:\n");

    cl_uint num_platforms = 0;
    cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
    if (status != CL_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    printf("Number of OpenCL platforms: %d\n", num_platforms);
    if (num_platforms <= 0) {
        exit(EXIT_FAILURE);
    }

    cl_platform_id* platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (status != CL_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_platforms; ++i) {
        printf("  Platform ID: %d\n", i);

        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

        if (num_devices <= 0) {
            continue;
        }
        printf("  Number of OpenCL devices: %d\n", num_devices);

        cl_device_id* devices = (cl_device_id*)calloc(sizeof(cl_device_id), num_devices);
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        if (status != CL_SUCCESS) {
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < num_devices; ++j) {
            printf("    Device ID: %d\n", i);

            cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

            char name[STR_LEN];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(name), name, NULL);
            printf("    Device name: %s\n", name);

            char vendor[STR_LEN];
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
            printf("    Vendor: %s\n", vendor);

            char version[STR_LEN];
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(version), version, NULL);
            printf("    Device version: %s\n", version);

            char driver_version[STR_LEN];
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(driver_version), driver_version, NULL);
            printf("    Driver version: %s\n", driver_version);

            cl_device_type type;
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            printf("    Device type: %s\n", (type == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU");

            cl_uint num_compute_units;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_compute_units), &num_compute_units, NULL);
            printf("    Compute units: %u\n", num_compute_units);

            cl_uint max_work_item_dim = 0;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dim), &max_work_item_dim, NULL);
            printf("    Work item dimensions: %u\n", max_work_item_dim);

            size_t work_item_sizes[10];
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(work_item_sizes), work_item_sizes, NULL);
            printf("    Work item sizes: ");
            for (int k = 0; k < max_work_item_dim; ++k) {
                printf("%zu ", work_item_sizes[k]);
            }
            printf("\n");

            cl_ulong vram_size;
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vram_size), &vram_size, NULL);
            printf("    VRAM size: %.2f GB\n", (double)vram_size/(1024*1024*1024));

            cl_ulong max_malloc_size;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_malloc_size), &max_malloc_size, NULL);
            printf("    Max memory allocation size: %.2f GB\n", (double)max_malloc_size/(1024*1024*1024));

            cl_ulong cache_size;
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cache_size), &cache_size, NULL);
            printf("    Cache size: %.2f kB\n", (double)cache_size/(1024));

            cl_uint max_core_clock;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_core_clock), &max_core_clock, NULL);
            printf("    Max core clock: %u MHz\n", max_core_clock);
        }

        free(devices);
    }

    free(platforms);

    return 0;
}
