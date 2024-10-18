#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <omp.h>

// Defauilt is ~13.6GB when used with int64
#ifndef ARR_LEN
    #define ARR_LEN 1700000000
#endif

enum {
        NS_PER_SECOND = 1000000000,
        ARR_ELEM_MAX = 100
    };

long* create_array(size_t len)
{
    const int prot_flags = PROT_READ|PROT_WRITE;
    const int map_flags = MAP_PRIVATE|MAP_ANON; // MAP_POPULATE
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

int main()
{
    long* array = create_array(ARR_LEN);
    init_array(array, ARR_LEN, 0xA77);

    #pragma omp target map(to: array[:ARR_LEN])
    {
        if (omp_is_initial_device()) {
            printf("Running on host\n");
        } else {
            printf("Running on target\n");
        }
        printf("Available number of threads: %d\n", omp_get_max_threads());

        double start = omp_get_wtime();

        long sum = 0;
        #pragma omp parallel for reduction(+: sum)
        for (size_t i = 0; i < ARR_LEN; ++i) {
            sum += array[i];
        }

        double res = (double)sum/ARR_LEN;

        double end = omp_get_wtime();

        printf("\n");
        printf("Calculation time: %lf\n", end - start);
        printf("Result: avg=%f\n", res);
    }

    return 0;
}
