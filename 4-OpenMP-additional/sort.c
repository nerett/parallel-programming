#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <omp.h>

#ifndef ARR_LEN
    #define ARR_LEN 1 << 28
#endif

enum {
        ARR_ELEM_MAX = 100,
        MERGE_SORT_THRESHHOLD = 64
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

void _insertion_sort(long *array, size_t n) {
    for (size_t i = 1; i < n; i++) {

        long key = array[i];
        size_t j = i;
        while (j > 0 && array[j - 1] > key) {
            array[j] = array[j - 1];
            j--;
        }

        array[j] = key;
    }
}

void _merge(long *array, long *temp, int left, int mid, int right) {
    int i = left, j = mid, k = left;

    while (i < mid && j < right) {
        if (array[i] <= array[j]) temp[k++] = array[i++];
        else temp[k++] = array[j++];
    }

    while (i < mid) temp[k++] = array[i++];
    while (j < right) temp[k++] = array[j++];

    for (i = left; i < right; i++) array[i] = temp[i];
}

void _parallel_merge_sort(long *array, long *temp, int left, int right, int threshold) {
    if (right - left <= threshold) {
        _insertion_sort(array + left, right - left);
    } else {
        int mid = left + (right - left)/2;

        #pragma omp task shared(array, temp) if(right - left > threshold)
            _parallel_merge_sort(array, temp, left, mid, threshold);
        #pragma omp task shared(array, temp) if(right - left > threshold)
            _parallel_merge_sort(array, temp, mid, right, threshold);

        #pragma omp taskwait
        _merge(array, temp, left, mid, right);
    }
}

void merge_sort(long *array, size_t n, int threshold) {
    long *temp = (long *)malloc(n * sizeof(long));
    #pragma omp parallel
    {
        #pragma omp single
            _parallel_merge_sort(array, temp, 0, n, threshold);
    }
    free(temp);
}

int is_sorted(long *array, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (array[i - 1] > array[i]) return 0;
    }
    return 1;
}

int main()
{
    printf("Array size: %d\n", ARR_LEN);

    long* array = create_array(ARR_LEN);
    init_array(array, ARR_LEN, 0xA77);

    double start = omp_get_wtime();

    merge_sort(array, ARR_LEN, MERGE_SORT_THRESHHOLD);

    double end = omp_get_wtime();

    printf("\n");
    printf("Calculation time: %lf\n", end - start);

    if (is_sorted(array, ARR_LEN)) {
        printf("Array is sorted.\n");
    } else {
        printf("Array is NOT sorted!\n");
    }
}
