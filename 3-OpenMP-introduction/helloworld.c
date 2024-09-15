#include <stdio.h>
#include <omp.h>

int main()
{
    #pragma omp parallel
        printf("OpenMP thread %d/%d: Hello World!\n", omp_get_thread_num(), omp_get_num_threads());
    #pragma endregion

    return 0;
}