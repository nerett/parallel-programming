#include <stdio.h>
#include <omp.h>

int main()
{
    printf("omp_get_num_procs(): %d\n", omp_get_num_procs());

    return 0;
}