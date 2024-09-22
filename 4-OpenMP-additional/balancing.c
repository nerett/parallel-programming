#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define ITERATIONS 65
#define THREADS 4
#define CHUNK 4
#define mark_threads()                                 \
        for (int i = 0; i < ITERATIONS; i++) {         \
            job_table[omp_get_thread_num()][i] = '+';  \
        }                                              \

void print_thread_matrix(char job_table[THREADS][ITERATIONS])
{
    for (int i = 0; i < THREADS; i++) {
        printf("Thread %d: ", i);
        for (int j = 0; j < ITERATIONS; j++) {
            printf("%c", job_table[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    printf("CHUNK=%d, THREADS=%d\n\n", CHUNK, THREADS);
    omp_set_num_threads(THREADS);
    char job_table[THREADS][ITERATIONS] = { 0 };


    printf("No scheduling\n");
    memset(job_table, '-', sizeof(char) * ITERATIONS * THREADS);
    #pragma omp parallel for
        mark_threads();
    print_thread_matrix(job_table);

    printf("Balancing: static\n");
    memset(job_table, '-', sizeof(char) * ITERATIONS * THREADS);
    #pragma omp parallel for schedule(static, CHUNK)
        mark_threads();
    print_thread_matrix(job_table);

    printf("Balancing: dynamic\n");
    memset(job_table, '-', sizeof(char) * ITERATIONS * THREADS);
    #pragma omp parallel for schedule(dynamic, CHUNK)
        mark_threads();
    print_thread_matrix(job_table);

    printf("Balancing: guided\n");
    memset(job_table, '-', sizeof(char) * ITERATIONS * THREADS);
    #pragma omp parallel for schedule(guided, CHUNK)
        mark_threads();
    print_thread_matrix(job_table);

    return 0;
}
