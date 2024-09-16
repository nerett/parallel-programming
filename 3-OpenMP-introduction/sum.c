#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv)
{
    if (argc != 2 && !argv[1]) {
        fprintf(stderr, "Usage: a.out <N>\n");
        exit(EXIT_FAILURE);
    }
    unsigned long N = strtoul(argv[1], argv + argc, 10);
    printf("Calculating sum of (1/n) from 1 to N=%lu\n", N);

    long double result = 0;
    #pragma omp parallel for reduction(+:result)
        for (unsigned long n = 1; n < N; n++) {
            result += 1.0/n;
        }

    printf("Result: %Lf\n", result);

    return 0;
}