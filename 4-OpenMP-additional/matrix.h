#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

enum {
        NS_PER_SECOND = 1000000000,
        MAGIC_KEY = 0xDEAD10CC,
        MATRIX_ELEM_MAX = 100
    };

long* create_matrix(size_t dim)
{
    const int prot_flags = PROT_READ|PROT_WRITE;
    const int map_flags = MAP_PRIVATE|MAP_ANON; // MAP_POPULATE
    void* ptr = mmap(NULL, sizeof(long)*dim*dim, prot_flags, map_flags, -1, 0);
    if(ptr == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }

    return (long*)ptr;
}

void delete_matrix(long* matrix, size_t dim)
{
    munmap(matrix, sizeof(long)*dim*dim);
}

void init_matrix(long* matrix, size_t dim, unsigned int seed)
{
    srand(seed);

    for (size_t i = 0; i < dim*dim; ++i) {
        matrix[i] = rand() % MATRIX_ELEM_MAX;
    }
}

void transpose_matrix(long* A, long* T, size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            T[j*dim + i] = A[i*dim + j];
        }
    }
}

void print_matrix(long* matrix, size_t dim)
{
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            printf("%ld ", matrix[i*dim + j]);
        }
        printf("\n");
    }
}

unsigned int hash_matrix(long* matrix, size_t dim)
{
    unsigned int hash = 0;
    for (size_t i = 0; i < dim*dim; ++i) {
        hash += (i % dim) * ((long)matrix[i] ^ MAGIC_KEY);
    }

    return hash;
}
