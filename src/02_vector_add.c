#define N 10000000
#define BILLION 1000000000L // L - treat as long integer

#include <stdio.h>
#include <stdlib.h>
#include <time.h> // https://people.cs.rutgers.edu/~pxk/416/notes/c-tutorials/gettime.html
#include <stdint.h>

void vector_add(float *out, float *a, float *b, int n)
{
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

void show_n(float *v, int n)
{
    if (n > N)
    {
        printf("Your print is longer than the defined limit %i, cliped the output", N);
        n = N;
    }
    for (int i = 0; i < n; i++)
    {
        printf("%.3f ", v[i]);
    }
    printf("\n");
}

int main(void)
{
    uint64_t diff;
    struct timespec start, end;
    
    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Allocate memory location of given size with type float to each pointer
    float *a   = (float*)malloc(sizeof(float) * N);
    float *b   = (float*)malloc(sizeof(float) * N);
    float *out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    vector_add(out, a, b, N);

    // End timer
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Get time in seconds
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

    // Print first 5 of a, b, out
    printf("a: ");
    show_n(a, 5);
    printf("\nb: ");
    show_n(b, 5);
    printf("\nout: ");
    show_n(out, 5);

    // FLOPS
    printf("\nFLOPS: %f\n", (float)N / (float)diff);

    // Free memory
    free(a); free(b); free(out);
}