#define N 10000000
#define new(dtype,length) (dtype*)malloc(sizeof(dtype) * length)
#define BILLION 1E9

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

__global__ void vector_add(float *out, float *a, float *b, int n)
{
    for (int i = 0; i < n; i++)
    {
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

/*
In order to change the computation unit from CPU to GPU there is 3 things to do:
- chanege the definition of kernel function
- move the data to GPU memory
- change the call of this kernel function to include <<<x, y>>> (grid and block sizes)

After executing the kernel data has to be moved back to host memory (RAM) for further manipulations
*/

int main(void)
{
    // Initialize time variables
    struct timespec start, end;
    uint64_t diff;

    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Allocate memory for vectors
    // float *a    = (float*)malloc(sizeof(float) * N);
    // float *b    = (float*)malloc(sizeof(float) * N);
    // float *out  = (float*)malloc(sizeof(float) * N);
    float *a    = new(float, N);
    float *b    = new(float, N);
    float *out  = new(float, N);

    // Populate vectors
    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0; b[i] = 2.0; 
    }

    // Allocate GPU memory
    float *d_a, *d_b, *d_out;
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Move data to allocated GPU memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Add vectors 
    vector_add<<<1, 1>>>(d_out, d_a, d_b, N);

    // Move data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Stop timer
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate time difference
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

    // Show first N values of vectors
    printf("a: ");
    show_n(a, 5);
    printf("\nb: ");
    show_n(b, 5);
    printf("\nout: ");
    show_n(out, 5);

    // FLOPS
    printf("\nFLOPS: %.3f\n", (float)N / (float)diff);

    // Free memory
    free(a); free(b); free(out);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}
