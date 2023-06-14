/*
<<<M, T>>> syntax is used to allocate:
    M thread blocks 
    T threads each, 
this is a way of parallelizing the work on gpu

previously used <<1, 1>> vector addition only used 1 thread to do the loop so we basically added the memory moving overhead
and did the same thing as the cpu before, withouth using the benefits of the gpu


    threadIdx.x contains the index of the thread within the block
    blockDim.x contains the size of thread block (number of threads in the thread block).

*/

#define new(dtype, length) (dtype*)malloc(sizeof(dtype) * length)
#define N 1000000
#define BILLION 1E9

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

__global__ void vector_add(float *out, float *a, float *b, int n){
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride){
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

int main(void){
    // Initialize time variables
    uint64_t diff;
    struct timespec start, end;
    
    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Allocate memory
    float *a = new(float, N);
    float *b = new(float, N);
    float *out = new(float, N);

    // Fill arrays
    for (int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate GPU memory
    float *d_a, *d_b, *d_out;
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Move from RAM to GPU memory
    cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);

    // Run kernel
    vector_add<<<1, 256>>>(d_out, d_a, d_b, N);

    // Move results from GPU to RAM
    cudaMemcpy(d_out, out, sizeof(float)*N, cudaMemcpyDeviceToHost);

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
