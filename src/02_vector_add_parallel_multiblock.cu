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

/*
This code expands on the previous implemantion which was only using a single thread block

Optimal parallelization on GPUs should be based on both thread blocks and threads

In previous approach we gave each initialized thread a loop to iterate over

This time we want to initialize enough threads to do all the computations at once
*/

#define new(dtype, length) (dtype*)malloc(sizeof(dtype) * length)
#define N 10000000
#define BILLION 1E9
#define MAX_ERR 1e-6

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>


__global__ void vector_add(float *out, float *a, float *b, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid <= N){
        out[tid] = a[tid] + b[tid];
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

int discover_number_of_gpu_thread_blocks(int array_size, int thread_number){
    float   float_division;
    int     int_division;

    float_division = (float)array_size / (float)thread_number; // this casting division works, cool
    int_division = array_size / thread_number;
    // printf("%f, %i\n", float_division, int_division);

    float difference = float_division - int_division;
    if (difference != 0){
        return int_division + 1;
    }
    else{
        return int_division;
    }
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

    // Discover the number of thread blocks needed
    // int number_of_gpu_thread_blocks_needed = discover_number_of_gpu_thread_blocks(N, 256);
    // printf("Assigned %i thread blocks\n", number_of_gpu_thread_blocks_needed);

    // Easier, more brutal method
    int block_size = 256;
    int grid_size = (N + block_size) / block_size;

    // Run kernel
    vector_add<<<grid_size, 256>>>(d_out, d_a, d_b, N);

    // Move results from GPU to RAM
    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);

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
