#define N 10000000

#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add(float *out, float *a, float *b, int n)
{
    for (int i = 0; i < n i++)
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
    // Allocate memory for vectors
    float *a    = malloc(sizeof(float) * N);
    float *b    = malloc(sizeof(float) * N);
    float *out  = malloc(sizeof(float) * N);

    // Populate vectors
    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0; b[i] = 2.0; 
    }

    // Allocate GPU memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_c, sizeof(float) * N);

    // Move data to allocated GPU memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Add vectors 
    vector_add<<<1, 1>>>(out, a, b, N);

    // Move data back to host memory
    cudaMemcpy(out, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Show first N values of vectors
    printf("a: ");
    show_n(a, 5);
    printf("\nb: ");
    show_n(b, 5);
    printf("\nout: ");
    show_n(out, 5);
}
