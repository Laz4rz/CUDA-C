#define N 10000000

#include <stdio.h>
#include <stdlib.h>

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
    // Allocate memory location of given size with type float to each pointer
    float *a   = malloc(sizeof(float) * N);
    float *b   = malloc(sizeof(float) * N);
    float *out = malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    vector_add(out, a, b, N);

    // Print first 5 of a, b, out
    printf("a: ");
    show_n(a, 5);
    printf("\nb: ");
    show_n(b, 5);
    printf("\nout: ");
    show_n(out, 5);
}