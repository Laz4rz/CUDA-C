#include <stdio.h>

/*
__global__ specifier indicates that a function will run on GPU
these functions are also called kernels
*/
__global__ void cuda_hello()
{
    printf("hello from GPU\n");
}

int main()
{
    cuda_hello<<<1, 1>>>(); // <<x, y>> is called a kernel launch, x and y values specify the numbers of blocks and grids for each kernel
    return 0;
}
