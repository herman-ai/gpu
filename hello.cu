#include <assert.h>
const int N = 1<<10;

__global__ void cuda_hello(int * a){
	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i < N; i+=gridDim.x*blockDim.x) {
		a[i] *= 2;
	}
}

int main(void) {
    int *a;

    cudaMallocManaged(&a, sizeof(int)*N);

    for (int i=0; i<N; i++) {
        a[i] = i;
    }

    int blocks = N/256;
    cuda_hello<<<blocks,256>>>(a);

    cudaDeviceSynchronize();

    for (int i=0; i<N; i++) {
        assert(a[i] == i*2);
    }


    cudaFree(a);

    return 0;
}
