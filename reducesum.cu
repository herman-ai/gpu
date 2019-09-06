#include <iostream>
#include <cooperative_groups.h>

using namespace std;
using namespace cooperative_groups;

const int N = 1<<27;

__device__
int warp_reduce(int val) {
	for (int i=warpSize/2; i>0; i/=2) {
		val += __shfl_down_sync(0xffffffff, val, i);
	}
	return val;
}

__device__
int thread_sum(int * a) {
	int4 v;
	int sum = 0;
	for (int i=threadIdx.x+blockIdx.x*blockDim.x;
		i < N/4; i += gridDim.x*blockDim.x) {
		v = reinterpret_cast<int4*>(a)[i];
		sum += v.x + v.y + v.z + v.w;
	}
	return sum;
}

__global__
void reduce_sum(int * a, int *sum) {
	__shared__ int temp[32];

	int mysum = thread_sum(a);
	
	mysum = warp_reduce(mysum);

	int wid = threadIdx.x / warpSize;
	int tid = threadIdx.x % warpSize;

	if (tid == 0) {
		temp[wid] = mysum;
	}

	__syncthreads();

	mysum = (threadIdx.x < blockDim.x / warpSize)?temp[tid] : 0;

	if (wid==0) {  //only one warp is needed for this reduction
		mysum = warp_reduce(mysum);
	}	

	if (threadIdx.x == 0) {
		atomicAdd(sum, mysum);
	}	
}

int main() {
	int *a;
	int *sum;
	cudaMallocManaged(&a, sizeof(int)*N);
	cudaMallocManaged(&sum, sizeof(int));

	for (int i=0; i<N; i++) {
		a[i] = 1;
	}

	reduce_sum<<<1<<2, 256>>>(a, sum);

	cudaDeviceSynchronize();

	cout << "The sum is " << *sum << endl;

	cudaFree(a);
	cudaFree(sum);
	return 0;
}
