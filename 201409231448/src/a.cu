/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void ADD(float * A, float*O, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		O[i] = A[i] + threadIdx.x;
	}
}

int main(void) {
	int N = 1024 * 256;
	while (N < pow(2.0, 30)) {
		size_t size = N * sizeof(float);
		float* h_A = (float*) malloc(size);
		float* h_O = (float*) malloc(size);
		float* d_A;
		cudaMalloc((void**) &d_A, size);
		float* d_O;
		cudaMalloc((void**) &d_O, size);
		cudaEvent_t stop, stop1, stop2;
		cudaEvent_t start, start1, start2;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		for (int i = 0; i < N; i++) {
			h_A[i] = rand() / (float) RAND_MAX;
		}
		cudaEventRecord(start);
		cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float elapsedTime = 0;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		int threadsPerBlock = 256;
		int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
		cudaEventRecord(start1);
		ADD<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_O, N);
		cudaEventRecord(stop1);
		cudaDeviceSynchronize();
		cudaEventSynchronize(stop1);
		float elapsedTime1 = 0;
		cudaEventElapsedTime(&elapsedTime1, start1, stop1);
		cudaEventRecord(start2);
		cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop2);
		cudaEventSynchronize(stop2);
		float elapsedTime2 = 0;
		cudaEventElapsedTime(&elapsedTime2, start2, stop2);
		cudaFree(d_A);
		cudaFree(d_O);
		free(h_A);
		free(h_O);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaEventDestroy(start1);
		cudaEventDestroy(stop1);
		cudaEventDestroy(start2);
		cudaEventDestroy(stop2);
		if (N == 1024 *256) {
			printf("%s\n%s\n%s\n", "FIRSTNAME: XINYUN", "LASTNAME: LV",
					"E-MAIL: xinyunlv0425@gmail.com");
			printf("%-28s%-15s%-15s%-15s\n", "Elements(M)", " CPUtoGPU(ms)",
					" Kernel(ms)", " GPUtoCPU(ms)");
		}
		printf("%-30d%-15f%-15f%-15f\n", N / (1024 * 256), elapsedTime, elapsedTime1,
				elapsedTime2);

		N = N * 2;
	}
}
