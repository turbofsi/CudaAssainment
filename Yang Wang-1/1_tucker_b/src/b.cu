/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 * Yang Wang
 * Department of ECE
 * University of Toronto
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void ADD(float * A, float*O, int N, float x)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
	{
		O[i] = A[i] + x;
	}
}

int main(void)
{
	int 		N = 32 * 1024 * 256;
	double 	input = 32;
	float 		x = 10.0;

	printf("%s\n%s\n%s\n", "FIRSTNAME: Yang", "LASTNAME: Wang",
						"E-MAIL: tucker.wang@mail.utoronto.ca");
	printf("%-15s%-15s%-15s%-15s%-15s\n", "Elements(M)", "Block_size", "CPUtoGPU(ms)",
						"Kernel(ms)", "GPUtoCPU(ms)");

	for(double threadsPerBlock = 16; threadsPerBlock < pow(2.0, 15); threadsPerBlock *= 2)
	{
		size_t 	size = N * sizeof(float);
		float * hA = (float *) malloc(size);
		float * hO = (float *) malloc(size);

		float * dA;
		cudaMalloc((void**) &dA, size);
		float * dO;
		cudaMalloc((void**) &dO, size);

		cudaEvent_t start, start_1, start_2;
		cudaEvent_t end, end_1, end_2;

		cudaEventCreate(&start);
		cudaEventCreate(&start_1);
		cudaEventCreate(&start_2);

		cudaEventCreate(&end);
		cudaEventCreate(&end_1);
		cudaEventCreate(&end_2);

		for (int i = 0; i < N; i++) {
			hA[i] = rand() / (float) RAND_MAX;
		}

		cudaEventRecord(start);
		cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		float eTime = 0;
		cudaEventElapsedTime(&eTime, start, end);

		int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

		cudaEventRecord(start_1);
		ADD<<<blocksPerGrid, threadsPerBlock>>>(dA, dO, N,x);
		cudaEventRecord(end_1);
		cudaDeviceSynchronize();
		cudaEventSynchronize(end_1);
		float eTime1 = 0;
		cudaEventElapsedTime(&eTime1, start_1, end_1);

		cudaEventRecord(start_2);
		cudaMemcpy(hO, dO, size, cudaMemcpyDeviceToHost);
		cudaEventRecord(end_2);
		cudaEventSynchronize(end_2);
		float eTime2 = 0;
		cudaEventElapsedTime(&eTime2, start_2, end_2);

		cudaFree(dA);
		cudaFree(dO);
		free(hA);
		free(hO);
		cudaEventDestroy(start);
		cudaEventDestroy(end);
		cudaEventDestroy(start_1);
		cudaEventDestroy(end_1);
		cudaEventDestroy(start_2);
		cudaEventDestroy(end_2);

		printf("%-15.0f%-15.0f%-15f%-15f%-15f\n", input, threadsPerBlock, eTime,
				eTime1, eTime2);

	}
}
