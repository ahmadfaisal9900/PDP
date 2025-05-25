#include <iostream>
#include <cuda_runtime.h>

#define TYPE float
#define N 256

__global__ void HelloWorld(TYPE* A, TYPE* B, TYPE* C){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(){

    float * hA = new float[N];
    float * hB = new float[N];
    float * hC = new float[N];

    for(int i=0; i<N; i++){
        hA[i] = 1.0f;
        hB[i] = 2.0f;
        hC[i] = 0.0f;
    }

    float * dA, *dB, *dC;

    cudaMalloc(&dA, N * sizeof(float));
    cudaMalloc(&dB, N * sizeof(float));
    cudaMalloc(&dC, N * sizeof(float));


    cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, N * sizeof(float), cudaMemcpyHostToDevice);

    int BlockDim = 1;
    int threadPerBlock = 128;
    HelloWorld<<<BlockDim, threadPerBlock>>>(dA, dB, dC);

    cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First element " << hC[0] << std::endl;

    delete[] hA;
    delete[] hB;
    delete[] hC;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

}