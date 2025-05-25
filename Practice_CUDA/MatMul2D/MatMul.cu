#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul(float * A, float * B, float * C, int N){

    int rows = blockIdx.x * blockDim.x + threadIdx.x;
    int cols = blockIdx.y * blockDim.y + threadIdx.y;

    if (rows < N && cols < N){
        float sum = 0.0f;
        for(int k = 0; k<N; k++){
            sum+= A[rows * N + k] * B[cols + (k * N)];
        }
        C[cols + (rows * N)] = sum;
    }
}

int main(){

    const int N = 1024;

    float * hA = new float[N*N];
    float * hB = new float[N*N];
    float * hC = new float[N*N];

    for(int i = 0; i<N*N; i++){
        hA[i] = 2.0f;
        hB[i] = 3.0f;
        hC[i] = 0.0f;
    }

    float * dA, *dB, *dC;
    cudaMalloc(&dA, N*N * sizeof(float));
    cudaMalloc(&dB, N*N * sizeof(float));
    cudaMalloc(&dC, N*N * sizeof(float));

    cudaMemcpy(dA, hA, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 BlockSize(8, 8);
    
    dim3 ThreadsPerBlock(16, 16);

    matmul<<<BlockSize, ThreadsPerBlock>>>(dA, dB, dC, N);

    cudaMemcpy(hC, dC, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    for(int i = 0; i<5; i++){
        std::cout << hC[i] << std::endl;
    }
    delete[] hA;
    delete[] hB;
    delete[] hC;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}