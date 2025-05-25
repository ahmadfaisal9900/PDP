#ifdef _WIN32
#include <kernel.hpp>	// only for syntax highlighting
#endif
#define TYPE float

kernel void matmul(global TYPE * A, global TYPE * B, global TYPE * C, int N){
    int row = get_global_id(0);
    int col = get_global_id(1);

    if(row < N && col < N){
        float sum = 0.0f;
        for(int k = 0; k<N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row*N + col] = sum;
    }
}