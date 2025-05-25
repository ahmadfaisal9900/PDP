#ifdef _WIN32
#include <kernel.hpp>	// only for syntax highlighting
#endif
#define TYPE float

kernel void scalar_mult(global TYPE * A, global TYPE * B, int scalar, int N){
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    // Calculate linear index for 2D matrix
    int idx = x + y * (int)N;

    if(x < (int)N && y < (int)N){
        B[idx] = A[idx] * (scalar);
    }
}