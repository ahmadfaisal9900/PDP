#ifdef _WIN32
#include <kernel.hpp>	// only for syntax highlighting
#endif
#define TYPE float

kernel void dot_product(global TYPE * A, global TYPE * B, global float * sum){
    const int n = get_global_id(0);
    sum[n] = A[n] * B[n];
}