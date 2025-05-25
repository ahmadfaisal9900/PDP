#ifdef _WIN32
#include <kernel.hpp>	// only for syntax highlighting
#endif
#define TYPE float

kernel void add_kernel(global TYPE * A, global TYPE * B, global TYPE * C){
    const int n = get_global_id(0);
    C[n] = A[n] + B[n];
}