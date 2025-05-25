#include <opencl.hpp>
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// Timing
struct msClock
{
	typedef std::chrono::high_resolution_clock clock;
	std::chrono::time_point<clock> t1, t2;
	void Start() { t1 = high_resolution_clock::now(); }
	void Stop() { t2 = high_resolution_clock::now(); }
	double ElapsedTime()
	{
		duration<double, std::milli> ms_doubleC = t2-t1;
		return ms_doubleC.count();
	}
}
Clock;

int main(){
    Device device(select_device_with_most_flops(), "kernels.cl");
    
    const int N = 1024;
    Memory<float>A(device, N*N);
    Memory<float>B(device, N*N);
    Memory<float>C(device, N*N);

    for(int i = 0; i<N*N; i++){
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    A.write_to_device();
    B.write_to_device();

    Kernel MatMul(device, N, "matmul", A, B, C, N);

    MatMul.run();

    C.read_from_device();

    print_info("The first element of C is "+to_string(C[0]));
}