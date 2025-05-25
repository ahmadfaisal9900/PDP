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

    Memory<float>A (device, N*N);
    Memory<float>B (device, N*N);
    const int scalar = 5;
    
    for(int i = 0; i<N*N; i++){
        A[i] = 2.0f;
        B[i] = 0.0f;
    }

    A.write_to_device();
    B.write_to_device();
    
    Kernel scalar_mult(device, N, "scalar_mult", A, B, scalar, N);

    scalar_mult.run();

    B.read_from_device();

    print_info("The first of element of B is "+to_string(B[0]));

}