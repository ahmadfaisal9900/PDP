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

    //select device 
    Device device(select_device_with_most_flops(), "kernels.cl");

    //define data and memory on device and host
    const int N = 1024;

    Memory<float> A(device, N);
    Memory<float> B(device, N);
    Memory<float> C(device, N);
    
    //add kernel
    Kernel add_kernel(device, N, "add_kernel", A, B, C);

    //init data and write to device 
    for(int i = 0; i<N; i++){
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    A.write_to_device();
    B.write_to_device();
    C.write_to_device();

    //start kernel 
    Clock.Start();
    add_kernel.run();
    Clock.Stop();

    //read from device
    C.read_from_device();

	print_info("Value after kernel execution: C[0] = "+to_string(C[0]));
}