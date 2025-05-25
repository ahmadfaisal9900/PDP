#include <opencl.hpp>
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

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

    //define the device
    Device device(select_device_with_most_flops(), "kernels.cl");

    //define memory overall and size
    const int N = 1024;

    Memory<float> A(device, N);
    Memory<float> B(device, N);
    Memory<float> sum(device, N);

    for(int i = 0; i<N; i++){
        A[i] = 1.0f;
        B[i] = 2.0f;
        sum[i] = 0.0f;
    }

    Kernel dot_product(device, N, "dot_product", A, B, sum);

    //transfer to kernel
    A.write_to_device();
    B.write_to_device();
    sum.write_to_device();
    //run kernel

    dot_product.run();
    //read from kernel
    sum.read_from_device();
    //print
    float actual_sum = 0;
    for(int i = 0; i<N; i++){
        actual_sum += sum[i];
    }
    print_info("Actual Sum is "+to_string(actual_sum));
}