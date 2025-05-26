#include <iostream>
#include <cuda_runtime.h>

int main() {
    int count;
    cudaGetDeviceCount(&count);
    std::cout << "CUDA Device Count: " << count << std::endl;
    return 0;
}
