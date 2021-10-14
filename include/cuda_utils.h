#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

//#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
                assert(0);                                                                     \
        }                                                                                      \
    }
#endif  // CUDA_CHECK

namespace Tn                                                                                                                       
{
    template<typename T>  
    void write(char*& buffer, const T& val)
    {   
            *reinterpret_cast<T*>(buffer) = val;
            buffer += sizeof(T);
        }   

    template<typename T>  
    void read(const char*& buffer, T& val)
    {   
            val = *reinterpret_cast<const T*>(buffer);
            buffer += sizeof(T);
        }   
}

#endif  // TRTX_CUDA_UTILS_H_

