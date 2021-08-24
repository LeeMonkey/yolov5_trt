#ifndef _YOLO_NETWORK_H_
#define _YOLO_NETWORK_H_

#include <iostream>
#include <chrono>
#include <cmath>
#include <config.h>
#include <cuda_utils.h>
#include <common.h>
#include <utils.h>

inline int get_width(int x, float gw, int divisor = 8);

inline int get_depth(int x, float gd);

ICudaEngine* build_engine(unsigned int maxBatchSize, 
                          IBuilder* builder, 
                          DataType dt, 
                          float& gd, 
                          float& gw, 
                          const std::string& wts_name);

ICudaEngine* build_engine_p6(unsigned int maxBatchSize, 
                             IBuilder* builder, 
                             DataType dt, 
                             float& gd, 
                             float& gw, 
                             const std::string& wts_name);

#if 0
void APIToModel(unsigned int maxBatchSize, 
                IHostMemory** modelStream, 
                bool& is_p6, 
                float& gd, 
                float& gw, 
                std::string& wts_name);
void doInference(IExecutionContext& context, 
                 cudaStream_t& stream, 
                 void **buffers, 
                 float* input, 
                 float* output, 
                 int batchSize);
#endif

bool parse_args(int argc, 
                char** argv, 
                std::string& wts, 
                std::string& engine, 
                bool& is_p6, 
                float& gd, 
                float& gw, 
                std::string& img_dir);
#endif
