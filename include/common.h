#ifndef _YOLO_COMMON_H_
#define _YOLO_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <yololayer.h>

using namespace nvinfer1;

cv::Rect get_rect(cv::Mat& img, float bbox[4]);

float iou(float lbox[4], float rbox[4]);


// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file);

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps, DataType dataType=DataType::kFLOAT);

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, DataType dataType=DataType::kFLOAT); 

ILayer* focus(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname, DataType dataType=DataType::kFLOAT);

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname, DataType dataType=DataType::kFLOAT);

ILayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname, DataType dataType=DataType::kFLOAT);

ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname, DataType dataType=DataType::kFLOAT);

ILayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname, DataType dataType=DataType::kFLOAT);

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname);

ILayer* toTensor(INetworkDefinition *network, ITensor& input, DataType dataType=DataType::kFLOAT);

IPluginLayer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets);

#endif

