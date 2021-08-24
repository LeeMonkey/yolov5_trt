#ifndef _YOLO_DETECTOR_H_
#define _YOLO_DETECTOR_H_

#include <iostream>
#include <chrono>
#include <cmath>
#include <memory>
#include <utils.h>
#include <cuda_utils.h>
#include <logging.h>
#include <common.h>
#include <network.h>

namespace ctdet
{
    class ctdetNet
    {
    	public:
    		ctdetNet(const std::string& engineName, const std::string& wtsName, int maxBatchSize=1);
    		ctdetNet(const std::string& engineName);
    		ctdetNet();
    		~ctdetNet()
    		{
    			if(mStream){
    				cudaStreamDestroy(mStream);
    				mStream = nullptr;
    			}
    
    			for (auto& buffer: mBuffers)
    			{
    				if(buffer)
    					CUDA_CHECK(cudaFree(buffer));
    				buffer = nullptr;
    			}
    
    			if(mContext){
    				mContext->destroy();
    				mContext = nullptr;
    			}
    
    			if(mEngine){
    				mEngine->destroy();
    				mEngine = nullptr;
    			}
    
    			if(mRuntime){
    				mRuntime->destroy();
    				mRuntime = nullptr;
    			}
    		}
    		void forward(const cv::Mat& image, std::vector<Detection>& boxes);
			void doInference(const void* inputData, void* outputData);
			void printLayerTimes(size_t num_interations);

		public:
			int outputBufferSize;
    
    	private:
    		IRuntime* mRuntime;
    		std::shared_ptr<YoloPluginFactory> mFactory;
    		ICudaEngine* mEngine;
    		IExecutionContext* mContext;
    		cudaStream_t mStream;
    		std::vector<void*> mBuffers;
    		std::map<std::string, size_t> mBufferSizeMap;
    };
}

#endif
