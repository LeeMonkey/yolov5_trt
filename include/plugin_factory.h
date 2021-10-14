/*************************************************************************
	> File Name: include/plugin_factory.h
	> Author: Lee
	> Created Time: 2021年10月12日 星期二 17时27分44秒
 ************************************************************************/
#ifndef __PLUGIN_FACTORY_H_
#define __PLUGIN_FACTORY_H_

#include <vector>
#include <upsample_layer.h>
#include <yololayer.h>
#include <NvInferPlugin.h>
#include <NvCaffeParser.h>
namespace nvinfer1
{
   class PluginFactory: public IPluginFactory
    {
        public:
            virtual IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
    };                    
}
#endif

