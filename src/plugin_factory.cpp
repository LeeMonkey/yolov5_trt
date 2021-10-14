/*************************************************************************
	> File Name: src/plugin_factory.cpp
	> Author: Lee
	> Created Time: 2021年10月12日 星期二 17时40分44秒
 ************************************************************************/
#include <plugin_factory.h>
#include <yololayer.h>
#include <upsample_layer.h>
namespace nvinfer1
{
    IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
    {   
        if(std::string(layerName).find("yolo") != std::string::npos)
        {
            YoloLayerPlugin *ylp =  new YoloLayerPlugin(layerName, serialData, serialLength);
            return ylp;
        }
        else if(std::string(layerName).find("upsample") != std::string::npos)
        {
            UpsampleLayerPlugin *ulp = new UpsampleLayerPlugin(serialData, serialLength);
            return ulp;
        }
        return nullptr;
    }       

}

