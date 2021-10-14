#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include <NvInfer.h>
#include <config.h>

static constexpr int LOCATIONS = 4;
static constexpr int ANCHOR_NUM = 3;

struct YoloKernel
{
    int width;
    int height;
    float anchors[ANCHOR_NUM * 2];
};

struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[LOCATIONS];
    float conf;
    float class_id;
};

namespace nvinfer1
{
    // class YoloLayerPlugin : public IPluginV2IOExt
    class YoloLayerPlugin : public IPluginExt
    {
    public:
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const char* name, const void* data, size_t length);

        ~YoloLayerPlugin();

        int getNbOutputs() const override;

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        int initialize() override;

        virtual void terminate() override;

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

        virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize();

        virtual void serialize(void* buffer);

        bool supportsFormat(DataType type, PluginFormat format) const override;

        void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;


    private:
        void forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize = 1);
        int mThreadCount = 512;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<YoloKernel> mYoloKernel;
        void** mAnchor;
    };
};
#endif 
