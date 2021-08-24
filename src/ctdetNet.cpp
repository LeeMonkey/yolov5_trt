#include <utils.h>
#include <ctdetNet.h>

static Logger gLogger;
#if 1
struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes(size_t num_interations=1)
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / num_interations);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / num_interations);
    }

} gProfiler;
#endif
namespace ctdet 
{
    ctdetNet::ctdetNet(const std::string& engineName, const std::string& wtsName, int maxBatchSize):mRuntime(nullptr),
        mEngine(nullptr), mContext(nullptr), mStream(nullptr)
    {
        if(!isFileExists_access(wtsName))
            LOGERROR("no such weights path:%s", wtsName.c_str());
        assert(engineName.size());
    
        ConfigManager& config = ConfigManager::getInstance();
    
        // Create builder
        IBuilder* builder = createInferBuilder(gLogger);
    
        // Create model to populate the network, then set the outputs and create an engine
        ICudaEngine* engine = build_engine(maxBatchSize, 
                                           builder, 
                                           DataType::kFLOAT, 
                                           config.depthMultiple, 
                                           config.widthMultiple, 
                                           wtsName);
        assert(engine != nullptr);
    
        // Serialize the engine
        IHostMemory* modelStream = engine->serialize();
    
        std::ofstream ofs(engineName, std::ios::binary);
        if(!ofs.good())
        {
            LOGERROR("could not open plan output file: `%s`", engineName.c_str());
            exit(-1);
        }
        ofs.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        ofs.close();
    
        // Close everything down
        modelStream->destroy();
        engine->destroy();
        builder->destroy();
    }

    ctdetNet::ctdetNet():mRuntime(nullptr),
        mEngine(nullptr), mContext(nullptr), mStream(nullptr)
    {
        ConfigManager& config = ConfigManager::getInstance();
        ctdetNet(config.enginePath);
    }
    
    ctdetNet::ctdetNet(const std::string& engineName):mRuntime(nullptr),
        mEngine(nullptr), mContext(nullptr), mStream(nullptr)
    {
        if(!isFileExists_access(engineName))
            LOGERROR("no such weights path:%s", engineName.c_str());
    
        ConfigManager& config = ConfigManager::getInstance();
        // read engine file
        std::ifstream ifs(engineName, std::ios::binary);
        if(!ifs.good())
        {
            LOGERROR("read %s error!", engineName.c_str());
            exit(-1);
        }
    
        ifs.seekg(0, ifs.end);
        size_t size = ifs.tellg();
        ifs.seekg(0, ifs.beg);
    
        char *engineStream = new char[size];
        assert(engineStream);
        ifs.read(engineStream, size);
        ifs.close();
    
        mRuntime = createInferRuntime(gLogger);
        assert(mRuntime);
        mFactory = std::make_shared<YoloPluginFactory>();
        assert(mFactory);
        mEngine = mRuntime->deserializeCudaEngine(engineStream, size, mFactory.get());
        assert(mEngine);
        mContext = mEngine->createExecutionContext();
        mContext->setProfiler(&gProfiler);
    
        CUDA_CHECK(cudaStreamCreate(&mStream));
    
        mBuffers.resize(mEngine->getNbBindings());
        for(int i = 0; i < mEngine->getNbBindings(); ++i) {
            std::string name{mEngine->getBindingName(i)};
            if(!strcmp(mEngine->getBindingName(i), config.inputBlobName.c_str())){
                mBufferSizeMap[config.inputBlobName] = config.width * config.height * config.channel * sizeof(float);
                CUDA_CHECK(cudaMalloc(&mBuffers[i], mBufferSizeMap[config.inputBlobName]));
            }
            else if(!strcmp(mEngine->getBindingName(i), config.outputBlobName.c_str())){
                mBufferSizeMap[config.outputBlobName] = config.maxOutputBBoxCount * sizeof(Detection) + sizeof(float);
                CUDA_CHECK(cudaMalloc(&mBuffers[i], mBufferSizeMap[config.outputBlobName]));
#if 0
                Dims dims = mEngine->getBindingDimensions(i);
                std::cout << dims.d[0] << " "
                          << dims.d[1] << " "
                          << dims.d[2] << " "
                          << dims.d[3] << std::endl;
#endif
            }
            else{
                Dims dims = mEngine->getBindingDimensions(i);
                std::string name{mEngine->getBindingName(i)};
                mBufferSizeMap[name] = dims.d[0] * dims.d[1] * dims.d[2] * sizeof(float); 
#if 0
                //std::cout << dims.d[0] << " "
                //          << dims.d[1] << " "
                //          << dims.d[2] << " "
                //          << dims.d[3] << " nbytes:"
                //        << mBufferSizeMap[name] << std::endl;
#endif
                CUDA_CHECK(cudaMalloc(&mBuffers[i], mBufferSizeMap[name]));
            }
        }
        //assert(inputIndex < 2 and outputIndex < 2);
        outputBufferSize = mBufferSizeMap[config.outputBlobName];
    
        delete[] engineStream;
    }
    
    void ctdetNet::forward(const cv::Mat& image, std::vector<Detection>& boxes){
        ConfigManager& config = ConfigManager::getInstance();
        float ratio = 1.f;
        float pad_w = 0.f;
        float pad_h = 0.f;
        cv::Mat prep_image = preprocess_img(image, config.width, config.height, ratio, pad_w, pad_h);
        //prep_image.setTo(cv::Scalar(1.f, 1.f, 1.f));
    
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        const int inputIndex = mEngine->getBindingIndex(config.inputBlobName.c_str());
        const int outputIndex = mEngine->getBindingIndex(config.outputBlobName.c_str());
        CUDA_CHECK(cudaMemcpyAsync(mBuffers[inputIndex], 
                                   prep_image.data, 
                                   mBufferSizeMap[config.inputBlobName], 
                                   cudaMemcpyHostToDevice, 
                                   mStream));
        mContext->enqueue(1, &mBuffers[inputIndex], mStream, nullptr);
#if 0 
        for(int i = 0; i < mEngine->getNbBindings(); ++i) {
            if(!strcmp(mEngine->getBindingName(i), config.inputBlobName.c_str())){
                continue;
            }
            else if(!strcmp(mEngine->getBindingName(i), config.outputBlobName.c_str())){
                continue;
            }
            std::string name{mEngine->getBindingName(i)};
            float * inter = reinterpret_cast<float*>(malloc(mBufferSizeMap[name]));
            CUDA_CHECK(cudaMemcpyAsync(inter, 
                                       mBuffers[i], 
                                       mBufferSizeMap[name],
                                       cudaMemcpyDeviceToHost, 
                                       mStream));
            Dims dims = mEngine->getBindingDimensions(i);
            LOGINFO("%s", name.c_str());
            for(int i = 0; i < dims.d[0]; ++i){
                for(int j = 0; j < dims.d[1]; ++j){
                    for(int k = 0; k < dims.d[2]; ++k){
                        std::cout << inter[i * dims.d[1] * dims.d[2] + j * dims.d[2] + k]
                                  << " ";
                    }
                    std::cout << std::endl;
                    goto stop;
                }
            }
            stop: "";
            free(inter);
        }
#endif

        float output[mBufferSizeMap[config.outputBlobName]/ sizeof(float)];
        CUDA_CHECK(cudaMemcpyAsync(output, 
                                   mBuffers[outputIndex], 
                                   mBufferSizeMap[config.outputBlobName],
                                   cudaMemcpyDeviceToHost, 
                                   mStream));
        cudaStreamSynchronize(mStream);
#if 0
        LOGINFO("%s", config.outputBlobName.c_str());
        for(int i = 0; i < 50; ++i){
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
#endif

        nms(boxes, output, config.confThresh, config.nmsThresh);

        const size_t num = std::min(static_cast<size_t>(output[0]), config.maxOutputBBoxCount);
        boxes.resize(num);
        memcpy(boxes.data(), &output[1], num * sizeof(Detection));
        
        postProcess(boxes, image.cols, image.rows);
    }

    void ctdetNet::doInference(const void* inputData, void* outputData) 
    {
        ConfigManager& config = ConfigManager::getInstance();

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        const int inputIndex = mEngine->getBindingIndex(config.inputBlobName.c_str());
        const int outputIndex = mEngine->getBindingIndex(config.outputBlobName.c_str());
        CUDA_CHECK(cudaMemcpyAsync(mBuffers[inputIndex], 
                                   inputData, 
                                   mBufferSizeMap[config.inputBlobName], 
                                   cudaMemcpyHostToDevice, 
                                   mStream));
        mContext->enqueue(1, &mBuffers[inputIndex], mStream, nullptr);
    
        CUDA_CHECK(cudaMemcpyAsync(outputData, 
                                   mBuffers[outputIndex], 
                                   mBufferSizeMap[config.outputBlobName],
                                   cudaMemcpyDeviceToHost, 
                                   mStream));
        cudaStreamSynchronize(mStream);
    }
#if 1
    void ctdetNet::printLayerTimes(size_t num_interations)
    {
        gProfiler.printLayerTimes(num_interations);
    }
#endif
}
