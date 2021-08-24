#include <iostream>
#include <chrono>
#include <cmath>
#include <config.h>
#include <network.h>

//#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32

int get_width(int x, float gw, int divisor) {
    return  int(ceil((x * gw) / divisor)) * divisor;
}

int get_depth(int x, float gd) {
    if (x == 1) {
        return 1;
    } else {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, DataType dt, float& gd, float& gw, const std::string& wts_name) {

	ConfigManager &config = ConfigManager::getInstance();

    INetworkDefinition* network = builder->createNetwork();

    ITensor* data = network->addInput(config.inputBlobName.c_str(), dt, Dims3{config.height, config.width, config.channel});
    assert(data);

	IShuffleLayer* shuffleLayer = network->addShuffle(*data);
	shuffleLayer->setReshapeDimensions(Dims3{config.channel, config.height, config.width});
	shuffleLayer->setFirstTranspose(Permutation{2, 0, 1});
	shuffleLayer->setName("shuffle_BGR2RGB");

	float* scale = new float[1];
	scale[0] = 1 / 255.;
	Weights scalewts{DataType::kFLOAT, nullptr, 1};
	scalewts.values = scale;
	Weights shiftwts{DataType::kFLOAT, nullptr, 0};

	float* power = new float[1];
	power[0] = 1.;
	Weights powerwts{DataType::kFLOAT, nullptr, 1};
	powerwts.values = power;

	auto scale_input = network->addScale(*shuffleLayer->getOutput(0), ScaleMode::kUNIFORM, shiftwts, scalewts, powerwts);
	scale_input->setName("scale_div255");

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *scale_input->getOutput(0), 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

	Weights emptywts{DataType::kFLOAT, nullptr, 0};
	float *upsample11_val = new float[conv10->getOutput(0)->getDimensions().d[0] * 2 * 2];
	for(int i = 0; i < conv10->getOutput(0)->getDimensions().d[0] * 2 * 2; ++i){
		upsample11_val[i] = 1.f;
	}
	Weights upsample11_wts{DataType::kFLOAT, 
		                   nullptr, 
						   conv10->getOutput(0)->getDimensions().d[0] * 2 * 2};
	upsample11_wts.values = upsample11_val;

	auto upsample11 = network->addDeconvolution(*conv10->getOutput(0), 
			                                    conv10->getOutput(0)->getDimensions().d[0], 
												DimsHW{2, 2},
												upsample11_wts,
												emptywts);
	upsample11->setStride(DimsHW{2, 2});
	upsample11->setNbGroups(conv10->getOutput(0)->getDimensions().d[0]);
	upsample11->setName("upsample11");

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
	cat12->setName("concat12");

    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

	float* upsample15_val = new float[conv14->getOutput(0)->getDimensions().d[0] * 2 * 2];
	for (int i = 0; i < conv14->getOutput(0)->getDimensions().d[0] * 2 * 2; ++i)
		upsample15_val[i] = 1.f;

	Weights upsample15_wts{DataType::kFLOAT, 
		                   nullptr, 
						   conv14->getOutput(0)->getDimensions().d[0] * 2 * 2};
	upsample15_wts.values = upsample15_val;
	auto upsample15 = network->addDeconvolution(*conv14->getOutput(0), 
			                                    conv14->getOutput(0)->getDimensions().d[0], 
												DimsHW{2, 2},
												upsample15_wts,
												emptywts);
	upsample15->setStride(DimsHW{2, 2});
	upsample15->setNbGroups(conv14->getOutput(0)->getDimensions().d[0]);
	upsample15->setName("upsample15");

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
	cat16->setName("concat16");

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolution(*bottleneck_csp17->getOutput(0), config.numAnchor * (config.className.size() + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
	det0->setStride(DimsHW{1, 1});

    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
	cat19->setName("concat19");

    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolution(*bottleneck_csp20->getOutput(0), config.numAnchor * (config.className.size() + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
	det1->setStride(DimsHW{1, 1});

    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
	cat22->setName("concat22");

    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolution(*bottleneck_csp23->getOutput(0), config.numAnchor * (config.className.size() + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);
	det2->setStride(DimsHW{1, 1});
#if 0
	det2->getOutput(0)->setName("det2_tensor");
    network->markOutput(*det2->getOutput(0));

	det1->getOutput(0)->setName("det1_tensor");
    network->markOutput(*det1->getOutput(0));

	det0->getOutput(0)->setName("det0_tensor");
    network->markOutput(*det0->getOutput(0));
#endif

    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});

    yolo->getOutput(0)->setName(config.outputBlobName.c_str());
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
	builder->setFp16Mode(true);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, DataType dt, float& gd, float& gw, const std::string& wts_name) {
	ConfigManager &config = ConfigManager::getInstance();

    INetworkDefinition* network = builder->createNetwork();

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(config.inputBlobName.c_str(), dt, Dims3{config.channel, config.height, config.width});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
    auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
    auto spp10 = SPP(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), 3, 5, 7, "model.10");
    auto c3_11 = C3(network, weightMap, *spp10->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.11");

    /* ------ yolov5 head ------ */
    auto conv12 = convBlock(network, weightMap, *c3_11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");

	Weights emptywts{DataType::kFLOAT, nullptr, 0};
	std::vector<float> upsample13_val(conv12->getOutput(0)->getDimensions().d[0] * 2 * 2, 1.0f);
	Weights upsample13_wts{DataType::kFLOAT, 
		                   const_cast<const float*>(upsample13_val.data()), 
						   static_cast<int64_t>(upsample13_val.size())};
	auto upsample13 = network->addDeconvolution(*conv12->getOutput(0), 
			                                    conv12->getOutput(0)->getDimensions().d[0], 
												DimsHW{2, 2},
												upsample13_wts,
												emptywts);
	upsample13->setStride(DimsHW{2, 2});
	upsample13->setNbGroups(conv12->getOutput(0)->getDimensions().d[0]);
	upsample13->setName("upsample13");


    ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
    auto cat14 = network->addConcatenation(inputTensors14, 2);
    auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

    auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
	
	std::vector<float> upsample17_val(conv16->getOutput(0)->getDimensions().d[0] * 2 * 2, 1.0f);
	Weights upsample17_wts{DataType::kFLOAT, 
		                   const_cast<const float*>(upsample17_val.data()), 
						   static_cast<int64_t>(upsample17_val.size())};
	auto upsample17 = network->addDeconvolution(*conv16->getOutput(0), 
			                                    conv16->getOutput(0)->getDimensions().d[0], 
												DimsHW{2, 2},
												upsample17_wts,
												emptywts);
	upsample17->setStride(DimsHW{2, 2});
	upsample17->setNbGroups(conv16->getOutput(0)->getDimensions().d[0]);
	upsample17->setName("upsample17");

    ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
    auto cat18 = network->addConcatenation(inputTensors18, 2);
    auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

    auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");

	std::vector<float> upsample21_val(conv20->getOutput(0)->getDimensions().d[0] * 2 * 2, 1.0f);
	Weights upsample21_wts{DataType::kFLOAT, 
		                   const_cast<const float*>(upsample21_val.data()), 
						   static_cast<int64_t>(upsample21_val.size())};
	auto upsample21 = network->addDeconvolution(*conv20->getOutput(0), 
			                                    conv20->getOutput(0)->getDimensions().d[0], 
												DimsHW{2, 2},
												upsample21_wts,
												emptywts);
	upsample21->setStride(DimsHW{2, 2});
	upsample21->setNbGroups(conv20->getOutput(0)->getDimensions().d[0]);
	upsample21->setName("upsample21");

    ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors21, 2);
    auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

    auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
    ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
    auto cat25 = network->addConcatenation(inputTensors25, 2);
    auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

    auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
    ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
    auto cat28 = network->addConcatenation(inputTensors28, 2);
    auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

    auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
    ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
    auto cat31 = network->addConcatenation(inputTensors31, 2);
    auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolution(*c3_23->getOutput(0), 3 * (config.className.size() + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
	det0->setName("det0");

    IConvolutionLayer* det1 = network->addConvolution(*c3_26->getOutput(0), 3 * (config.className.size() + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
	det1->setName("det1");

    IConvolutionLayer* det2 = network->addConvolution(*c3_29->getOutput(0), 3 * (config.className.size() + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
	det2->setName("det2");

    IConvolutionLayer* det3 = network->addConvolution(*c3_32->getOutput(0), 3 * (config.className.size() + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);
	det3->setName("det3");

    auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
    yolo->getOutput(0)->setName(config.outputBlobName.c_str());
    network->markOutput(*yolo->getOutput(0));


    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
	builder->setFp16Mode(true);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

#if 0
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    if (is_p6) {
        engine = build_engine_p6(maxBatchSize, builder, DataType::kFLOAT, gd, gw, wts_name);
    } else {
        engine = build_engine(maxBatchSize, builder, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}
#endif
