#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <common.h>
#include <yololayer.h>

using namespace nvinfer1;

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
	ConfigManager &config = ConfigManager::getInstance();
    float r_w = config.width / (img.cols * 1.0);
    float r_h = config.height / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (config.height - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (config.height - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (config.width - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (config.width - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps, DataType dataType) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ dataType, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{dataType, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0f;
    }
    Weights power{dataType, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
	scale_1->setName((lname + ".bn").c_str());
    return scale_1;
}

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, DataType dataType) {
    Weights emptywts{ dataType, nullptr, 0 };
    int p = ksize / 2;
    //IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ s, s });
    conv1->setPadding(DimsHW{ p, p });
    conv1->setNbGroups(g);
	conv1->setName((lname + ".conv").c_str());

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 0.001);

    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
	sig->setName((lname + ".sigmoid").c_str());
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
	ew->setName((lname + ".silu").c_str());
    return ew;
}

ILayer* focus(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname, DataType dataType) {

	/*
    ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
	*/
	Weights onewts{dataType, nullptr, inch * 4};
	float* vOnes = new float[inch * 4];
	for(int i = 0; i < inch * 4; ++i)
	{
		if(i % 4 == 0)
			vOnes[i] = 1.f;
		else
			vOnes[i] = 0.f;

	}
	onewts.values = vOnes;
    Weights emptywts{dataType, nullptr, 0 };

	IConvolutionLayer *s1 = network->addConvolution(input, inch, DimsHW{2, 2}, onewts, emptywts);
    assert(s1);
    s1->setStride(DimsHW{2, 2});
    // conv1->setPadding(DimsHW{ p, p });
    s1->setNbGroups(inch);
	s1->setName((lname + ".focus.slice1").c_str());


	float* s2_weight_data = new float[inch * 4];
	for(int i = 0; i < inch * 4; ++i)
	{
		if(i % 4 == 2)
			s2_weight_data[i] = 1.f;
		else
			s2_weight_data[i] = 0.f;
	}
	Weights s2_k_wts{dataType, nullptr, inch * 4};
	s2_k_wts.values = s2_weight_data;
	IConvolutionLayer *s2 = network->addConvolution(input, inch, DimsHW{2, 2}, s2_k_wts, emptywts);
    assert(s3);
    s2->setStride(DimsHW{ 2, 2});
    s2->setNbGroups(inch);
	s2->setName((lname + ".focus.slice2").c_str());

	float* s3_weight_data = new float[inch * 4];
	for(int i = 0; i < inch * 4; ++i)
	{
		if(i % 4 == 1)
			s3_weight_data[i] = 1.f;
		else
			s3_weight_data[i] = 0.f;
	}
	Weights s3_k_wts{dataType, nullptr, inch * 4};
	s3_k_wts.values = s3_weight_data;

	IConvolutionLayer *s3 = network->addConvolution(input, inch, DimsHW{2, 2}, s3_k_wts, emptywts);
    assert(s3);
    s3->setStride(DimsHW{ 2, 2});
    s3->setNbGroups(inch);
	s3->setName((lname + ".focus.slice3").c_str());

	float* s4_weight_data = new float[inch * 4];
	for(int i = 0; i < inch * 4; ++i)
	{
		if(i % 4 == 3)
			s4_weight_data[i] = 1.f;
		else
			s4_weight_data[i] = 0.f;
	}
	Weights s4_k_wts{dataType, nullptr, inch * 4};
	s4_k_wts.values = s4_weight_data;
	IConvolutionLayer *s4 = network->addConvolution(input, inch, DimsHW{2, 2}, s4_k_wts, emptywts);
    assert(s4);
    s4->setStride(DimsHW{ 2, 2});
    s4->setNbGroups(inch);
	s4->setName((lname + ".focus.slice4").c_str());

    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
	int nb_input = 0; 
	for (auto it = std::begin(inputTensors); it!=std::end(inputTensors); ++it){
		nb_input += (*it)->getDimensions().d[0];
		assert(it);
	}
    auto cat = network->addConcatenation(inputTensors, 4);
	assert(cat);
	cat->setName((lname + ".focus.concat").c_str());

    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv", dataType);
    return conv;
}

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname, DataType dataType) {
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1", dataType);
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2", dataType);
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
		assert(ew);
		ew->setName((lname + ".add").c_str());
        return ew;
    }
    return cv2;
}

ILayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname, DataType dataType) {
    Weights emptywts{dataType, nullptr, 0 };
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1", dataType);
    auto cv2 = network->addConvolution(input, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv2.weight"], emptywts);
	cv2->setName((lname + ".cv2").c_str());

    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i), dataType);
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolution(*y1, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv3.weight"], emptywts);
	cv3->setName((lname + "cv3").c_str());

    ITensor* inputTensors[] = { cv3->getOutput(0), cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);
	cat->setName((lname + ".cv.concat").c_str());

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 0.001, dataType);

	// LEAKY_RELU
    // auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    // lr->setAlpha(0.1);
	std::vector<float> vOnes(1, 1.f);
	Weights onewts{dataType, const_cast<const float*>(vOnes.data()), 1};
	std::vector<float> alpha(1, 0.1f);
	Weights alphawts{dataType, const_cast<const float*>(alpha.data()), 1};

	IScaleLayer* scale = network->addScale(*bn->getOutput(0), ScaleMode::kUNIFORM, emptywts, alphawts, onewts);
	scale->setName((lname + ".leaky.scale").c_str());
	IElementWiseLayer* lr = network->addElementWise(*bn->getOutput(0), *scale->getOutput(0), ElementWiseOperation::kMAX);
	lr->setName((lname + ".leaky.max").c_str());

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4", dataType);
    return cv4;
}

ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname, DataType dataType) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1", dataType);
    //auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2", dataType);

	/* cv2 start */
    Weights emptywts{ dataType, nullptr, 0 };
	constexpr int ksize = 1;
	constexpr int s = 1;
	constexpr int g_ = 1;
	int outch = c_;
    int p = ksize / 2;
    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".cv2.conv.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ s, s });
    conv1->setPadding(DimsHW{ p, p });
    conv1->setNbGroups(g_);
	conv1->setName((lname + ".cv2.conv").c_str());

#if 0
	conv1->getOutput(0)->setName((lname + ".cv2.conv.output").c_str());
	network->markOutput(*conv1->getOutput(0));
#endif

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".cv2.bn", 0.001);

    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
	sig->setName((lname + ".cv2.sigmoid").c_str());
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
	ew->setName((lname + ".cv2.silu").c_str());
	/* cv2 end */

    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i), dataType);
        y1 = b->getOutput(0);
    }

    ITensor* inputTensors[] = { y1, ew->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);
	cat->setName((lname + ".cv.concat").c_str());

    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3", dataType);
    return cv3;
}

ILayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname, DataType dataType) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1", dataType);

    auto pool1 = network->addPooling(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k1, k1 });
    pool1->setPadding(DimsHW{ k1 / 2, k1 / 2 });
    pool1->setStride(DimsHW{ 1, 1 });
	pool1->setName((lname + ".pool1").c_str());

    auto pool2 = network->addPooling(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k2, k2 });
    pool2->setPadding(DimsHW{ k2 / 2, k2 / 2 });
    pool2->setStride(DimsHW{ 1, 1 });
	pool2->setName((lname + ".pool2").c_str());

    auto pool3 = network->addPooling(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k3, k3 });
    pool3->setPadding(DimsHW{ k3 / 2, k3 / 2 });
    pool3->setStride(DimsHW{ 1, 1 });
	pool3->setName((lname + ".pool3").c_str());

    ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
	cat->setName((lname + ".pool.concat").c_str());

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2", dataType);
    return cv2;
}

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = ANCHOR_NUM * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto *p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

ILayer* toTensor(INetworkDefinition *network, ITensor& input, DataType dataType) {
    auto shuffle_layer = network->addShuffle(input);
    assert(shuffle_layer != nullptr);

    const auto &input_dim = input.getDimensions();
    shuffle_layer->setReshapeDimensions(nvinfer1::Dims3{input_dim.d[2], input_dim.d[0], input_dim.d[1]});
    shuffle_layer->setFirstTranspose(nvinfer1::Permutation{2, 0, 1});

    std::vector<float> vecScale, vecShift, vecPower;
    for (int i = 0; i < input_dim.d[2]; ++i){
	vecScale.push_back(1 / 255.f);
	vecShift.push_back(0.f);
	vecPower.push_back(1.f);
    }
    const nvinfer1::Weights power{dataType, 
            vecPower.data(), 
            static_cast<int64_t>(vecPower.size())};

    const nvinfer1::Weights scale{dataType, vecScale.data(), 
	                          static_cast<int64_t>(vecScale.size())};

    const nvinfer1::Weights shift{dataType, 
            vecShift.data(), 
            static_cast<int64_t>(vecShift.size())};

    auto norm = network->addScale(*shuffle_layer->getOutput(0), 
                nvinfer1::ScaleMode::kCHANNEL, 
                shift, 
                scale, 
                power);
    return norm;
}

IPluginLayer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets) {
	ConfigManager &config = ConfigManager::getInstance();
    auto anchors = getAnchors(weightMap, lname);
    int scale = 8;
    std::vector<YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        YoloKernel kernel;
        kernel.width = config.width / scale;
        kernel.height = config.height / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }

    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
	YoloLayerPlugin *yolo_plugin = new YoloLayerPlugin(config.className.size(), config.width, config.height, config.maxOutputBBoxCount, kernels);
    auto yolo = network->addPluginExt(&input_tensors[0], input_tensors.size(), *yolo_plugin);
    return yolo;
}

