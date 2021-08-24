#include <utility>
#include <utils.h>
#include <config.h>
#include <ctdetApi.h>

namespace ctdet
{
	ctdetDetector::ctdetDetector(){
		ConfigManager &config = ConfigManager::getInstance();
		this->net.reset(new ctdetNet(config.enginePath));
	}

    void ctdetDetector::detect(cv::Mat& input, cv::Mat &output){
		std::vector<Detection> boxes;
        this->net->forward(input, boxes);

        input.copyTo(output);
        drawImg(boxes, output);
    }  
}
