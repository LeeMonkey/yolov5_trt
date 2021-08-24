//
// Created by cao on 19-10-26.
//

#ifndef CTDET_TRT_CTDETAPI_H
#define CTDET_TRT_CTDETAPI_H

#include <opencv2/opencv.hpp>
#include <ctdetNet.h>

namespace ctdet
{
    class ctdetDetector
    {
        public:
            ctdetDetector();

            void detect(cv::Mat& input, cv::Mat &output);

        private:
            std::unique_ptr<ctdetNet> net;
    };
}


#endif //CTDET_TRT_CTDETNET_H
