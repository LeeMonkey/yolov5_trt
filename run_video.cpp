/*************************************************************************
    > File Name: main.cpp
    > Author: Lee
    > Created Time: 2021年08月12日 星期四 10时38分55秒
 ************************************************************************/
#include <iostream>
#include <string>
#include <iomanip>
#include <chrono>
#include <ctdetNet.h>
#include <config.h>


bool parse_args(int argc, char** argv, std::string& original_video_path, std::string& save_video_path) {
    if (argc < 3) return false;

    original_video_path = std::string(argv[1]);
    save_video_path = std::string(argv[2]);
    return true;
}

int main(int argc, char** argv)
{
    std::string original_video_path{""};
    std::string save_video_path{""};
    if(!parse_args(argc, argv, original_video_path, save_video_path))
    {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./bin/runvideo (original_video_path) (save_video_path)" << std::endl;
        return -1;
    }

    ConfigManager& config = ConfigManager::getInstance();

    std::string& engineName = config.enginePath;

    ctdet::ctdetNet detector(engineName);
    if(!isFileExists_access(original_video_path)) {
        std::cerr << "No such original_video_path: `" 
                  << original_video_path 
                  << "`" << std::endl;
        return -1;
    }

    cv::VideoCapture cap;
    cap.open(original_video_path);

    double fps = cap.get(cv::CAP_PROP_FPS);
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer(save_video_path,
                           cv::VideoWriter::fourcc('m', 'p', '4', '2'),
                           fps,
                           cv::Size(width, height),
                           true);
	size_t num_interations = 0;
    while(cap.isOpened())
    {
        cv::Mat image;
        bool ret = cap.read(image);
        if(!ret)
            break;
		++num_interations;

        std::vector<Detection> boxes;
        cudaDeviceSynchronize();
        auto start = std::chrono::system_clock::now();
		//image.setTo(cv::Scalar(1.0f, 1.0f, 1.0f));
        detector.forward(image, boxes);
        cudaDeviceSynchronize();
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << "ms" << std::endl;
        for(const auto& box: boxes)
        {
            float xmin = box.bbox[0] - box.bbox[2] / 2.f;
            float ymin = box.bbox[1] - box.bbox[3] / 2.f;
            float xmax = xmin + box.bbox[2];
            float ymax = ymin + box.bbox[3];
            if(std::find(config.showClassIds.begin(),
                         config.showClassIds.end(),
                         static_cast<int>(box.class_id))
               == config.showClassIds.end())
                continue;

            cv::rectangle(image,
                          cv::Point(xmin, ymin),
                          cv::Point(xmax, ymax),
                          box.class_id == 0 ? cv::Scalar(0, 50, 250): cv::Scalar(50, 250, 0));

            cv::putText(image, 
                        std::to_string(box.conf),
                        cv::Point(xmin, ymin - 1),
                        cv::FONT_HERSHEY_PLAIN,
                        1,
                        cv::Scalar(250, 250, 250));
        }
        writer.write(image);

        std::cout << std::string(60, '#') << std::endl;
    }
	detector.printLayerTimes(num_interations);
    cap.release();
}

