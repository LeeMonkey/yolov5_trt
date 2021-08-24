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

bool parse_args(int argc, char** argv, std::string& wtsName, std::string& engineName, std::string& image_list) {
    if (argc < 2) return false;
    if (std::string(argv[1]) == "-s") {
        for(int i = 2; i < argc; ++i)
        {
            if(i == 2)
                wtsName = std::string(argv[2]);
            else if(i == 3)
                engineName = std::string(argv[3]);
        }
    } else if (std::string(argv[1]) == "-d" && argc == 3) {
        image_list = std::string(argv[2]);
    } else
        return false;
    return true;
}

int main(int argc, char** argv)
{
    std::string wtsName{""};
    std::string engineName{""};
    std::string image_list_path;
    if(!parse_args(argc, argv, wtsName, engineName, image_list_path))
    {
        std::cerr << "Arguments not right! \n\t[]: optional (): " 
                  << "required. \n\t*Notice: optional item must specify in config/perproties.json" << std::endl;
        std::cerr << "./bin/rundet -s [.wts] [.engine]" << std::endl;
        std::cerr << "./bin/rundet -d (image_list_path)" << std::endl;
        return -1;
    }

    ConfigManager& config = ConfigManager::getInstance();

    if(image_list_path.empty()) {
        if(wtsName.empty() && wtsName.size() == 0)
            wtsName = config.weightsPath;
        if(engineName.empty() && engineName.size() == 0)
            engineName = config.enginePath;
		ctdet::ctdetNet detector(engineName, wtsName);
    } 
    else { 
        if(engineName.empty() && engineName.size() == 0)
            engineName = config.enginePath;

		ctdet::ctdetNet detector(engineName);
        if(!isFileExists_access(image_list_path)) {
            std::cerr << "No such image list: `" 
                      << image_list_path 
                      << image_list_path 
                      << "`" << std::endl;
            return -1;
        }
        for(int i = 0; i < 1; ++i){
            std::ifstream ifs(image_list_path, std::ios::out);
            std::string image_path;

            while(getline(ifs, image_path))
            {
                cv::Mat image = cv::imread(image_path);
                std::vector<Detection> boxes;
                cudaDeviceSynchronize();
                auto start = std::chrono::system_clock::now();
                detector.forward(image, boxes);
                cudaDeviceSynchronize();
                auto end = std::chrono::system_clock::now();
                std::cout << "inference time: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                    << "ms" << std::endl;
                for(const auto& box: boxes)
                {
#if 0
                    std::cout << std::setprecision(2) 
                              << "conf: " << box.conf
                              << " class_id" << box.class_id
                              << " bbox: ["
                              << box.bbox[0] << ", "
                              << box.bbox[1] << ", "
                              << box.bbox[2] << ", "
                              << box.bbox[3] << "]"
                              << std::endl;
#endif
                    float xmin = box.bbox[0] - box.bbox[2] / 2.f;
                    float ymin = box.bbox[1] - box.bbox[3] / 2.f;
                    float xmax = xmin + box.bbox[2];
                    float ymax = ymin + box.bbox[3];
                    cv::rectangle(image,
                                  cv::Point(xmin, ymin),
                                  cv::Point(xmax, ymax),
                                  box.class_id == 0 ? cv::Scalar(0, 50, 250): cv::Scalar(50, 250, 0));

                    cv::putText(image, 
                                config.className[static_cast<int>(box.class_id)],
                                cv::Point(xmin, ymin - 1),
                                cv::FONT_HERSHEY_PLAIN,
                                1,
                                cv::Scalar(250, 250, 250));
                }
                size_t nIndex = image_path.rfind("/");
                std::string image_name {image_path};
                if (nIndex != std::string::npos)
                    image_name = image_name.substr(nIndex + 1);
                cv::imwrite("results/" + image_name, image);
                std::cout << std::string(60, '#') << std::endl;
            }
        }
    }
}

