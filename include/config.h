//
// Created by cao on 19-10-26.
//

#ifndef _YOLO_CONFIG_H
#define _YOLO_CONFIG_H
#include <stdio.h>
#include <string>
#include <vector>

#define __output_info(...) \
        fprintf(stdout, __VA_ARGS__);

#define __output_error(...) \
        fprintf(stderr, __VA_ARGS__);
     
#define __format(__fmt__) "%s:%d - <%s>: " __fmt__ "\n"
     
#define LOGINFO(__fmt__, ...) \
        __output_info(__format(__fmt__), __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);

#define LOGERROR(__fmt__, ...) \
    __output_error(__format(__fmt__), __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);

const std::string ENGINE_PATH="config/properties.json";

class ConfigManager
{
    public:
        ConfigManager(ConfigManager &&) = delete;

        ConfigManager(const ConfigManager &) = delete;

        static ConfigManager& getInstance()
        {
            static ConfigManager instance;
            return instance;
        }

    private:
        void init(const std::string& config_path);
        ConfigManager();

    public:
        int width;
        int height;
        int channel;

		// anchor
		int numAnchor;
		size_t maxOutputBBoxCount;
		float depthMultiple;
		float widthMultiple;

		// thresh 
        float nmsThresh;
        float confThresh;
		float ignoreThresh;

		// control count of boxes
        bool returnMultiBox;
		bool isGlobalNMS;

        std::string enginePath;
		std::string weightsPath;
		std::string inputBlobName;
		std::string outputBlobName;

		//normalize
        std::vector<std::string> className;

		std::vector<int> showClassIds;
};
#endif
