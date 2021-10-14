#ifndef _YOLO_UTILS_H_
#define _YOLO_UTILS_H_

#include <opencv2/opencv.hpp>
#include <string>
#include <unistd.h>
#include <dirent.h>
#include <yololayer.h>

cv::Mat preprocess_img(const cv::Mat& img, int input_w, int input_h, float& scale, float& pad_w, float& pad_h);

std::vector<float> prepareImage(cv::Mat& image);

inline bool isFileExists_access(const std::string& name) {
    return access(name.c_str(), F_OK ) != -1;
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);

void get_image_target_in_dir(const std::string &image_dir_name, std::vector<std::pair<std::string, std::string> > &vecPath);

void drawImg(const std::vector<Detection> & result, cv::Mat& img);

void drawArrow(cv::Mat& img, 
	cv::Point pStart, 
	cv::Point pEnd, 
	int len, 
	int alpha,
	cv::Scalar color, 
	int thickness, 
	int lineType);    

void getScaleAndPad(const int width, const int height, float& ratio, float& pad_w, float& pad_h);

void postProcess(std::vector<Detection> &boxes, const int Width, const int Height);

void nms(std::vector<Detection>& output, float conf_thresh, float nms_thresh=0.5);

inline bool cmp(const Detection& a, const Detection& b);

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5); 

void load_targets(const std::string &target_file, const cv::Size& size, std::vector<Detection> &targets);

#endif  // _YOLO_UTILS_H_
