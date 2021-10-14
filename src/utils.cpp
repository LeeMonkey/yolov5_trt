#include <utils.h>
#include <math.h>
#include <common.h>
#include <sys/stat.h>
#include <memory>
#include <sstream>

#if 0
cv::Mat preprocess_img(const cv::Mat& img, int input_w, int input_h, float& ratio,
    float& pad_w, float& pad_h) {
    int w, h;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        pad_w = 0;
        pad_h = (input_h - h) / 2;
        ratio = r_w;
    } else {
        w = r_h * img.cols;
        h = input_h;
        pad_w = (input_w - w) / 2;
        pad_h = 0;
        ratio = r_h;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(pad_w, pad_h, re.cols, re.rows)));
    out.convertTo(out, CV_32FC3, 1.f);
    return out;
}
#endif

cv::Mat preprocess_img(const cv::Mat& img, int input_w, int input_h, float& ratio,
    float& pad_w, float& pad_h) {
    ratio = std::min(input_w * 1.0 / img.cols, input_h * 1.0 / img.rows);

    int new_w = static_cast<int>(std::round(img.cols * ratio));
    int new_h = static_cast<int>(std::round(img.rows * ratio));

    float dw = (input_w - new_w) / 2.0;
    float dh = (input_h - new_h) / 2.0;

    cv::Mat new_image;
    img.copyTo(new_image);

    if(new_w != input_w or new_h != input_h)
        cv::resize(new_image, new_image, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int top = static_cast<int>(std::round(dh - 0.1));
    int bottom = static_cast<int>(std::round(dh + 0.1));
    int left = static_cast<int>(std::round(dw - 0.1));
    int right = static_cast<int>(std::round(dw + 0.1));
    pad_w = left;
    pad_h = top;

    cv::copyMakeBorder(new_image, 
                       new_image, 
                       top, 
                       bottom,
                       left,
                       right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
#ifdef IS_NORM_IN_TENSORRT
    new_image.convertTo(new_image, CV_32FC3, 1.);
#else
    new_image.convertTo(new_image, CV_32FC3, 1 / 255., 0);
#endif
    return new_image;
}


std::vector<float> prepareImage(cv::Mat& image)
{
    ConfigManager& config = ConfigManager::getInstance();

    int channel = config.channel;
    int width = config.width;
    int height = config.height;

    float ratio = 1.f;
    float pad_w = 0.f;
    float pad_h = 0.f;

    cv::Mat outputImage = preprocess_img(image, width, height, ratio, pad_w, pad_h);
    std::vector<float> result(height * width * channel); 
    memcpy(result.data(), outputImage.data, width * height * channel * sizeof(float));

    return result;
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

void get_image_target_in_dir(const std::string &image_dir_name, std::vector<std::pair<std::string, std::string> > &vecPath)
{ 
    DIR *dir=opendir(image_dir_name.c_str());
    if(!dir)
        return;

    dirent *ent = nullptr;
    while((ent=readdir(dir)) != nullptr)
    {
            if(strcmp(ent->d_name,".")==0||strcmp(ent->d_name,"..")==0)
            {
                continue;
            }
            std::string name{image_dir_name};
            if(name.at(name.size() - 1) != '/')
                name.append("/");
            name.append(ent->d_name);
            struct stat st;
            stat(name.c_str(), &st);
            if(S_ISDIR(st.st_mode))
                get_image_target_in_dir(name, vecPath);
            else
            {
                std::size_t idx = name.find("images");
                if(idx != std::string::npos)
                {
                    std::string lname{name};
                    lname.replace(idx, 6, "labels");
                    std::size_t ext_idx = lname.find_last_of(".");
                    if(ext_idx != std::string::npos)
                    {
                        lname.replace(lname.begin() + ext_idx + 1, lname.end(), "txt");
                        if(isFileExists_access(lname.c_str()))
                        {
                            if(S_ISLNK(st.st_mode))
                            {
                                char buf[1024];
                                int len = -1;
                                if((len = readlink(name.c_str(), buf, 1024 - 1)) != -1)
                                {
                                    buf[len] = '\0';
                                    name.assign(buf);
                                    std::cout << buf << std::endl;
                                    vecPath.emplace_back(name, lname);
                                }
                            }
                            else
                                vecPath.emplace_back(name, lname);
                        }
                    }
                }
            }
    }       
    closedir(dir);
}

void drawImg(const std::vector<Detection> &result, cv::Mat& img)
{
    int default_font_size = std::max(static_cast<int>(std::sqrt(img.rows * img.cols) / 90), 10);

    cv::Point text_pos;
    std::string label;
    std::stringstream stream;
    int x_off = 0;
    int y_off = 0;
    //int base_line;
    for (size_t i = 0; i < result.size(); ++i) {
        auto item = result.at(i);

        float height_ratio = (item.bbox[3] - item.bbox[1]) 
            / std::sqrt(img.rows * img.cols);

        float font_size = std::min(
            std::max(0.8f, (height_ratio - 0.02f) / 0.08f + 1.f), 
            1.6f) * .05f * default_font_size;

        stream << std::fixed << std::setprecision(2) 
            << item.conf * 100 <<"%";

        if(i == 0){
            x_off = (item.bbox[2] + item.bbox[0] - img.cols) / 2;
            y_off = (item.bbox[3] + item.bbox[1] - img.rows) / 2;
            stream << ": (" << x_off << ", " << y_off << ")"; 
        }

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(stream.str(), 
            cv::FONT_HERSHEY_COMPLEX,
            font_size,
            1,
            &baseline);

        int line_width = std::max(font_size / 4.f, 2.f);

        text_pos.x = item.bbox[0];
        text_pos.y = item.bbox[1] - text_size.height / 2;

        cv::rectangle(img,
            cv::Point(item.bbox[0], item.bbox[1]),
            cv::Point(item.bbox[2], item.bbox[3]),
            i == 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 
            line_width, 
            8, 
            0);

        cv::rectangle(img,
            text_pos + cv::Point(-line_width, baseline),
            text_pos + cv::Point(text_size.width - line_width, -(text_size.height + line_width)),
            cv::Scalar(0, 255, 127), 
            -1, 
            8, 
            0);

        cv::putText(img,
            stream.str(),
            text_pos,
            cv::FONT_HERSHEY_SIMPLEX,
            font_size,
            cv::Scalar(255, 255, 255),
            line_width,
            8,
            0);

        stream.clear();
        stream.str("");
    }

    cv::Point start, end, lstart, lend;
    cv::Scalar line_color(0, 255, 255);
    int thickness = 2;
    int lineType = 8; 
    int len = 10;
    int alpha = 30;
    if(x_off > 0){
        start.x = img.cols / 2 + 10;
        start.y = img.rows / 2;
        end.x = img.cols / 2 + 30;
        end.y = img.rows / 2;
    }else if(x_off < 0){
        start.x = img.cols / 2 - 10;
        start.y = img.rows / 2;
        end.x = img.cols / 2 - 30;
        end.y = img.rows / 2;
    }

    if(x_off != 0){
        lstart = cv::Point(img.cols, img.rows) - end;
        lend = cv::Point(img.cols, img.rows) - start;
        cv::line(img, 
            lstart, 
            lend, 
            line_color, 
            thickness, 
            lineType);

        drawArrow(img, 
            start,
            end,
            len,
            alpha,
            line_color, 
            thickness,
            lineType);
    }

    if(y_off > 0){
        start.x = img.cols / 2;
        start.y = img.rows / 2 + 10;
        end.x = img.cols / 2;
        end.y = img.rows / 2 + 30;
    }else if(y_off < 0){
        start.x = img.cols / 2;
        start.y = img.rows / 2 - 10;
        end.x = img.cols / 2;
        end.y = img.rows / 2 - 30;
    }

    if(y_off != 0){
        lstart = cv::Point(img.cols, img.rows) - end;
        lend = cv::Point(img.cols, img.rows) - start;
        cv::line(img, 
            lstart, 
            lend, 
            line_color, 
            thickness,
            lineType);

        drawArrow(img, 
            start,
            end,
            len,
            alpha,
            line_color, 
            thickness,
            lineType);
    }
}

void drawArrow(cv::Mat& img, 
    cv::Point pStart, 
    cv::Point pEnd, 
    int len, 
    int alpha,
    cv::Scalar color, 
    int thickness, 
    int lineType){    

     const double PI = 3.1415926;    
     cv::Point arrow;    
     //计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面）   
     double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));  
     cv::line(img, pStart, pEnd, color, thickness, lineType);   

     //计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置） 
     arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);     
     arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);  

     cv::line(img, pEnd, arrow, color, thickness, lineType);   

     arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);     
     arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);    
     cv::line(img, pEnd, arrow, color, thickness, lineType);
 }

void getScaleAndPad(const int width, const int height, float& ratio,
    float& pad_w, float& pad_h) {
    ConfigManager &config = ConfigManager::getInstance();

    int input_w = config.width;
    int input_h = config.height;
    ratio = std::min(input_w * 1.0 / width, input_h * 1.0 / height);

    int new_w = static_cast<int>(std::round(width * ratio));
    int new_h = static_cast<int>(std::round(height * ratio));

    float dw = (input_w - new_w) / 2.0;
    float dh = (input_h - new_h) / 2.0;

    int top = static_cast<int>(std::round(dh - 0.1));
    int left = static_cast<int>(std::round(dw - 0.1));
    pad_w = left;
    pad_h = top;
}

void postProcess(std::vector<Detection> &boxes, const int width, const int height){
    ConfigManager &config = ConfigManager::getInstance();

    nms(boxes, config.confThresh, config.nmsThresh);

    float ratio = 1.f;
    float pad_w = 0.f;
    float pad_h = 0.f;

    getScaleAndPad(width, height, ratio, pad_w, pad_h);

    for(auto& box: boxes)
    {
        auto it = std::begin(box.bbox);
        auto start = std::begin(box.bbox);
        while(it != std::end(box.bbox)){
            size_t idx = it - start;
            if(idx == 0)
                *it -= pad_w;
            else if(idx == 1)
                *it -= pad_h;
    
            *it /= ratio;
            ++it;
        }
    }

    // show specify class id
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [&config](Detection &box){
                               return config.showClassIds.end() == std::find(config.showClassIds.begin(), 
                                      config.showClassIds.end(), static_cast<int>(box.class_id)); }),
                boxes.end());

    // if or not return multiboxes
    if (!config.returnMultiBox)
    {
        std::sort(boxes.begin(), boxes.end(), [&width, &height](Detection &box1, Detection &box2){
            return (std::pow(width / 2.f - box1.bbox[0], 2) + std::pow(height / 2.f - box1.bbox[1], 2)) 
                < (std::pow(width / 2.f - box2.bbox[0], 2) + std::pow(height / 2.f - box2.bbox[1], 2));
        });
        boxes.erase(boxes.begin() + 1, boxes.end());
    }
}

void nms(std::vector<Detection>& output, float conf_thresh, float nms_thresh) {
    ConfigManager &config = ConfigManager::getInstance();

    std::map<int, std::vector<Detection>> m;

    for(size_t i = 0; i < output.size() && i < config.maxOutputBBoxCount; ++i)
    {
        if(output.at(i).conf <= conf_thresh)
            continue;

        int class_id = static_cast<int>(output.at(i).class_id);
        if(config.isGlobalNMS)
            class_id = 0;
        if(m.count(class_id) == 0)
            m.emplace(class_id, std::vector<Detection>());

        m[class_id].push_back(output.at(i));
    }
    
    std::vector<Detection>().swap(output);

    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), 
                  [](const Detection &a, const Detection &b)
                  {
                      return a.conf > b.conf;
                  });

        std::vector<bool> vecFlag(dets.size(), false);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            if(!vecFlag.at(m)){
                output.push_back(item);
                vecFlag.at(m) = true;
            }
            else
                continue;
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if(vecFlag.at(n))
                    continue;
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    vecFlag.at(n) = true;
                }
            }
        }
    }
}

bool cmp(const Detection& a, const Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh) {
    ConfigManager &config = ConfigManager::getInstance();

    int det_size = sizeof(Detection) / sizeof(float);
    std::map<int, std::vector<Detection>> m;
    for (size_t i = 0; i < output[0] && i < config.maxOutputBBoxCount; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));

        int class_id = static_cast<int>(det.class_id);
        if(config.isGlobalNMS) {
            class_id = 0;
        } 

        if (m.count(class_id) == 0) 
            m.emplace(class_id, std::vector<Detection>());
        m[class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        std::vector<bool> vecFlag(dets.size(), false);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            if(!vecFlag.at(m)){
                res.push_back(item);
                vecFlag.at(m) = true;
            }
            else
                continue;
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if(vecFlag.at(n))
                    continue;
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    vecFlag.at(n) = true;
                }
            }
        }
    }
}

void load_targets(const std::string &target_file, const cv::Size& size, std::vector<Detection>& targets)
{
    assert(isFileExists_access(target_file));
    assert(targets.size() == 0);
    std::ifstream ifs(target_file, std::ios::in);
    std::string line;
    while(getline(ifs, line))
    {
        std::istringstream label(line);
        float classes = -1.f;
        float cx = 0.f, cy = 0.f, w = 0.f, h = 0.f; 
        std::string value;
        label >> value; 
        classes = std::stof(value);
        label >> value;
        cx = std::stof(value);
        label >> value;
        cy = std::stof(value);
        label >> value;
        w = std::stof(value);
        label >> value;
        h = std::stof(value);
        if((classes + 1) > 1e-5)
        {
            Detection det{cx * size.width, cy * size.height, w * size.width, h * size.height, 0.f, classes};
            targets.push_back(det);
        }
    }
}
