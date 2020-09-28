#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <map>

void OnMouse(int event, int x, int y, int flags, void* param);

class LaneLabeler {
    friend void OnMouse(int event, int x, int y, int flags, void* param);
    public:
        int label(const std::string& winname,
                const cv::Mat& img,
                std::map<int, std::vector<cv::Point2d>>& lane_points);

        cv::Scalar getLaneColor(int lane_id);

    private:
        void process();
        

    private:
        int _edit_status = 1;
        int _edit_id = 0;
        cv::Mat _image_input;
        cv::Mat _image_output;
        int _mouse_x;
        int _mouse_y;

        std::map<int, std::vector<cv::Point2d>> *_ptr_lane_points;
};