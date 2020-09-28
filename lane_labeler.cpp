#include "highway_check/lane_labeler.h"
#include "opencv2/highgui.hpp"

void OnMouse(int event, int x, int y, int flags, void* param) {
    LaneLabeler *labeler = (LaneLabeler*)param;

    if(event == CV_EVENT_LBUTTONDOWN && labeler->_edit_status == 1) {
        auto &lane_points = *(labeler->_ptr_lane_points);
        lane_points[labeler->_edit_id].push_back(cv::Point2d(x, y));
    }

    if(event == CV_EVENT_MOUSEMOVE) {
        labeler->_mouse_x = x;
        labeler->_mouse_y = y;
    }
}

int LaneLabeler::label(const std::string& winname,
                    const cv::Mat& img,
                    std::map<int, std::vector<cv::Point2d>>& lane_points) {
    lane_points.clear();
    _edit_id = 1;
    _ptr_lane_points = &lane_points;

    cv::destroyAllWindows();
    cv::namedWindow(winname, CV_WINDOW_NORMAL);
    cv::resizeWindow(winname, cv::Size(2 * img.cols, 2 * img.rows));
    cv::imshow(winname, img);

    img.copyTo(_image_input);
    img.copyTo(_image_output);
    cv::setMouseCallback(winname, OnMouse, this);

    cv::putText(_image_output, winname + std::string(" lane labelling: edit mode"),
        cv::Point(0, 100), cv::FONT_HERSHEY_PLAIN, 5, cv::Scalar(0, 255, 0), 3);
    cv::putText(_image_input, winname + std::string(" lane labeling: browser mode"),
        cv::Point(0, 100), cv::FONT_HERSHEY_PLAIN, 5, cv::Scalar(0, 255, 0), 3);

    while(true) {
        process();
        if(_edit_status) {
            
            cv::imshow(winname, _image_output);
        } else {
            cv::imshow(winname, _image_input);
        }
        char key = cv::waitKey(30);
        switch(key) {
            case 32:    // space: switch edit/browse mode
                if(_edit_status == 1) {
                    _edit_status = 0;
                } else {
                    _edit_status = 1;
                }
                break;

            case 27:    // Backspace: clear all lanes
                (*_ptr_lane_points)[_edit_status].clear();
                break;

            case '1':    // key 1: left-left lane
                _edit_id = 0;
                break;
            
            case '2':    // key 2: left lane
                _edit_id = 1;
                break;

            case '3':    // key 3: right lane
                _edit_id = 2;
                break;

            case '4':    // key 4: right right lane
                _edit_id = 3;
                break;

            default:
                break;
        }

        if(key == 13) {

            break;
        }
    }
    cv::destroyWindow(winname);

    for(auto &lane: lane_points) {
        std::vector<cv::Point2d>& lane_points_sparse = lane.second;
        std::vector<cv::Point2d> lane_points_dense;
        for(size_t i=1; i<lane_points_sparse.size(); i++) {
            cv::Point2d p1 = lane_points_sparse[i];
            cv::Point2d p2 = lane_points_sparse[i - 1];
            float len = hypot(p1.x - p2.x, p1.y - p2.y);
            for(float j=0; j<len; j+=1) {
                lane_points_dense.push_back(p1 + (p2 - p1) * (j / len));
            }
        }
        lane_points_sparse = lane_points_dense;
    }
    
    return 0;
}

cv::Scalar LaneLabeler::getLaneColor(int lane_id) {
    cv::Scalar color;
    switch(lane_id) {
        case 0:
            color = cv::Scalar(255, 0, 0);   // left left lane: blue
            break;
        case 1:
            color = cv::Scalar(0, 255, 0);   // left lane: green
            break;
        case 2:
            color = cv::Scalar(0, 0, 255);   // right lane: green
            break;
        case 3:
            color = cv::Scalar(255, 255, 0); // right right lane:
            break;
        
        default:
            color = cv::Scalar(0, 255, 255); // others
            break;
    }

    return color;
}

void LaneLabeler::process() {
    for(auto &lane: *_ptr_lane_points) {
        int lane_id = lane.first;
        std::vector<cv::Point2d>& lane_points = lane.second;
        for(size_t j=0; j<lane_points.size(); j++) {
            cv::circle(_image_output, cv::Point(lane_points[j].x, lane_points[j].y),
                        1, getLaneColor(lane_id));
            if(j != 0) {
                cv::line(_image_output,
                        cv::Point(lane_points[j-1].x, lane_points[j-1].y),
                        cv::Point(lane_points[j].x, lane_points[j].y),
                        getLaneColor(lane_id), 1);
            }
        }
    }
}