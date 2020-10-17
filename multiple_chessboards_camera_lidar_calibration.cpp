// Tool to generate camera to lidar transformation based on mutiple chessboards
// Geiger, Andreas, et al. "Automatic camera and range sensor calibration using a single shot." 2012
// IEEE International Conference on Robotics and Automation. IEEE, 2012.
// http://www.cvlibs.net/publications/Geiger2012ICRA.pdf
#include <common/sensor_calib_db.h>
#include <cv_bridge/cv_bridge.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <image_transport/image_transport.h>
#include <message_filters/simple_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <boost/algorithm/string/split.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include "base/common_config.h"
#include "common/sensor_calib_db.h"
#include "base/program.h"
#include "pcl/pcl_types.h"

using namespace drive::common::base;
using namespace drive::perception::sensor;

DEFINE_string(bag_file, "",
              "input single ROS bag file, if empty will try to load image/lidar file directly");
DEFINE_string(calib_dir, "/opt/plusai/var/calib_db", "Calibrations directory");
DEFINE_string(car, "mkz", "car name");
DEFINE_string(camera_calib_name, "/front_left_camera", "camera name for query camera_calib");
DEFINE_string(camera_calib_date, "19700101", "camera date for query camera_calib");
DEFINE_string(lidar_topic, "/upuck1/velodyne_points", "Lidar ROS topic");
DEFINE_string(camera_topic, "/front_left_camera/image_color/compressed", "Camera ROS topic");
DEFINE_string(image, "", "dumped image");
DEFINE_string(lidar_scan, "", "dumped pointcloud");
DEFINE_double(checker_width, 0.10775, "rect width size of chessboard");
DEFINE_double(checker_height, 0.10775, "rect height size of chessboard");
DEFINE_double(board_width, 1.1176, "board width(meter)");
DEFINE_double(board_height, 0.889, "board height(meter)");
DEFINE_int32(board_column, 8, "column number of chessboard");
DEFINE_int32(board_row, 6, "row number of chessboard");
DEFINE_double(board_origin_offset_width, -0.201, "board origin width offset (meter)");
DEFINE_double(board_origin_offset_height, -0.151, "board origin height offset (meter)");
DEFINE_double(plane_smoothness_threshold, 7.0,
              "allowable range for the normals deviation in region growing in degrees");
DEFINE_double(plane_curvature_threshold, 1.0, "allowable curvature in same clusters");
DEFINE_int32(max_iter, 1000000, "max iterations of randomly selected plane correspondence");
DEFINE_double(fine_max_iter, 1000, "max iterations for refine alignment of icp");
DEFINE_double(fine_max_corr_dist, 0.1, "max correspondence distance for refine alignment of icp");
DEFINE_int32(num_select_tr, 20, "Select top-K transformation to do refine alignment");

DEFINE_bool(show_images, false, "display images with chessboard corners drawn(to check data)");
DEFINE_bool(show_planes, false, "display detected planes in lidar data");
DEFINE_bool(show_refine_transformation, false, "display transformation result");
DEFINE_bool(show_registration, false, "display the selected planes for registration");
DEFINE_bool(half_resolution, false, "half the resolution of image");
DEFINE_int32(max_board_number, 10, "the max number of board in each images");
DEFINE_bool(fix_normal_sign, true, "Make the estimated normal faces toward the sensor");
DEFINE_string(output_dir, "", "output directory, default: Date_Lidar_to_Camera");
DEFINE_double(display_cam_rotation_y, -0.1, "euler angle of y axis of the display camera pose");
DEFINE_double(display_cam_translation_x, 0.0, "translation x of the display camera pose");
DEFINE_double(display_cam_translation_y, 1.0, "translation y of the display camera pose");
DEFINE_double(display_cam_translation_z, 0.0, "translation z of the display camera pose");

const auto BAG_FORMAT = "%Y%m%dT%H%M%S";
Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ",\n", "", "", "[",
                          "]");

typedef pcl::PointXYZRGBNormal PointXYZRGBNormal;
typedef pcl::PointCloud<PointXYZRGBNormal> PointCloud_XYZRGBNormal;
typedef PointXYZIRT Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointCloud<PointXYZ> PointCloud_XYZ;

struct IdandScore {
    int id;
    double score;
    std::vector<int> list;
};

template <typename T>
class BagSubscriber : public message_filters::SimpleFilter<T> {
    using message_filters::SimpleFilter<T>::signalMessage;

  public:
    void newMessage(const boost::shared_ptr<T const> &msg) { signalMessage(msg); }
};

class BagLoader {
  public:
    struct MessageSet {
        double cloud_timestamp = NAN;
        double img_timestamp = NAN;
        PointCloud::Ptr pointcloud;
        cv::Mat image;
    };

    BagLoader(double sample_rate) : _sample_rate(sample_rate) {}

    typedef std::vector<MessageSet> MessageSetList;
    const MessageSetList &getMessages() { return _messages; }
    void loadBag(const std::string bag_file, std::string lidar_topic, std::string camera_topic) {
        _messages.clear();
        _last_timestamp = 0;
        rosbag::Bag bag;
        std::cout << "\n\nBag file being read is " << bag_file << "\n\n";
        bag.open(bag_file, rosbag::bagmode::Read);
        std::vector<std::string> topics;
        topics.push_back(lidar_topic);
        topics.push_back(camera_topic);
        rosbag::View view(bag, rosbag::TopicQuery(topics));

        BagSubscriber<sensor_msgs::PointCloud2> pointcloud_sub;
        BagSubscriber<sensor_msgs::CompressedImage> camera_sub;

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                                sensor_msgs::CompressedImage>
            MySyncPolicy;

        message_filters::Synchronizer<MySyncPolicy> synchronized_sub(MySyncPolicy(10),
                                                                     pointcloud_sub, camera_sub);
        synchronized_sub.registerCallback(boost::bind(&BagLoader::callback, this, _1, _2));
        for (const auto &m : view) {
            if (m.getTopic() == lidar_topic) {
                sensor_msgs::PointCloud2::ConstPtr pcl_msg =
                    m.instantiate<sensor_msgs::PointCloud2>();
                pointcloud_sub.newMessage(pcl_msg);
            } else if (m.getTopic() == camera_topic) {
                sensor_msgs::CompressedImage::ConstPtr compressed_img_msg =
                    m.instantiate<sensor_msgs::CompressedImage>();
                camera_sub.newMessage(compressed_img_msg);
            }
        }
        LOG(INFO) << "loaded " << _messages.size() << " message sets.";
    }

  private:
    void callback(const sensor_msgs::PointCloud2::ConstPtr &pointcloud_msg,
                  const sensor_msgs::CompressedImage::ConstPtr &image_msg) {
        if (pointcloud_msg->header.stamp.toSec() - _last_timestamp < _sample_rate) {
            return;
        }
        _last_timestamp = pointcloud_msg->header.stamp.toSec();
        MessageSet message_set;

        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*pointcloud_msg, pcl_pc2);
        message_set.cloud_timestamp = _last_timestamp;
        message_set.pointcloud.reset(new PointCloud);
        pcl::fromPCLPointCloud2(pcl_pc2, *message_set.pointcloud);
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat img = cv_ptr->image;
        message_set.image = img;
        message_set.img_timestamp = image_msg->header.stamp.toSec();
        _messages.push_back(message_set);
    }

    MessageSetList _messages;
    double _last_timestamp;
    double _sample_rate;
};

class MultipleChessboardCamLidarCalibration : public drive::common::base::ROSProgram {
  public:
    MultipleChessboardCamLidarCalibration() : ROSProgram("MultipleChessboardCamLidarCalibration") {}

  protected:
    void go() override;

  protected:
    bool init() override;

  private:
    bool findImageChessboardCorners3D(cv::Mat image,
                                      std::vector<PointCloud_XYZRGBNormal::Ptr> &vec_chessboard,
                                      int max_board_number);
    bool findLidarPlanes(PointCloud::Ptr pointcloud,
                         std::vector<PointCloud_XYZRGBNormal::Ptr> &board_candidates);
    void initialPlaneRegister(
        std::vector<PointCloud_XYZRGBNormal::Ptr> cam_planes,
        std::vector<PointCloud_XYZRGBNormal::Ptr> lidar_planes, int num_tr_candidate,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
            &vec_tr_lidar_to_cam);
    bool initial3PlaneSVD(std::vector<PointCloud_XYZRGBNormal::Ptr> planes_A,
                          std::vector<PointCloud_XYZRGBNormal::Ptr> planes_B,
                          Eigen::Matrix4f &tr_B_to_A, double &score);
    void RefineAlignment(PointCloud_XYZRGBNormal::Ptr camera_cloud,
                         PointCloud_XYZRGBNormal::Ptr lidar_cloud, Eigen::Matrix4f &tr_lidar_to_cam,
                         double &score);
    cv::Mat _camera_matrix, _dist_coeffs, _R, _P;
    Eigen::Matrix<double, 3, 4> _p_eigen;
    cv::Size _image_size;
    boost::filesystem::path _output_path;
    std::string _camera_name, _lidar_name, date;
};
bool MultipleChessboardCamLidarCalibration::init() {
    const SensorCalibDB& calib_db = SensorCalibDB::loadSingleton(FLAGS_calib_dir, FLAGS_car);
    CameraCalib::ConstPtr camera_calib = 
            calib_db.getCameraCalib(FLAGS_camera_calib_name, FLAGS_camera_calib_date);
    _camera_matrix = camera_calib->M();
    _dist_coeffs = camera_calib->D();
    _P = camera_calib->P();
    cv::cv2eigen(_P, _p_eigen);
    _P = _P(cv::Range(0, 3), cv::Range(0, 3));
    _R = camera_calib->R();
    _image_size.height = camera_calib->height();
    _image_size.width = camera_calib->width();
    std::vector<std::string> lidar_topic_split;
    boost::split(lidar_topic_split, FLAGS_lidar_topic, boost::is_any_of("/"));
    _lidar_name = lidar_topic_split[1];
    std::vector<std::string> camera_topic_split;
    boost::split(camera_topic_split, FLAGS_camera_topic, boost::is_any_of("/"));
    _camera_name = camera_topic_split[1];

    if (FLAGS_output_dir == "") {
        _output_path = boost::filesystem::path("./");
        boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
        std::stringstream msg;
        boost::posix_time::time_facet *f = new boost::posix_time::time_facet(BAG_FORMAT);
        msg.imbue(std::locale(msg.getloc(), f));
        msg << now;
        date = msg.str();
        msg << "_tr_" << _lidar_name << "_to_" << _camera_name;
        _output_path /= boost::filesystem::path(msg.str());
        if (!boost::filesystem::exists(_output_path.string())) {
            boost::filesystem::create_directories(_output_path.string());
        }
    } else {
        _output_path = boost::filesystem::path(FLAGS_output_dir);
    }

    return true;
}

void MultipleChessboardCamLidarCalibration::go() {
    cv::Mat image;
    PointCloud::Ptr cloud(new PointCloud);
    if (FLAGS_bag_file != "") {
        double sample_rate = 1000;  // we just nees one frame here
        BagLoader bag_loader(sample_rate);
        bag_loader.loadBag(FLAGS_bag_file, FLAGS_lidar_topic, FLAGS_camera_topic);
        const BagLoader::MessageSetList &messages = bag_loader.getMessages();
        messages[0].image.copyTo(image);
        cloud = messages[0].pointcloud;
    } else {  // directly load from file
        image = cv::imread(FLAGS_image, true);
        CHECK(-1 != pcl::io::loadPCDFile<Point>(FLAGS_lidar_scan, *cloud))
            << "could not load " << FLAGS_lidar_scan;
    }
    PointCloud_XYZRGBNormal::Ptr all_lidar_points(new PointCloud_XYZRGBNormal);
    pcl::copyPointCloud(*cloud, *all_lidar_points);
    std::vector<PointCloud_XYZRGBNormal::Ptr> vec_chessboards;
    std::vector<PointCloud_XYZRGBNormal::Ptr> vec_lidar_planes;
    findImageChessboardCorners3D(image, vec_chessboards, FLAGS_max_board_number);
    findLidarPlanes(cloud, vec_lidar_planes);
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
        vec_tr_lidar_to_cam_candidates;
    initialPlaneRegister(vec_chessboards, vec_lidar_planes, FLAGS_num_select_tr,
                         vec_tr_lidar_to_cam_candidates);
    PointCloud_XYZRGBNormal::Ptr all_camera_chessboards(new PointCloud_XYZRGBNormal);
    for (auto pc : vec_chessboards) {
        *all_camera_chessboards += *pc;
    }
    PointCloud_XYZRGBNormal::Ptr all_lidar_planes(new PointCloud_XYZRGBNormal);
    for (auto pc : vec_lidar_planes) {
        *all_lidar_planes += *pc;
    }
    for (int i = 0; i < (int)vec_tr_lidar_to_cam_candidates.size(); i++) {
        Eigen::Matrix4f tr = vec_tr_lidar_to_cam_candidates[i];
        double score;
        // use all planes
        RefineAlignment(all_camera_chessboards, all_lidar_planes, tr, score);
        // use all points
        RefineAlignment(all_camera_chessboards, all_lidar_points, tr, score);

        PointCloud_XYZRGBNormal::Ptr tmp_cloud(new PointCloud_XYZRGBNormal);
        PointCloud_XYZRGBNormal::Ptr tr_result(new PointCloud_XYZRGBNormal);
        pcl::copyPointCloud(*cloud, *tmp_cloud);
        pcl::transformPointCloudWithNormals(*tmp_cloud, *tr_result, tr);
        for (auto &p : tr_result->points) {
            p.r = 50;
            p.g = 50;
            p.b = 50;
        }
        *tr_result += *all_camera_chessboards;

        {  // write down files
            boost::filesystem::path ss_pc =
                _output_path /
                boost::filesystem::path(_lidar_name + "_" + std::to_string(i) + "_result.pcd");
            pcl::io::savePCDFileASCII(ss_pc.string(), *tr_result);

            Eigen::Affine3f new_camera_pose =
                Eigen::Translation3f(FLAGS_display_cam_translation_x,
                                     FLAGS_display_cam_translation_y,
                                     FLAGS_display_cam_translation_z) *
                Eigen::AngleAxisf(FLAGS_display_cam_rotation_y, Eigen::Vector3f::UnitY());
            cv::Mat project_image(_image_size.height, _image_size.width, CV_8UC3,
                                  cv::Scalar(255, 255, 255));
            for (auto &p : tr_result->points) {
                Eigen::Matrix<float, 3, 4> _p_eigen_display = _p_eigen.cast<float>();

                _p_eigen_display(0, 0) -= 300;
                _p_eigen_display(1, 1) -= 300;

                Eigen::Vector3f proj_p = _p_eigen_display * new_camera_pose.matrix() *
                                         Eigen::Vector4f(p.x, p.y, p.z, 1.0);
                proj_p(0) /= proj_p(2);
                proj_p(1) /= proj_p(2);
                if (proj_p(2) > 0 and proj_p(0) > 0 and proj_p(0) < _image_size.width and
                    proj_p(1) > 0 and proj_p(1) < _image_size.height) {
                    cv::circle(project_image, cv::Point(proj_p(0), proj_p(1)), 2,
                               cv::Scalar(p.b, p.g, p.r, 0.5), -1);
                }
            }
            boost::filesystem::path ss_proj_img =
                _output_path /
                boost::filesystem::path(_lidar_name + "_" + std::to_string(i) + "_projection.png");
            cv::imwrite(ss_proj_img.string(), project_image);

            boost::filesystem::path ss_tr =
                _output_path /
                boost::filesystem::path(_lidar_name + "_" + std::to_string(i) + "_result.yml");
            cv::FileStorage fs(ss_tr.string(), cv::FileStorage::WRITE);
            Eigen::Matrix3f r_eigen = tr.block(0, 0, 3, 3);
            Eigen::Vector3f t_eigen = tr.block(0, 3, 3, 1);
            cv::Mat r;
            cv::eigen2cv(Eigen::Matrix3d(r_eigen.cast<double>()), r);
            cv::Mat t;
            cv::eigen2cv(Eigen::Vector3d(t_eigen.cast<double>()), t);
            fs << "chessboard_detected" << (int)vec_chessboards.size();
            fs << "Calibration_Date" << date;
            fs << "average_of_sqaure_distance" << score / vec_chessboards.size();
            fs << "tr_lidar_to_cam_R" << r << "tr_lidar_to_cam_T" << t;
            auto tr_inv = tr.inverse();
            r_eigen = tr_inv.block(0, 0, 3, 3);
            t_eigen = tr_inv.block(0, 3, 3, 1);
            cv::eigen2cv(Eigen::Matrix3d(r_eigen.cast<double>()), r);
            cv::eigen2cv(Eigen::Vector3d(t_eigen.cast<double>()), t);
            fs << "tr_cam_to_lidar_R" << r << "tr_cam_to_lidar_T" << t;
            fs.release();
        }
    }
}

bool MultipleChessboardCamLidarCalibration::findImageChessboardCorners3D(
    cv::Mat image, std::vector<PointCloud_XYZRGBNormal::Ptr> &vec_chessboard,
    int max_board_number) {
    std::vector<std::vector<cv::Point2f>> vec_corners;
    cv::Size board_pattern(FLAGS_board_column, FLAGS_board_row);
    bool found = false;
    int flags = cv::CALIB_CB_FAST_CHECK;
    cv::Mat img, display_img;
    if (FLAGS_half_resolution) {
        cv::pyrDown(image, img, cv::Size(image.cols / 2, image.rows / 2));
        img.copyTo(display_img);
        cv::cvtColor(img, img, CV_BGR2GRAY);
    } else {
        cv::cvtColor(image, img, CV_BGR2GRAY);
        image.copyTo(display_img);
    }
    for (int i = 0; i < max_board_number; i++) {
        std::vector<cv::Point2f> corners;
        found = cv::findChessboardCorners(img, board_pattern, corners, flags);
        if (!found) {
            // refind the chessboard by using other threshold
            flags = cv::CALIB_CB_ADAPTIVE_THRESH;
            flags |= cv::CALIB_CB_NORMALIZE_IMAGE;
            found = cv::findChessboardCorners(img, board_pattern, corners, flags);
        }
        if (!found) {
            continue;
        }
        cv::cornerSubPix(
            img, corners, cv::Size(5, 5), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.001));
        auto rotatedRectangle = cv::minAreaRect(corners);
        cv::Point2f vertices2f[4];
        rotatedRectangle.points(vertices2f);
        cv::Point vertices[4];
        for (int id = 0; id < 4; ++id) {
            vertices[id] = vertices2f[id];
        }
        cv::fillConvexPoly(img, vertices, 4, cv::Scalar(0, 0, 0));
        cv::drawChessboardCorners(display_img, board_pattern, corners, true);
        cv::putText(display_img, std::to_string(vec_corners.size()), corners[0], 3, 1.0,
                    cv::Scalar(255, 0, 0));
        vec_corners.push_back(corners);
    }
    LOG(INFO) << "Find " << vec_corners.size() << " board";
    if (FLAGS_show_images) {
        cv::imshow("corners", display_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    if (vec_corners.size() < 3) {
        LOG(ERROR) << "Need at least 3 chessboards(" << vec_corners.size() << ")";
        return false;
    }

    std::vector<cv::Point3f> object_points;
    for (int j = 0; j < FLAGS_board_row; j++) {
        for (int k = 0; k < FLAGS_board_column; k++) {
            object_points.push_back(
                cv::Point3f(k * FLAGS_checker_width, j * FLAGS_checker_height, 0));
        }
    }
    // outside physical boarder
    std::vector<cv::Point3f> bounding_points;
    // generate the ground truth board
    cv::Size2f offset(FLAGS_board_origin_offset_width, FLAGS_board_origin_offset_height);
    for (int i = 0; i <= 10; i++) {
        cv::Point3f p;
        p.x = i * FLAGS_board_width / 10.0 + offset.width;
        p.y = offset.height;
        p.z = 0;
        bounding_points.push_back(p);
    }
    for (int i = 0; i <= 10; i++) {
        cv::Point3f p;
        p.x = i * FLAGS_board_width / 10.0 + offset.width;
        p.y = FLAGS_board_height + offset.height;
        p.z = 0;
        bounding_points.push_back(p);
    }
    for (int i = 0; i <= 10; i++) {
        cv::Point3f p;
        p.x = offset.width;
        p.y = offset.height + i * FLAGS_board_height / 10.0;
        p.z = 0;
        bounding_points.push_back(p);
    }
    for (int i = 0; i <= 10; i++) {
        cv::Point3f p;
        p.x = offset.width + FLAGS_board_width;
        p.y = offset.height + i * FLAGS_board_height / 10.0;
        p.z = 0;
        bounding_points.push_back(p);
    }

    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<PointCloud_XYZRGBNormal::Ptr> camera_clouds;
    PointCloud_XYZRGBNormal::Ptr display_cloud(new PointCloud_XYZRGBNormal);
    cv::RNG rng(12345);
    for (size_t id = 0; id < vec_corners.size(); id++) {
        PointCloud_XYZRGBNormal::Ptr chessboard(new PointCloud_XYZRGBNormal);
        cv::Scalar color =
            cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

        // get corner at undistorted and rectified frame
        std::vector<cv::Point2f> corners;
        cv::undistortPoints(vec_corners[id], corners, _camera_matrix, _dist_coeffs, _R, _P);
        cv::Mat rvec, tvec;
        // cv::Mat rvec, tvec;
        cv::solvePnP(cv::Mat(object_points), corners, _P, cv::Mat(), rvec, tvec);
        rvecs.push_back(rvec);
        tvecs.push_back(tvec);

        // transform points to 3D in camera coordinate system and put them in a cloud
        cv::Mat rot = cv::Mat::zeros(3, 3, CV_64F);
        cv::Rodrigues(rvec, rot);
        Eigen::Affine3f xform;
        xform.matrix() = Eigen::Matrix4f::Identity();
        Eigen::Matrix3f rot_eigen;
        cv::cv2eigen(rot, rot_eigen);
        xform.matrix().block(0, 0, 3, 3) = rot_eigen;
        xform(0, 3) = tvec.at<double>(0);
        xform(1, 3) = tvec.at<double>(1);
        xform(2, 3) = tvec.at<double>(2);
        Eigen::Vector3f normal(rot_eigen(0, 2), rot_eigen(1, 2), rot_eigen(2, 2));
        normal.normalized();
        if (FLAGS_fix_normal_sign) {
            Eigen::Vector3f center(0, 0, 0);
            for (size_t i = 0; i < object_points.size(); i++) {
                Eigen::Vector3f pt(object_points[i].x, object_points[i].y, object_points[i].z);
                pt = xform * pt;
                center += pt;
            }
            center /= object_points.size();
            normal *= normal.dot(center) > 0 ? -1.0 : 1.0;
        }

        std::vector<cv::Point2f> image_points;
        for (size_t i = 0; i < object_points.size(); i++) {
            PointXYZRGBNormal p;
            p.r = color[0];
            p.g = color[1];
            p.b = color[2];
            p.x = object_points[i].x;
            p.y = object_points[i].y;
            p.z = object_points[i].z;
            p.normal_x = normal(0);
            p.normal_y = normal(1);
            p.normal_z = normal(2);

            p.getVector3fMap() = xform * p.getVector3fMap();
            display_cloud->push_back(p);
            chessboard->push_back(p);
        }
        for (size_t i = 0; i < bounding_points.size(); i++) {
            PointXYZRGBNormal p;
            p.r = 128;
            p.g = 128;
            p.b = 0;
            p.x = bounding_points[i].x;
            p.y = bounding_points[i].y;
            p.z = bounding_points[i].z;
            p.normal_x = normal(0);
            p.normal_y = normal(1);
            p.normal_z = normal(2);
            p.getVector3fMap() = xform * p.getVector3fMap();
            display_cloud->push_back(p);
            chessboard->push_back(p);
        }
        vec_chessboard.push_back(chessboard);
    }
    boost::filesystem::path image_with_chessboard =
        _output_path / boost::filesystem::path(_camera_name + "_corners.png");
    cv::imwrite(image_with_chessboard.string(), display_img);
    boost::filesystem::path chessboard_pointcloud =
        _output_path / boost::filesystem::path(_camera_name + "_pointcloud.pcd");
    pcl::io::savePCDFileASCII(chessboard_pointcloud.string(), *display_cloud);
    return true;
}
bool MultipleChessboardCamLidarCalibration::findLidarPlanes(
    PointCloud::Ptr pointcloud, std::vector<PointCloud_XYZRGBNormal::Ptr> &board_candidates) {
    PointCloud_XYZ::Ptr pc_xyz(new PointCloud_XYZ);
    pcl::copyPointCloud(*pointcloud, *pc_xyz);
    // find planes in lidar
    double diagonal =
        std::sqrt(FLAGS_board_width * FLAGS_board_width + FLAGS_board_height * FLAGS_board_height);
    size_t max_count = 0;
    cv::RNG rng(12345);
    pcl::NormalEstimation<PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<PointXYZ>::Ptr tree(new pcl::search::KdTree<PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setInputCloud(pc_xyz);
    ne.setKSearch(40);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*cloud_normals);

    pcl::RegionGrowing<PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(10);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(20);
    reg.setInputCloud(pc_xyz);
    reg.setInputNormals(cloud_normals);
    reg.setSmoothnessThreshold(FLAGS_plane_smoothness_threshold / 180.0 * M_PI);
    reg.setCurvatureThreshold(FLAGS_plane_curvature_threshold);
    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    PointCloud_XYZRGBNormal::Ptr display_lidar_planes(new PointCloud_XYZRGBNormal);
    for (size_t i = 0; i < clusters.size(); i++) {
        pcl::ExtractIndices<PointXYZ> extract;
        extract.setInputCloud(pc_xyz);
        extract.setIndices(boost::make_shared<const pcl::PointIndices>(clusters[i]));
        PointCloud_XYZ::Ptr board_candidate_xyz(new PointCloud_XYZ);
        extract.setNegative(false);
        extract.filter(*board_candidate_xyz);
        PointXYZ min, max;
        pcl::getMinMax3D(*board_candidate_xyz, min, max);
        Eigen::Vector3f diff = max.getVector3fMap() - min.getVector3fMap();
        // remove the too large or too small planes
        if (diff.norm() < (1.1) * diagonal and diff.norm() > 0.5 * diagonal) {
            cv::Scalar color =
                cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            PointCloud_XYZRGBNormal::Ptr board_candidate(new PointCloud_XYZRGBNormal);
            pcl::PCA<PointXYZ> pca;
            pca.setInputCloud(board_candidate_xyz);
            Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
            Eigen::Vector3f normal(eigen_vectors(0, 2), eigen_vectors(1, 2), eigen_vectors(2, 2));
            normal.normalize();
            if (FLAGS_fix_normal_sign) {
                Eigen::Vector3f center(0, 0, 0);
                for (auto p : board_candidate_xyz->points) {
                    center += Eigen::Vector3f(p.x, p.y, p.z);
                }
                center /= board_candidate_xyz->size();
                normal *= normal.dot(center) > 0 ? -1.0 : 1.0;
            }
            for (auto p_xyz : board_candidate_xyz->points) {
                PointXYZRGBNormal p;
                p.x = p_xyz.x;
                p.y = p_xyz.y;
                p.z = p_xyz.z;
                p.r = color[0];
                p.g = color[1];
                p.b = color[2];
                p.normal_x = normal(0);
                p.normal_y = normal(1);
                p.normal_z = normal(2);
                board_candidate->push_back(p);
                display_lidar_planes->push_back(p);
            }
            board_candidates.push_back(board_candidate);
        }
    }
    if (FLAGS_show_planes) {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("detected lidar planes"));
        viewer->initCameraParameters();
        viewer->addCoordinateSystem(1.0, "axes");
        viewer->addPointCloud<PointXYZRGBNormal>(display_lidar_planes, "lidar_planes");
        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
        viewer->close();
    }
    boost::filesystem::path lidar_planes =
        _output_path / boost::filesystem::path(_lidar_name + "_planes.pcd");
    pcl::io::savePCDFileASCII(lidar_planes.string(), *display_lidar_planes);
    if (board_candidates.size() < 3) {
        LOG(ERROR) << "Need at least 3 planes( " << board_candidates.size()
                   << "), you should adjust region growing parameters";
        return false;
    }
    return true;
}

void MultipleChessboardCamLidarCalibration::initialPlaneRegister(
    std::vector<PointCloud_XYZRGBNormal::Ptr> cam_planes,
    std::vector<PointCloud_XYZRGBNormal::Ptr> lidar_planes, int num_tr_candidate,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &vec_tr_lidar_to_cam) {
    int select_num = 3;
    int cam_plane_combination_sz = 5;
    int lidar_plane_size = (int)lidar_planes.size();
    int cam_plane_size = (int)cam_planes.size();

    // at least 3 planes
    if (lidar_plane_size < 3 or cam_plane_size < 3) {
        LOG(ERROR) << "need at least 3 planes"
                   << ", camera: " << lidar_plane_size << " planes, lidar: " << cam_plane_size
                   << " planes";
        return;
    }

    // choose the most perpendicular 3 planes combinations in cam_planes

    std::vector<std::vector<int>> cam_plane_combinations;
    {
        std::vector<int> chessboard_ids;
        for (int i = 0; i < cam_plane_size; i++) {
            chessboard_ids.push_back(i);
        }
        std::vector<bool> v(cam_plane_size);
        std::fill(v.begin(), v.begin() + select_num, true);

        do {
            std::vector<int> combination;
            for (int i = 0; i < cam_plane_size; ++i) {
                if (v[i]) {
                    combination.push_back(chessboard_ids[i]);
                }
            }
            cam_plane_combinations.push_back(combination);
        } while (std::prev_permutation(v.begin(), v.end()));

        // sort the combination
        std::vector<Eigen::Vector3f> cam_plane_normals;
        for (int i = 0; i < cam_plane_size; i++) {
            Eigen::Vector3f normal(cam_planes[i]->points[0].normal_x,
                                   cam_planes[i]->points[0].normal_y,
                                   cam_planes[i]->points[0].normal_z);
            cam_plane_normals.push_back(normal);
        }
        std::sort(
            cam_plane_combinations.begin(), cam_plane_combinations.end(),
            [&cam_plane_normals](const std::vector<int> &a, const std::vector<int> &b) -> bool {
                double normal_score_a =
                    std::abs(cam_plane_normals[a[0]].dot(cam_plane_normals[a[1]])) +
                    std::abs(cam_plane_normals[a[0]].dot(cam_plane_normals[a[2]])) +
                    std::abs(cam_plane_normals[a[1]].dot(cam_plane_normals[a[2]]));
                double normal_score_b =
                    std::abs(cam_plane_normals[b[0]].dot(cam_plane_normals[b[1]])) +
                    std::abs(cam_plane_normals[b[0]].dot(cam_plane_normals[b[2]])) +
                    std::abs(cam_plane_normals[b[1]].dot(cam_plane_normals[b[2]]));
                return a > b;
            });

        if ((int)cam_plane_combinations.size() > cam_plane_combination_sz) {
            cam_plane_combinations.resize(cam_plane_combination_sz);
        } else {
            cam_plane_combination_sz = cam_plane_combinations.size();
        }
    }
    // random pick 3 planes in lidar and 1 pair in camera_planes
    int chessboard_select_a, chessboard_select_b, chessboard_select_c;
    int lidar_plane_select_a, lidar_plane_select_b, lidar_plane_select_c;
    int count = 0;
    double max_score = -100000;
    std::vector<IdandScore> score_vec;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> tf_vec;
    std::unordered_map<double, int> map;
    do {
        int rand_a = rand();
        chessboard_select_a = cam_plane_combinations[rand_a % cam_plane_combination_sz][0];
        chessboard_select_b = cam_plane_combinations[rand_a % cam_plane_combination_sz][1];
        chessboard_select_c = cam_plane_combinations[rand_a % cam_plane_combination_sz][2];
        lidar_plane_select_a = rand() % lidar_plane_size;
        do {
            lidar_plane_select_b = rand() % lidar_plane_size;
        } while (lidar_plane_select_a == lidar_plane_select_b);
        do {
            lidar_plane_select_c = rand() % lidar_plane_size;
        } while (lidar_plane_select_a == lidar_plane_select_c or
                 lidar_plane_select_b == lidar_plane_select_c);
        double hash_value = 3357.3 * chessboard_select_a * chessboard_select_a +
                            7.7 * chessboard_select_b +
                            9.55 * chessboard_select_c * chessboard_select_c * chessboard_select_c +
                            8.412 * lidar_plane_select_a + 10485 * lidar_plane_select_b +
                            0.511 * lidar_plane_select_c * lidar_plane_select_c;
        auto got = map.find(hash_value);
        double tmp_score = max_score;
        Eigen::Matrix4f tr_lidar_to_cam = Eigen::Matrix4f::Identity();
        if (got == map.end()) {
            std::vector<PointCloud_XYZRGBNormal::Ptr> planes_camera;
            planes_camera.push_back(cam_planes[chessboard_select_a]);
            planes_camera.push_back(cam_planes[chessboard_select_b]);
            planes_camera.push_back(cam_planes[chessboard_select_c]);
            std::vector<PointCloud_XYZRGBNormal::Ptr> planes_lidar;
            planes_lidar.push_back(lidar_planes[lidar_plane_select_a]);
            planes_lidar.push_back(lidar_planes[lidar_plane_select_b]);
            planes_lidar.push_back(lidar_planes[lidar_plane_select_c]);
            initial3PlaneSVD(planes_camera, planes_lidar, tr_lidar_to_cam, tmp_score);
            map.insert({hash_value, 1});
        }
        std::vector<int> list = {chessboard_select_a,  chessboard_select_b,  chessboard_select_c,
                                 lidar_plane_select_a, lidar_plane_select_b, lidar_plane_select_c};
        IdandScore id_s = {count, tmp_score, list};
        score_vec.push_back(id_s);
        tf_vec.push_back(tr_lidar_to_cam);
        count++;
    } while (count < FLAGS_max_iter);

    std::sort(score_vec.begin(), score_vec.end(),
              [](const IdandScore &a, const IdandScore &b) -> bool { return a.score > b.score; });
    for (int i = 0; i < num_tr_candidate; i++) {
        LOG(INFO) << "id " << i;
        LOG(INFO) << "score " << score_vec[i].score;
        LOG(INFO) << "initial tr " << tf_vec[score_vec[i].id].format(OctaveFmt);
        vec_tr_lidar_to_cam.push_back(tf_vec[score_vec[i].id]);
        if (FLAGS_show_registration) {
            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
                new pcl::visualization::PCLVisualizer("registration CHECK"));
            viewer->initCameraParameters();
            viewer->addCoordinateSystem(1.0, "axes");
            PointCloud_XYZRGBNormal::Ptr tr_lidar_planes_c(new PointCloud_XYZRGBNormal);
            viewer->addPointCloud<PointXYZRGBNormal>(lidar_planes[score_vec[i].list[3]],
                                                     "lidar_cloud_a");
            viewer->addPointCloud<PointXYZRGBNormal>(lidar_planes[score_vec[i].list[4]],
                                                     "lidar_cloud_b");
            viewer->addPointCloud<PointXYZRGBNormal>(lidar_planes[score_vec[i].list[5]],
                                                     "lidar_cloud_c");
            viewer->addPointCloudNormals<PointXYZRGBNormal>(lidar_planes[score_vec[i].list[3]], 20,
                                                            0.2, "lidar_cloud_a_normal");
            viewer->addPointCloudNormals<PointXYZRGBNormal>(lidar_planes[score_vec[i].list[4]], 20,
                                                            0.2, "lidar_cloud_b_normal");
            viewer->addPointCloudNormals<PointXYZRGBNormal>(lidar_planes[score_vec[i].list[5]], 20,
                                                            0.2, "lidar_cloud_c_normal");

            viewer->addPointCloud<PointXYZRGBNormal>(cam_planes[score_vec[i].list[0]],
                                                     "camera_cloud_a");
            viewer->addPointCloud<PointXYZRGBNormal>(cam_planes[score_vec[i].list[1]],
                                                     "camera_cloud_b");
            viewer->addPointCloud<PointXYZRGBNormal>(cam_planes[score_vec[i].list[2]],
                                                     "camera_cloud_c");
            viewer->addPointCloudNormals<PointXYZRGBNormal>(cam_planes[score_vec[i].list[0]], 20,
                                                            0.2, "camera_cloud_a_normal");
            viewer->addPointCloudNormals<PointXYZRGBNormal>(cam_planes[score_vec[i].list[1]], 20,
                                                            0.2, "camera_cloud_b_normal");
            viewer->addPointCloudNormals<PointXYZRGBNormal>(cam_planes[score_vec[i].list[2]], 20,
                                                            0.2, "camera_cloud_c_normal");
            viewer->addLine(cam_planes[score_vec[i].list[0]]->points[0],
                            lidar_planes[score_vec[i].list[3]]->points[0], 255.0, 0.0, 0.0,
                            "corr_a", 0);
            viewer->addLine(cam_planes[score_vec[i].list[1]]->points[0],
                            lidar_planes[score_vec[i].list[4]]->points[0], 0.0, 255.0, 0.0,
                            "corr_b", 0);
            viewer->addLine(cam_planes[score_vec[i].list[2]]->points[0],
                            lidar_planes[score_vec[i].list[5]]->points[0], 0.0, 0.0, 255.0,
                            "corr_c", 0);
            while (!viewer->wasStopped()) {
                viewer->spinOnce(100);
                boost::this_thread::sleep(boost::posix_time::microseconds(100000));
            }
            viewer->close();
        }
    }
    return;
}
bool MultipleChessboardCamLidarCalibration::initial3PlaneSVD(
    std::vector<PointCloud_XYZRGBNormal::Ptr> planes_A,
    std::vector<PointCloud_XYZRGBNormal::Ptr> planes_B, Eigen::Matrix4f &tr_B_to_A, double &score) {
    if (planes_A.size() != planes_B.size()) {
        LOG(ERROR) << "planes_A.size() != planes_B.size()";
        return false;
    }
    if (planes_A.size() != 3) {
        LOG(ERROR) << "planes_A.size() != 3";
        return false;
    }
    std::vector<Eigen::Vector3f> vec_a_centroid;
    std::vector<Eigen::Vector3f> vec_b_centroid;
    size_t camera_point_count = 0, lidar_point_count = 0;
    for (size_t i = 0; i < planes_A.size(); i++) {
        Eigen::Vector3f a_centroid(0, 0, 0);
        Eigen::Vector3f b_centroid(0, 0, 0);
        camera_point_count += planes_A[i]->size();
        for (auto p : planes_A[i]->points) {
            a_centroid += p.getVector3fMap();
        }
        vec_a_centroid.push_back(a_centroid / planes_A[i]->size());
        lidar_point_count += planes_B[i]->size();
        for (auto p : planes_B[i]->points) {
            b_centroid += p.getVector3fMap();
        }
        vec_b_centroid.push_back(b_centroid / planes_B[i]->size());
    }

    double max_rotation_score = -1000000;
    Eigen::Matrix3f best_r;
    std::vector<Eigen::Vector3f> normals_A;
    auto p_c_a = planes_A[0]->points[0];
    auto p_c_b = planes_A[1]->points[0];
    auto p_c_c = planes_A[2]->points[0];
    Eigen::Vector3f normal_A_a(p_c_a.normal_x, p_c_a.normal_y, p_c_a.normal_z);
    Eigen::Vector3f normal_A_b(p_c_b.normal_x, p_c_b.normal_y, p_c_b.normal_z);
    Eigen::Vector3f normal_A_c(p_c_c.normal_x, p_c_c.normal_y, p_c_c.normal_z);
    normal_A_a.normalize();
    normal_A_b.normalize();
    normal_A_c.normalize();

    normals_A.push_back(normal_A_a);
    normals_A.push_back(normal_A_b);
    normals_A.push_back(normal_A_c);

    auto p_l_a = planes_B[0]->points[0];
    auto p_l_b = planes_B[1]->points[0];
    auto p_l_c = planes_B[2]->points[0];
    Eigen::Vector3f normal_B__a(p_l_a.normal_x, p_l_a.normal_y, p_l_a.normal_z);
    Eigen::Vector3f normal_B__b(p_l_b.normal_x, p_l_b.normal_y, p_l_b.normal_z);
    Eigen::Vector3f normal_B__c(p_l_c.normal_x, p_l_c.normal_y, p_l_c.normal_z);
    normal_B__a.normalize();
    normal_B__b.normalize();
    normal_B__c.normalize();

    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    cov += normal_B__a * (normal_A_a.transpose());
    cov += normal_B__b * (normal_A_b.transpose());
    cov += normal_B__c * (normal_A_c.transpose());
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3f r = svd.matrixV() * svd.matrixU().transpose();
    if (r.determinant() < 0) {
        Eigen::Matrix3f tmp = Eigen::Matrix3f::Identity();
        tmp(2, 2) = r.determinant();
        r = svd.matrixV() * tmp * svd.matrixU().transpose();
    }

    // solve point to plane distance
    Eigen::Matrix3f A;
    A.block(0, 0, 1, 3) = normals_A[0].transpose();
    A.block(1, 0, 1, 3) = normals_A[1].transpose();
    A.block(2, 0, 1, 3) = normals_A[2].transpose();
    Eigen::Vector3f B;
    B(0) = -normals_A[0].dot(r * vec_b_centroid[0] - vec_a_centroid[0]);
    B(1) = -normals_A[1].dot(r * vec_b_centroid[1] - vec_a_centroid[1]);
    B(2) = -normals_A[2].dot(r * vec_b_centroid[2] - vec_a_centroid[2]);
    Eigen::Vector3f t = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    tr_B_to_A = Eigen::Matrix4f::Identity();
    tr_B_to_A.block(0, 0, 3, 3) = r;
    tr_B_to_A.block(0, 3, 3, 1) = t;
    score = 0.0;
    for (size_t i = 0; i < vec_a_centroid.size(); i++) {
        Eigen::Vector3f tr_b_centroid = (tr_B_to_A * vec_b_centroid[i].homogeneous()).hnormalized();
        Eigen::Vector3f diff = vec_a_centroid[i] - tr_b_centroid;
        float p_to_plane = diff.dot(normals_A[i]);
        float p_to_p = diff.norm();
        score += -(p_to_plane * p_to_plane + 0.01 * p_to_p * p_to_p);
    }
    return true;
}

void MultipleChessboardCamLidarCalibration::RefineAlignment(
    PointCloud_XYZRGBNormal::Ptr camera_cloud, PointCloud_XYZRGBNormal::Ptr lidar_cloud,
    Eigen::Matrix4f &tr_lidar_to_cam, double &score) {
    Eigen::Matrix4f initial_tr = tr_lidar_to_cam;
    PointCloud_XYZRGBNormal::Ptr init_tr(new PointCloud_XYZRGBNormal);
    pcl::transformPointCloudWithNormals(*lidar_cloud, *init_tr, tr_lidar_to_cam);

    pcl::IterativeClosestPoint<PointXYZRGBNormal, PointXYZRGBNormal> icp;
    icp.setInputSource(init_tr);
    icp.setInputTarget(camera_cloud);
    icp.setMaxCorrespondenceDistance(FLAGS_fine_max_corr_dist);
    icp.setMaximumIterations(FLAGS_fine_max_iter);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);

    PointCloud_XYZRGBNormal::Ptr fine_registered(new PointCloud_XYZRGBNormal);
    icp.align(*fine_registered);
    score = icp.getFitnessScore();
    tr_lidar_to_cam = icp.getFinalTransformation() * initial_tr;
    LOG(INFO) << "refine fitness score: " << score;
    LOG(INFO) << "refine transform(tr_lidar_to_cam) is: ";
    LOG(INFO) << endl << tr_lidar_to_cam.format(OctaveFmt);
    LOG(INFO) << "refine transform(tr_cam_to_lidar) is: ";
    LOG(INFO) << endl << tr_lidar_to_cam.inverse().format(OctaveFmt);

    pcl::transformPointCloudWithNormals(*lidar_cloud, *fine_registered, tr_lidar_to_cam);

    if (FLAGS_show_refine_transformation) {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("fine registration"));
        viewer->initCameraParameters();
        viewer->addCoordinateSystem(1.0, "axes");
        viewer->addPointCloud<PointXYZRGBNormal>(fine_registered, "lidar_cloud");
        // viewer->addPointCloud<Point>(lidar_cloud, "oringal_lidar_cloud");
        viewer->addPointCloud<PointXYZRGBNormal>(camera_cloud, "camera_cloud");
        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
        viewer->close();
    }
}

int main(int argc, char **argv) {
    MultipleChessboardCamLidarCalibration p;
    return p.run(argc, argv);
}
