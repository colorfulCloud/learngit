#include <calib_checker/radar_based_stereo_calib_checker.h>

using namespace drive::common::perception;
using namespace drive::perception::sensor;

namespace drive {
namespace perception {
namespace calib_checker {

RadarBasedStereoCalibChecker::RadarBasedStereoCalibChecker(
        const RadarStereoParam& radar_stereo_param,
        StereoCameraCalib::ConstPtr stereo_calib,
        RadarCalib::ConstPtr radar_calib)
    : _stereo_calib(stereo_calib),
      _radar_calib(radar_calib),
      _radar_stereo_param(radar_stereo_param) {
    cv::cv2eigen(_stereo_calib->R(), _init_R);
    cv::cv2eigen(_stereo_calib->T(), _init_T);
    cv::cv2eigen(_stereo_calib->R(), _calib_R);
    cv::cv2eigen(_stereo_calib->T(), _calib_T);
    _inv_Kl = _stereo_calib->M1().inv();
    _inv_Kr = _stereo_calib->M2().inv();

    _feature_tracker.reset(new FeatureTracker(stereo_calib));
    _radar_filter.reset(new RadarFilter(radar_stereo_param.radar_filter_param()));
    _radar_image_associate.reset(
            new RadarImageAssociate(radar_stereo_param.radar_image_associate_param()));
    _radar_box_matcher.reset(new RadarBoxMatcher());

    // init parameter blocks
    Eigen::Map<Eigen::Vector3d> trans(_translation_param);
    Eigen::Map<Eigen::Quaterniond> quat(_rotation_param);

    // std::default_random_engine engine(std::random_device{}());
    // std::uniform_real_distribution<double> rot_noise(-3 / 180. * M_PI, 3 / 180. * M_PI);
    // _init_R = _init_R * \
    //             Eigen::AngleAxisd(rot_noise(engine), Eigen::Vector3d::UnitX()) *\
    //             Eigen::AngleAxisd(rot_noise(engine), Eigen::Vector3d::UnitY()) *\
    //             Eigen::AngleAxisd(rot_noise(engine), Eigen::Vector3d::UnitZ());

    trans = _init_T;
    quat = Eigen::Quaterniond(_init_R);

    EpipolarError::setSqrtInfo(_stereo_calib->M1().at<double>(0, 0));

    _undist_depth_keypts_l.reserve(_max_depth_feature_num);
    _undist_depth_keypts_r.reserve(_max_depth_feature_num);
    _depth.reserve(_max_depth_feature_num);
}

void RadarBasedStereoCalibChecker::distortCameraDetections(
        const std::vector<cv::Rect2d>& camera_detections,
        std::vector<cv::Rect2d>& camera_detections_distort) {
    camera_detections_distort.clear();

    // TODO: implement with undistort mapx mapy when exported PR in common repo gets merged
    for (auto const& det : camera_detections) {
        cv::Point2d left_top(det.x, det.y);
        cv::Point2d right_bottom(det.x + det.width, det.y + det.height);
        std::vector<cv::Point2d> distort_pts;
        distort_pts.push_back(left_top);
        distort_pts.push_back(right_bottom);
        std::vector<cv::Point2d> undistort_pts;
        const int max_iters = 5;
        for (int i = 0; i < max_iters; i++) {
            cv::undistortPoints(distort_pts,
                                undistort_pts,
                                _stereo_calib->M1(),
                                _stereo_calib->D1(),
                                _stereo_calib->R1(),
                                _stereo_calib->P1());

            distort_pts[0] += left_top - undistort_pts[0];
            distort_pts[1] += right_bottom - undistort_pts[1];
        }
        cv::Point2d size = distort_pts[1] - distort_pts[0];
        camera_detections_distort.push_back(
                cv::Rect2d(distort_pts[0].x, distort_pts[0].y, size.x, size.y));
    }
}

void RadarBasedStereoCalibChecker::radarProjAssociate(
        const cv::Mat img_l,
        const cv::Mat img_r,
        radar_msgs::RadarDetectionArray::ConstPtr radar_detections,
        const std::vector<cv::Rect2d>& camera_detections,
        std::vector<cv::Rect2d>& boxes,
        std::vector<double>& depth) {
    boxes.clear();
    depth.clear();
    if (camera_detections.empty()) {
        return;
    }

    // unified radar is already in imu coordinate system
    cv::Mat Tr_left_to_stereo = cv::Mat::eye(4, 4, CV_64F);
    _stereo_calib->R1().copyTo(Tr_left_to_stereo.rowRange(0, 3).colRange(0, 3));
    _Tr_radar_to_left = Tr_left_to_stereo.inv() * (CalibCheckerBase::_s_Tr_cam_to_imu_init.inv());

    radar_msgs::RadarDetectionArray::Ptr radar_detections_filtered(
            new radar_msgs::RadarDetectionArray);
    _radar_filter->filter(radar_detections, radar_detections_filtered);
    if (radar_detections_filtered->detections.empty()) {
        return;
    }

    std::vector<cv::Point3d> radar_pts_cam;
    for (auto const& det : radar_detections_filtered->detections) {
        cv::Mat pt_cam =
                (cv::Mat_<double>(4, 1) << det.position.x, det.position.y, det.position.z, 1);
        pt_cam = _Tr_radar_to_left * pt_cam;
        pt_cam /= pt_cam.at<double>(3, 0);
        radar_pts_cam.push_back(cv::Point3d(
                pt_cam.at<double>(0, 0), pt_cam.at<double>(1, 0), pt_cam.at<double>(2, 0)));
    }

    // project radar detection centers to left image
    std::vector<cv::Point2d> radar_pts_cam2d;
    cv::projectPoints(radar_pts_cam,
                      cv::Mat::eye(3, 3, CV_64F),
                      cv::Mat::zeros(3, 1, CV_64F),
                      _stereo_calib->M1(),
                      _stereo_calib->D1(),
                      radar_pts_cam2d);
    // project radar detection centers to right image
    std::vector<cv::Point2d> radar_pts_cam2d_right;
    cv::projectPoints(radar_pts_cam,
                      _stereo_calib->R(),
                      _stereo_calib->T(),
                      _stereo_calib->M2(),
                      _stereo_calib->D2(),
                      radar_pts_cam2d_right);

    std::vector<cv::Rect2d> radar_boxes;
    for (size_t i = 0; i < radar_pts_cam2d.size(); i++) {
        int bbox_width = _stereo_calib->M1().at<double>(0, 0) *
                         _radar_stereo_param.radar_fix_width() / radar_pts_cam[i].z;
        int bbox_height = _stereo_calib->M1().at<double>(0, 0) *
                          _radar_stereo_param.radar_fix_height() / radar_pts_cam[i].z;
        cv::Rect2d left_box(radar_pts_cam2d[i].x - bbox_width / 2,
                            radar_pts_cam2d[i].y - bbox_height / 2,
                            bbox_width,
                            bbox_height);
        left_box = left_box & cv::Rect2d(0, 0, _stereo_calib->width(), _stereo_calib->height());
        radar_boxes.push_back(left_box);
    }

    std::vector<cv::Rect2d> camera_detections_distort;
    distortCameraDetections(camera_detections, camera_detections_distort);

    cv::Mat img_draw_associate;
    img_l.copyTo(img_draw_associate);
    for (const auto& box : camera_detections_distort) {
        cv::rectangle(img_draw_associate, box, cv::Scalar(0, 0, 128));
    }
    for (size_t i = 0; i < radar_boxes.size(); ++i) {
        cv::rectangle(img_draw_associate, radar_boxes[i], cv::Scalar(0, 128, 0));
        std::string depth_str = std::to_string(radar_pts_cam[i].z);
        cv::putText(img_draw_associate,
                    depth_str,
                    radar_boxes[i].tl(),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    1);
    }

    // Jingsen TODO: add these parameters to configuration
    bool param_match_by_associate = false;
    double param_match_roi_ratio = 1.5;
    double param_match_roi_tol_angle = 3.0;
    double tol_pixel = _stereo_calib->M1().at<double>(0, 0) * 3.14159 / 180.0;

    std::vector<cv::Rect2d> draw_boxes, draw_masks, draw_matches;
    for (size_t i = 0; i < radar_boxes.size(); i++) {
        int idx = _radar_image_associate->associate(camera_detections_distort, radar_boxes[i]);
        if (idx != -1) {
            boxes.push_back(camera_detections_distort[idx]);
            depth.push_back(radar_pts_cam[i].z);

            cv::rectangle(img_draw_associate, radar_boxes[i], cv::Scalar(0, 255, 0));
            cv::rectangle(
                    img_draw_associate, camera_detections_distort[idx], cv::Scalar(0, 0, 255));
            cv::line(img_draw_associate,
                     cv::Point(radar_boxes[i].x, radar_boxes[i].y),
                     cv::Point(camera_detections_distort[idx].x, camera_detections_distort[idx].y),
                     cv::Scalar(0, 255, 0),
                     2);
        }

        cv::Rect2d left_box = radar_boxes[i];
        if (param_match_by_associate) {
            if (idx == -1) {
                continue;
            } else {
                left_box = camera_detections_distort[idx];
            }
        }
        int right_box_width = left_box.width * param_match_roi_ratio + 2 * tol_pixel;
        int right_box_height = left_box.height * param_match_roi_ratio + 2 * tol_pixel;
        cv::Rect2d right_box_init(radar_pts_cam2d_right[i].x - right_box_width / 2,
                                  radar_pts_cam2d_right[i].y - right_box_height / 2,
                                  right_box_width,
                                  right_box_height);
        cv::Rect2d right_box;
        if (_radar_box_matcher->match(img_l, img_r, left_box, right_box_init, right_box)) {
            _radar_points_left.push_back(
                    cv::Point2d(left_box.x + left_box.width / 2, left_box.y + left_box.height / 2));
            _radar_points_right.push_back(cv::Point2d(right_box.x + right_box.width / 2,
                                                      right_box.y + right_box.height / 2));
            _radar_points_depth.push_back(radar_pts_cam[i].z);

            draw_boxes.push_back(left_box);
            draw_masks.push_back(right_box_init);
            draw_matches.push_back(right_box);
        } else {
            continue;
        }
    }
    cv::Mat img_draw_match;
    _radar_box_matcher->drawMatch(
            img_l, img_r, img_draw_match, draw_boxes, draw_masks, draw_matches);

    cv::imshow("radar match", img_draw_match);
    cv::imshow("associate", img_draw_associate);
}

bool RadarBasedStereoCalibChecker::addFrame(
        const double stereo_timestamp,
        const cv::Mat front_left,
        const cv::Mat front_right,
        const double radar_timestamp,
        radar_msgs::RadarDetectionArray::ConstPtr radar_detections,
        const std::vector<cv::Rect2d>& camera_detections) {
    std::vector<cv::Rect2d> boxes;
    std::vector<double> depth;
    if (_radar_stereo_param.optimize_distance()) {
        radarProjAssociate(
                front_left, front_right, radar_detections, camera_detections, boxes, depth);
    }

    _feature_tracker->stereoCallback(front_left, front_right, boxes, depth);
    if (_feature_tracker->isMultiFrameFeatureReady()) {
        std::shared_ptr<GridFeatures> grid_features;
        _feature_tracker->getMultiFrameFeatures(grid_features);

        std::vector<cv::Point2f> keypts_l, keypts_r;
        std::vector<cv::Point2f> undist_keypts_l, undist_keypts_r;
        for (const auto& grid : *grid_features) {
            for (const auto& feat : grid.second) {
                keypts_l.emplace_back(feat.cam0_point);
                keypts_r.emplace_back(feat.cam1_point);
            }
        }
        cv::undistortPoints(keypts_l,
                            undist_keypts_l,
                            _stereo_calib->M1(),
                            _stereo_calib->D1(),
                            cv::noArray(),
                            _stereo_calib->M1());
        cv::undistortPoints(keypts_r,
                            undist_keypts_r,
                            _stereo_calib->M2(),
                            _stereo_calib->D2(),
                            cv::noArray(),
                            _stereo_calib->M2());

        if (_feature_tracker->isMultiFrameDepthFeatureReady()) {
            std::vector<FeatureMetaData> depth_features;
            _feature_tracker->getMultiFrameDepthFeature(depth_features);
            std::vector<cv::Point2f> depth_keypts_l, depth_keypts_r;
            std::vector<cv::Point2f> undist_depth_keypts_l, undist_depth_keypts_r;
            for (const auto& feat : depth_features) {
                depth_keypts_l.emplace_back(feat.cam0_point);
                depth_keypts_r.emplace_back(feat.cam1_point);
            }
            cv::undistortPoints(depth_keypts_l,
                                undist_depth_keypts_l,
                                _stereo_calib->M1(),
                                _stereo_calib->D1(),
                                cv::noArray(),
                                _stereo_calib->M1());
            cv::undistortPoints(depth_keypts_r,
                                undist_depth_keypts_r,
                                _stereo_calib->M2(),
                                _stereo_calib->D2(),
                                cv::noArray(),
                                _stereo_calib->M2());

            for (size_t i = 0; i < undist_depth_keypts_l.size(); ++i) {
                if (_undist_depth_keypts_l.size() >= _max_depth_feature_num) {
                    _undist_depth_keypts_l[_depth_feature_index] = undist_depth_keypts_l[i];
                    _undist_depth_keypts_r[_depth_feature_index] = undist_depth_keypts_r[i];
                    _depth[_depth_feature_index] = depth_features[i].depth;
                    _depth_feature_index = (_depth_feature_index + 1) % _max_depth_feature_num;
                } else {
                    _undist_depth_keypts_l.emplace_back(undist_depth_keypts_l[i]);
                    _undist_depth_keypts_r.emplace_back(undist_depth_keypts_r[i]);
                    _depth.emplace_back(depth_features[i].depth);
                }
            }
        }

        _is_calib_success = optimize(undist_keypts_l, undist_keypts_r);
        optimizeDistance();

        printResult();

        checkResult(front_left, front_right, undist_keypts_l, undist_keypts_r, _calib_R, _calib_T);
        return _is_calib_success;
    }

    return false;
}

bool RadarBasedStereoCalibChecker::optimize(const std::vector<cv::Point2f>& pts_l,
                                            const std::vector<cv::Point2f>& pts_r) {
    if (pts_l.size() != pts_r.size()) {
        LOG(ERROR) << "pts_l must have same size with pts_r!";
        return false;
    }

    ceres::Problem problem;

    BasePtr base1(new Eigen::Vector3d(1.0, 0.0, 0.0));
    BasePtr base2(new Eigen::Vector3d(0.0, 1.0, 0.0));

    ceres::LocalParameterization* t_local_param =
            new TranslationLocalParameterization(base1, base2);
    problem.AddParameterBlock(_translation_param, 3, t_local_param);

    ceres::LocalParameterization* r_local_param = new QuaternionLocalParameterization;
    problem.AddParameterBlock(_rotation_param, 4, r_local_param);

    // problem.SetParameterBlockConstant(_translation_param);
    // problem.SetParameterLowerBound(_translation_param, 0, _init_T.x() - _trans_limit);
    // problem.SetParameterUpperBound(_translation_param, 0, _init_T.x() + _trans_limit);
    // problem.SetParameterLowerBound(_translation_param, 1, _init_T.y() - _trans_limit);
    // problem.SetParameterUpperBound(_translation_param, 1, _init_T.y() + _trans_limit);
    // problem.SetParameterLowerBound(_translation_param, 2, _init_T.z() - _trans_limit);
    // problem.SetParameterUpperBound(_translation_param, 2, _init_T.z() + _trans_limit);

    std::vector<ceres::ResidualBlockId> epipolar_res_blocks;
    std::vector<ceres::ResidualBlockId> depth_res_blocks;
    ceres::LossFunction* loss = new ceres::HuberLoss(3.0 * 3.0);

    for (size_t i = 0; i < pts_l.size(); ++i) {
        EpipolarError* epipolar_factor =
                new EpipolarError(pts_l[i], pts_r[i], _inv_Kl, _inv_Kr, base1, base2);
        ceres::ResidualBlockId res_id = problem.AddResidualBlock(
                epipolar_factor, loss, _translation_param, _rotation_param);
        epipolar_res_blocks.emplace_back(res_id);
    }

    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::DENSE_QR;  // ceres::DENSE_SCHUR;
    solver_options.trust_region_strategy_type = ceres::DOGLEG;
    solver_options.minimizer_progress_to_stdout = true;
    // solver_options.max_num_iterations = 5;
    // solver_options.num_threads = 2;
    // solver_options.max_solver_time_in_seconds = 0.03;

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    LOG(INFO) << summary.BriefReport();

    std::set<ceres::ResidualBlockId> outlier_ids;
    findOutliers(problem, epipolar_res_blocks, depth_res_blocks, outlier_ids);

    for (const auto& id : outlier_ids) {
        problem.RemoveResidualBlock(id);
    }

    bool res = false;
    if (_undist_depth_keypts_l.size() >= _max_depth_feature_num) {
        res = true;
        for (size_t i = 0; i < _undist_depth_keypts_l.size(); ++i) {
            double* param = &_depth[i];
            problem.AddParameterBlock(param, 1);
            problem.SetParameterBlockConstant(param);
            // problem.SetParameterLowerBound(param, 0, 50);
            // problem.SetParameterUpperBound(param, 0, 150);

            StereoReprojectionFactor* reproj_factor =
                    new StereoReprojectionFactor(_undist_depth_keypts_l[i],
                                                 _undist_depth_keypts_r[i],
                                                 _inv_Kl,
                                                 _inv_Kr,
                                                 base1,
                                                 base2);
            reproj_factor->setSqrtInfo(_init_R, _init_T, *param, 3.0);
            ceres::ResidualBlockId res_id = problem.AddResidualBlock(
                    reproj_factor, loss, _translation_param, _rotation_param, param);
            depth_res_blocks.emplace_back(res_id);
        }
    }

    ceres::Solve(solver_options, &problem, &summary);
    LOG(INFO) << summary.BriefReport();

    _calib_T << _translation_param[0], _translation_param[1], _translation_param[2];
    _calib_R = Eigen::Map<Eigen::Quaterniond>(_rotation_param).toRotationMatrix();
    res = res && summary.termination_type == ceres::CONVERGENCE;

    return res;
}

void RadarBasedStereoCalibChecker::findOutliers(
        ceres::Problem& problem,
        const std::vector<ceres::ResidualBlockId>& epipolar_res_blocks,
        const std::vector<ceres::ResidualBlockId>& depth_res_blocks,
        std::set<ceres::ResidualBlockId>& outlier_id) {
    outlier_id.clear();

    ceres::Problem::EvaluateOptions eva_options;
    double total_cost = 0.0;
    double thresh = 0.0;
    size_t outlier_num = 0;
    std::vector<double> residuals;

    if (epipolar_res_blocks.size() > 0) {
        eva_options.residual_blocks = epipolar_res_blocks;
        problem.Evaluate(eva_options, &total_cost, &residuals, nullptr, nullptr);
        std::vector<double> residuals_tmp = residuals;
        for (auto& res : residuals_tmp) {
            res = std::fabs(res);
        }
        std::sort(residuals_tmp.begin(), residuals_tmp.end());
        thresh = residuals_tmp[epipolar_res_blocks.size() * 0.8];
        for (size_t i = 0; i < eva_options.residual_blocks.size(); ++i) {
            double res = std::fabs(residuals[i]);
            if (res > thresh) {
                const ceres::ResidualBlockId& res_id = eva_options.residual_blocks.at(i);
                outlier_id.insert(res_id);
            }
        }
        outlier_num = outlier_id.size();
        LOG(ERROR) << "total_epipolar_cost: " << total_cost << "; outlier_thresh: " << thresh
                   << "\nepipolar_residual_blocks_num: " << eva_options.residual_blocks.size()
                   << "; outlier_num: " << outlier_num;
    }

    if (depth_res_blocks.size() > 0) {
        total_cost = 0;
        residuals.clear();
        eva_options.residual_blocks = depth_res_blocks;
        problem.Evaluate(eva_options, &total_cost, &residuals, nullptr, nullptr);
        thresh = std::sqrt(total_cost / residuals.size()) * 2.0;
        for (size_t i = 0; i < eva_options.residual_blocks.size(); ++i) {
            double res = std::fabs(residuals[i]);
            if (res > thresh) {
                const ceres::ResidualBlockId& res_id = eva_options.residual_blocks.at(i);
                outlier_id.insert(res_id);
            }
        }
        outlier_num = outlier_id.size() - outlier_num;
        LOG(ERROR) << "total_depth_cost: " << total_cost << "; outlier_thresh: " << thresh
                   << "\ndepth_residual_blocks_num: " << eva_options.residual_blocks.size()
                   << "; outlier_num: " << outlier_num;
    }

    return;
}

// Jingsen TODO: _stereo_calib in this function should be based on the on after rectification
// optimization instead of original stereo calibration.
bool RadarBasedStereoCalibChecker::optimizeDistance() {
    // Jingsen TODO: move
    int minimal_optimize_distance_samples = 500;
    if (int(_radar_points_left.size()) < minimal_optimize_distance_samples) {
        return false;
    }

    std::vector<cv::Point2d> _radar_points_left_undistort, _radar_points_right_undistort;
    cv::undistortPoints(_radar_points_left,
                        _radar_points_left_undistort,
                        _stereo_calib->M1(),
                        _stereo_calib->D1(),
                        _stereo_calib->R1(),
                        _stereo_calib->P1());
    cv::undistortPoints(_radar_points_right,
                        _radar_points_right_undistort,
                        _stereo_calib->M2(),
                        _stereo_calib->D2(),
                        _stereo_calib->R2(),
                        _stereo_calib->P2());

    // Jingsen TODO: d_theta = - b * d_z / (z * z), MLS solution is easier to be affected by outlier
    // to get more robust result, refer to the method in highway_check radar_stereo in tools
    cv::Mat stereo_cloud_cv;
    cv::triangulatePoints(_stereo_calib->P1(),
                          _stereo_calib->P2(),
                          _radar_points_left_undistort,
                          _radar_points_right_undistort,
                          stereo_cloud_cv);
    double dyaw = 0;
    for (int i = 0; i < stereo_cloud_cv.cols; i++) {
        double w = stereo_cloud_cv.at<double>(3, i);
        double sz = stereo_cloud_cv.at<double>(2, i) / w;
        double rz = _radar_points_depth[i];

        dyaw += (sz - rz) / _stereo_calib->Q().at<double>(3, 2) / (rz * rz);
    }
    dyaw /= _radar_points_left.size();
    _radar_points_left.clear();
    _radar_points_right.clear();
    _radar_points_depth.clear();

    // Jingsen TODO: add this dyaw to new calibration

    return true;
}

void RadarBasedStereoCalibChecker::checkResult(const cv::Mat img_l,
                                               const cv::Mat img_r,
                                               const std::vector<cv::Point2f>& pts_l,
                                               const std::vector<cv::Point2f>& pts_r,
                                               Eigen::Matrix3d calib_R,
                                               Eigen::Vector3d calib_T) {
    Eigen::Matrix3d M1;
    Eigen::Matrix3d M2;
    cv::cv2eigen(_stereo_calib->M1(), M1);
    cv::cv2eigen(_stereo_calib->M2(), M2);

    for (size_t i = 0; i < _undist_depth_keypts_l.size(); ++i) {
        cv::circle(img_l, _undist_depth_keypts_l[i], 1, cv::Scalar(0, 255, 0), -1);
        cv::circle(img_r, _undist_depth_keypts_r[i], 1, cv::Scalar(0, 255, 0), -1);

        Eigen::Vector3d pt_caml =
                M1.inverse() *
                Eigen::Vector3d(_undist_depth_keypts_l[i].x, _undist_depth_keypts_l[i].y, 1.0) *
                _depth[i];
        // draw optimized projected points
        Eigen::Vector3d proj_pt = calib_R * pt_caml + calib_T;
        proj_pt = M2 * proj_pt / proj_pt.z();
        cv::circle(img_r, cv::Point2d(proj_pt.x(), proj_pt.y()), 1, cv::Scalar(0, 255, 255), -1);

        // draw original projected points
        proj_pt = _init_R * pt_caml + _init_T;
        proj_pt = M2 * proj_pt / proj_pt.z();
        cv::circle(img_r, cv::Point2d(proj_pt.x(), proj_pt.y()), 1, cv::Scalar(0, 0, 255), -1);
    }

    cv::Mat R, T;
    cv::eigen2cv(calib_R, R);
    cv::eigen2cv(calib_T, T);
    cv::Rect validRoi[2];
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(_stereo_calib->M1(),
                      _stereo_calib->D1(),
                      _stereo_calib->M2(),
                      _stereo_calib->D2(),
                      cv::Size(_stereo_calib->width(), _stereo_calib->height()),
                      R,
                      T,
                      R1,
                      R2,
                      P1,
                      P2,
                      Q,
                      cv::CALIB_ZERO_DISPARITY,
                      0,
                      cv::Size(_stereo_calib->width(), _stereo_calib->height()),
                      &validRoi[0],
                      &validRoi[1]);
    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(
            _stereo_calib->M1(), _stereo_calib->D1(), R1, P1, img_l.size(), CV_16SC2, map1x, map1y);
    cv::initUndistortRectifyMap(
            _stereo_calib->M2(), _stereo_calib->D2(), R2, P2, img_r.size(), CV_16SC2, map2x, map2y);
    cv::Mat img_lc, img_rc;
    cv::remap(img_l, img_lc, map1x, map1y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::remap(img_r, img_rc, map2x, map2y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::Mat img_concat;
    cv::hconcat(img_lc, img_rc, img_concat);
    int line = 50;
    while (line < img_concat.rows) {
        cv::line(img_concat,
                 cv::Point(0, line),
                 cv::Point(img_concat.cols - 1, line),
                 cv::Scalar(0, 255, 0),
                 1);
        line += 50;
    }
    cv::imshow("optimize result", img_concat);
    cv::waitKey(1);
}

void RadarBasedStereoCalibChecker::printResult() {
    Eigen::Quaterniond quat(_calib_R);
    Eigen::Vector3d euler;
    Quat2Euler(quat, euler);
    euler *= 180 / M_PI;

    Eigen::Quaterniond quat_gt(_init_R.cast<double>());
    Eigen::Vector3d euler_ori;
    Quat2Euler(quat_gt, euler_ori);
    euler_ori *= 180 / M_PI;

    LOG(ERROR) << std::fixed << std::setprecision(5) << "calib checker result: \n"
               << "euler_ori(pitch-yaw-roll): " << euler_ori.transpose()
               << ", trans_ori(x-y-z): " << _init_T.transpose() << "\n"
               << "euler_opt(pitch-yaw-roll): " << euler.transpose()
               << ", trans_opt(x-y-z): " << _calib_T.transpose() << "\n"
               << "euler_err(pitch-yaw-roll): " << (euler - euler_ori).transpose()
               << ", trans_err(x-y-z): " << (_calib_T - _init_T.cast<double>()).transpose() << "\n";
}

}  // namespace calib_checker
}  // namespace perception
}  // namespace drive