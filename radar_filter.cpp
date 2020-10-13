#include "highway_check/radar_filter.h"
#include <eigen3/Eigen/Dense>

radar_msgs::RadarTrackArray RadarFilter::filter(
        radar_msgs::RadarTrackArray& radar_tracks) {
    // filter by roi
    radar_msgs::RadarTrackArray radar_tracks_roi;
    std::vector<Eigen::Vector3d> track_centers_roi;
    std::vector<Eigen::Vector3d> track_sizes;
    for(size_t i=0; i<radar_tracks.tracks.size(); i++) {
        double min_x = 1e6, max_x = -1e6;   //1e6 as INF, -1e6 as -INF
        double min_y = 1e6, max_y = -1e6;
        double min_z = 1e6, max_z = -1e6;    

        if(_param.radar_fix_z > 0) {
            radar_tracks.tracks[i].track_shape.points[0].z = _param.radar_fix_z;
            radar_tracks.tracks[i].track_shape.points[1].z = _param.radar_fix_z;
            radar_tracks.tracks[i].track_shape.points[2].z = _param.radar_fix_z;
            radar_tracks.tracks[i].track_shape.points[3].z = _param.radar_fix_z;
        }

        if(_param.radar_fix_width > 0) {
            auto& pt0 = radar_tracks.tracks[i].track_shape.points[0];
            auto& pt1 = radar_tracks.tracks[i].track_shape.points[1];
            auto& pt2 = radar_tracks.tracks[i].track_shape.points[2];
            auto& pt3 = radar_tracks.tracks[i].track_shape.points[3];
            double y = (pt0.y + pt1.y + pt2.y + pt3.y) / 4;
            double z = (pt0.z + pt1.z + pt2.z + pt3.z) / 4;

            pt0.y = y - _param.radar_fix_width / 2;
            pt0.z = z - _param.radar_fix_width / 2;

            pt1.y = y - _param.radar_fix_width / 2;
            pt1.z = z + _param.radar_fix_width / 2;
            
            pt2.y = y + _param.radar_fix_width / 2;
            pt2.z = z + _param.radar_fix_width / 2;
            
            pt3.y = y + _param.radar_fix_width / 2;
            pt3.z = z - _param.radar_fix_width / 2;
        }

        for(int j=0; j<4; j++) {
            auto& pt = radar_tracks.tracks[i].track_shape.points[j];

            if(pt.x < min_x) {
                min_x = pt.x;
            }
            if(pt.x > max_x) {
                max_x = pt.x;
            }
            if(pt.y < min_y) {
                min_y = pt.y;
            }
            if(pt.y > max_y) {
                max_y = pt.y;
            }
            if(pt.z < min_z) {
                min_z = pt.z;
            }
            if(pt.z > max_z) {
                max_z = pt.z;
            }
        }

        Eigen::Vector3d track_velocity(radar_tracks.tracks[i].linear_velocity.x,
                                        radar_tracks.tracks[i].linear_velocity.y,
                                        radar_tracks.tracks[i].linear_velocity.z);

        if(max_x - min_x < _param.radar_max_xdiff
            && min_x > _param.radar_roi_min_x && max_x < _param.radar_roi_max_x
            && min_y > _param.radar_roi_min_y && max_y < _param.radar_roi_max_y) {
            radar_tracks_roi.tracks.push_back(radar_tracks.tracks[i]);
            track_centers_roi.push_back(Eigen::Vector3d(min_x + max_x, min_y + max_y, min_z + max_z) / 2);
            track_sizes.push_back(Eigen::Vector3d(max_x - min_x, max_y - min_y, max_z - min_z));
        }
    }

    // filter by shape, velocity
    std::vector<bool> filter_flag(track_centers_roi.size(), false);
    for(size_t i=0; i<track_centers_roi.size(); i++) {
        Eigen::Vector3d track_velocity(radar_tracks_roi.tracks[i].linear_velocity.x,
                                        radar_tracks_roi.tracks[i].linear_velocity.y,
                                        radar_tracks_roi.tracks[i].linear_velocity.z);

        if(track_velocity[0] < _param.radar_min_velocity || track_velocity[0] > _param.radar_max_velocity
            || track_sizes[i][1] < _param.radar_min_width || track_sizes[i][1] > _param.radar_max_width) {
                filter_flag[i] = true;
        }
    }

    // filter replicate trackers by space NMS 
    for(size_t i=0; i<track_centers_roi.size(); i++) {
        if(filter_flag[i]) {
            continue;
        }

        for(size_t j=0; j<track_centers_roi.size(); j++) {
            if(i != j) {
                double dist = (track_centers_roi[i] - track_centers_roi[j]).norm();
                if(dist < _param.radar_replicate_thresh) {
                    Eigen::Vector3d vec[2];
                    for(int k=0; k<3; k++) {
                        vec[0][k] = std::min(track_centers_roi[i][k] - track_sizes[i][k] / 2,
                                        track_centers_roi[j][k] - track_sizes[j][k] / 2);
                        vec[1][k] = std::max(track_centers_roi[i][k] + track_sizes[i][k] / 2,
                                        track_centers_roi[j][k] + track_sizes[j][k] / 2);
                    }
                    int pos[4][2] = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
                    for(int k=0; k<4; k++) {
                        radar_tracks_roi.tracks[i].track_shape.points[k].x = (vec[0][0] + vec[1][0]) / 2;
                        radar_tracks_roi.tracks[i].track_shape.points[k].y = vec[pos[k][0]][1];
                        radar_tracks_roi.tracks[i].track_shape.points[k].z = vec[pos[k][1]][2];
                    }

                    filter_flag[j] = true;
                }
            }
        }
    }
    radar_msgs::RadarTrackArray radar_tracks_rep;
    std::vector<Eigen::Vector3d> track_centers_rep;
    for(size_t i=0; i<radar_tracks_roi.tracks.size(); i++) {
        if(!filter_flag[i]) {
            radar_tracks_rep.tracks.push_back(radar_tracks_roi.tracks[i]);
            track_centers_rep.push_back(track_centers_roi[i]);
        }
    }

    // filter occuluded
    std::vector<bool> occluded_flag(track_centers_rep.size(), false);
    for(size_t i=0; i<track_centers_rep.size(); i++) {
        if(occluded_flag[i]) {
            continue;
        }
        for(size_t j=0; j<track_centers_rep.size(); j++) {
            if(i != j) {
                double dist = track_centers_rep[i].cross(track_centers_rep[j]).norm() /
                    std::min(track_centers_rep[i].norm(), track_centers_rep[j].norm());
                if(dist < _param.radar_occluded_thresh) {
                    if(track_centers_rep[i].norm() < track_centers_rep[j].norm()) {
                        occluded_flag[j] = true;
                    } else {
                        occluded_flag[i] = true;
                    }
                }
            }
        }
    }
    radar_msgs::RadarTrackArray radar_tracks_occlud;
    std::vector<Eigen::Vector3d> track_centers_occlud;
    for(size_t i=0; i<radar_tracks_rep.tracks.size(); i++) {
        if(!occluded_flag[i]) {
            radar_tracks_occlud.tracks.push_back(radar_tracks_rep.tracks[i]);
            track_centers_occlud.push_back(track_centers_rep[i]);
        }
    }

    return radar_tracks_occlud;
}