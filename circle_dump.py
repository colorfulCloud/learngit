#!/usr/bin/env python

import sys
import rosbag
import numpy as np
import sensor_msgs.point_cloud2
import argparse
import cv2
from cv_bridge import CvBridge
import os
import shutil

bridge = CvBridge()
def msg_time(msg):
    #return msg.timestamp
    return msg.header.stamp
    
def msg_to_velo_file(topic, msg, path):
    velo = np.array([p for p in sensor_msgs.point_cloud2.read_points(msg)])
    velo = velo.astype(np.float32)
    header = """VERSION 0.7
    FIELDS x y z intensity
    SIZE 4 4 4 4
    TYPE F F F Fc
    COUNT 1 1 1 1
    WIDTH %d
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS %d
    DATA ascii
    """ % (velo.shape[0], velo.shape[0])

    with open(path, "w") as o:
        o.write(header)
        intensity_column = 3     #for hesai and velodyne, intensity column = 3
        if "inno" in topic:      #for innovation, intensity column = 4
            intensity_column = 4

        for p in velo:
            o.write("%f %f %f %e\n" % (p[0], p[1], p[2], p[intensity_column]))

def rotate_img(img, rotate):
    out = img
    if rotate == 1:
        out = cv2.transpose(out)
        out = cv2.flip(out, flipCode = 1)
    if rotate == -1:
        out = cv2.transpose(out)
        out = cv2.flip(out, flipCode = 0)
    return out

def msg_to_png_file(msg, path, resize_image_flag, width, height, encoding, rotate):
    img = bridge.imgmsg_to_cv2(msg, encoding)
    if resize_image_flag and img.shape[0:2] != (height, width):
        print("Image Size: " + str(img.shape[0]) + ", " + str(img.shape[1]) + " -> " + str(height) + ", " + str(width))
        img = cv2.resize(img, (width, height))

    cv2.imwrite(path, rotate_img(img, rotate))

def msg_compressed_to_png_file(msg, path, resize_image_flag, width, height, encoding, rotate):
    img = bridge.compressed_imgmsg_to_cv2(msg, encoding)
    if resize_image_flag and img.shape[0:2] != (height, width):
        print("Image Size: " + str(img.shape[0]) + ", " + str(img.shape[1]) + " -> " + str(height) + ", " + str(width))
        img = cv2.resize(img, (width, height))

    cv2.imwrite(path, rotate_img(img, rotate))

def odom_msg_to_str(msg):
    msg_str = "header:\n"
    msg_str += "\tseq: %d\n" % msg.header.seq
    msg_str += "\tstamp: %f\n" % msg.header.stamp.to_sec()
    msg_str += "\tframe_id: %s\n" % msg.header.frame_id
    msg_str += "child_frame_id: %s\n" % msg.child_frame_id
    # pose
    msg_str += "pose:\n"
    msg_str += "\tpose:\n"
    msg_str += "\t\tposition:\n"
    msg_str += "\t\t\tx: %f\n" % msg.pose.pose.position.x
    msg_str += "\t\t\ty: %f\n" % msg.pose.pose.position.y
    msg_str += "\t\t\tz: %f\n" % msg.pose.pose.position.z
    msg_str += "\t\torientation:\n"
    msg_str += "\t\t\tx: %f\n" % msg.pose.pose.orientation.x
    msg_str += "\t\t\ty: %f\n" % msg.pose.pose.orientation.y
    msg_str += "\t\t\tz: %f\n" % msg.pose.pose.orientation.z
    msg_str += "\t\t\tw: %f\n" % msg.pose.pose.orientation.w
    msg_str += "\tcovariance[]\n"
    for i in range(len(msg.pose.covariance)):
        msg_str += "\t\tcovariance[%d]: %f\n" % (i, msg.pose.covariance[i])
    # twist
    msg_str += "twist:\n"
    msg_str += "\ttwist:\n"
    msg_str += "\t\tlinear:\n"
    msg_str += "\t\t\tx: %f\n" % msg.twist.twist.linear.x
    msg_str += "\t\t\ty: %f\n" % msg.twist.twist.linear.y
    msg_str += "\t\t\tz: %f\n" % msg.twist.twist.linear.z
    msg_str += "\t\tangular:\n"
    msg_str += "\t\t\tx: %f\n" % msg.twist.twist.angular.x
    msg_str += "\t\t\ty: %f\n" % msg.twist.twist.angular.y
    msg_str += "\t\t\tz: %f\n" % msg.twist.twist.angular.z
    msg_str += "\tcovariance[]\n"
    for i in range(len(msg.twist.covariance)):
        msg_str += "\t\tcovariance[%d]: %f\n" % (i, msg.twist.covariance[i])
    return msg_str

def msg_to_odom_file(msg, path):
    with open(path, 'w') as f:
        odom_str = odom_msg_to_str(msg)
        f.write(odom_str)

def radar_msg_to_str(msg):
    msg_str = "header:\n"
    msg_str += "\tseq: %d\n" % msg.header.seq
    msg_str += "\tstamp: %f\n" % msg.header.stamp.to_sec()
    msg_str += "\tframe_id: %s\n" % msg.header.frame_id
    msg_str += "tracks[]\n"
    for i in range(len(msg.tracks)):
        msg_str += "\ttracks[%d]:\n" % i
        msg_str += "\t\ttrack_id: %d\n" % msg.tracks[i].track_id
        msg_str += "\t\ttrack_shape:\n"
        msg_str += "\t\t\tpoints[]\n"
        for j in range(len(msg.tracks[i].track_shape.points)):
            msg_str += "\t\t\t\tpoints[%d]:\n" % j
            msg_str += "\t\t\t\t\tx: %f\n" % msg.tracks[i].track_shape.points[j].x
            msg_str += "\t\t\t\t\ty: %f\n" % msg.tracks[i].track_shape.points[j].y
            msg_str += "\t\t\t\t\tz: %f\n" % msg.tracks[i].track_shape.points[j].z
        msg_str += "\t\tlinear_velocity:\n"
        msg_str += "\t\t\tx: %f\n" % msg.tracks[i].linear_velocity.x
        msg_str += "\t\t\ty: %f\n" % msg.tracks[i].linear_velocity.y
        msg_str += "\t\t\tz: %f\n" % msg.tracks[i].linear_velocity.z

        msg_str += "\t\tlinear_acceleration:\n"
        msg_str += "\t\t\tx: %f\n" % msg.tracks[i].linear_acceleration.x
        msg_str += "\t\t\ty: %f\n" % msg.tracks[i].linear_acceleration.y
        msg_str += "\t\t\tz: %f\n" % msg.tracks[i].linear_acceleration.z 

    return msg_str

def msg_to_radar_file(msg, path):
    with open(path, 'w') as f:
        msg_str = radar_msg_to_str(msg)
        f.write(msg_str)

def buffered_message_generator(bag, tolerance, topics):
    buffers = dict([(t, []) for t in topics])
    skipcounts = dict([(t, 0) for t in topics])
    for msg in bag.read_messages(topics=topics):
        if msg.topic in topics:
            buffers[msg.topic].append(msg)
        else:
            continue
        while all(buffers.values()):
            time_and_bufs = sorted([(msg_time(b[0].message).to_sec(), b) for b in buffers.values()])
            if time_and_bufs[-1][0] - time_and_bufs[0][0] > tolerance:
                old_msg = time_and_bufs[0][1].pop(0)
                skipcounts[old_msg.topic] += 1
                continue
            msg_set = {}
            for topic, buf in buffers.items():
                m = buf.pop(0).message
                msg_set[topic] = m
            yield msg_set
    for t, c in skipcounts.items():
        print("skipped %d %s messages" % (c, t))
    sys.stdout.flush()

def msg_loop(bag, output_dir, rate, tolerance, frame_limit, velo_topics, cam_topics, odom_topics, radar_topics, msg_it, resize_image_flag ,width, height, offset, last_only, rotate):

    start_time = None
    last_frame = None
    timestamp_file = None
    frame_number = None
    encoding = "bgr8"
    for m in msg_it:
        if start_time is None:
            start_time = msg_time(m[topics[0]])
        frame_number = int(((msg_time(m[topics[0]]) - start_time).to_sec() + (rate / 2.0)) / rate) + offset
        if last_frame == frame_number:
            continue
        sys.stdout.flush()
        
        if last_only:
            dump_number = 0
        else:
            dump_number = frame_number

        for topic in m.keys():
            file_name = ""
            if topic in velo_topics:
                file_name = "%s/%s/%04d.pcd" % (output_dir, topic.split('/')[1], dump_number)
                msg_to_velo_file(topic, m[topic], file_name)

            elif topic in cam_topics:
                created_dir = topic.split('/')[1]
                if created_dir == 'usb_cam_left':
                    created_dir = 'front_left_camera'
                if created_dir == 'usb_cam_right':
                    created_dir = 'front_right_camera'
                file_name = "%s/%s/%04d.png" % (output_dir, created_dir, dump_number)
                if 'compressed' in topic:
                    msg_compressed_to_png_file(m[topic], file_name, resize_image_flag ,width, height, encoding, rotate)
                else:
                    msg_to_png_file(m[topic], file_name, resize_image_flag ,width, height, encoding, rotate)
            
            elif topic in odom_topics:
                file_name = "%s/%s/%04d.txt" % (output_dir, topic.split('/')[1], dump_number)
                msg_to_odom_file(m[topic], file_name)
            
            elif topic in radar_topics:
                file_name = "%s/%s/%04d.txt" % (output_dir, topic.split('/')[1], dump_number)
                msg_to_radar_file(m[topic], file_name)

            if timestamp_file is None:
                timestamp_file = open("%s/timestamp.txt" % output_dir, 'a')
            
            timestamp_file.write("%s\t%.3f\n" % (file_name, m[topic].header.stamp.to_sec()))
                
        last_frame = frame_number
        if frame_limit > 0 and frame_number >= frame_limit - 1:
            print("reach frame limit: %d, quit" % (frame_limit))
            sys.stdout.flush()
            exit(0)
        print("dump frame: %d" % frame_number)

    return frame_number

def make_topic_dir(output_dir, topic):
    topic_keys = topic.split("/")
    key = next(s for s in topic_keys if s)
    if key == 'usb_cam_left':
        key = 'front_left_camera'
    if key == 'usb_cam_right':
        key = 'front_right_camera'
    topic_dir = os.path.join(output_dir, key)
    if not os.path.isdir(topic_dir):
        print("creating topic data dir: %s" % topic_dir)
        os.mkdir(topic_dir)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bags", type=str, help="path to bags (comma separated)")
    parser.add_argument("--out", type=str, help="output path (a directory)")
    parser.add_argument("--rate", type=float, help="desired sample rate in seconds", default=0.4)
    parser.add_argument("--tolerance", type=float, help="tolerance", default=0.05)
    parser.add_argument("--velo_topics", type=str, help="velodyne topic (comma separated, don't add space between topics)")
    parser.add_argument("--cam_topics", type=str, help="camera topics (comma separated, don't add space between topics)")
    parser.add_argument("--odom_topics", type=str, help="odometry topic (comma separated, don't add space between topics)")
    parser.add_argument("--radar_topics", type=str, help="radar topic (comma separated, don't add space between topics)")
    parser.add_argument("--frame_limit", type=int, help="frame limit if > 0", default=0)
    parser.add_argument("--resize_image", type=bool, help="resize the image size", default=False)
    parser.add_argument("--width", type=int, help="full resolution image width", default=2064)
    parser.add_argument("--height", type=int, help="full resolution image height", default=1544)
    parser.add_argument("--rotate", type=int, help="1: clockwise 90 degree, -1: counterclock 90 degree", default=0)
    parser.add_argument("--keep", type=bool, help="keep last dumped files", default=False)
    parser.add_argument("--last_only", type=bool, help="keep only last frame", default=False)

    args = parser.parse_args()
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    
    if not args.keep:
        for root, dirs, files in os.walk(args.out, topdown=False):
            for name in dirs:
                if name=="result":
                    continue
                print( "removing " + os.path.join(root, name))
                shutil.rmtree(os.path.join(root, name))

    result_dir = os.path.join(args.out, "result")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    topics = []
    velo_topics = []
    if args.velo_topics != None:
        for t in args.velo_topics.split(","):
            velo_topics.append(t)
            topics.append(t)
            make_topic_dir(args.out, t)
            
    cam_topics = []
    if args.cam_topics != None:
        for t in args.cam_topics.split(","):
            cam_topics.append(t)
            topics.append(t)
            make_topic_dir(args.out, t)

    odom_topics = []
    if args.odom_topics != None:
        for t in args.odom_topics.split(","):
            odom_topics.append(t)
            topics.append(t)
            make_topic_dir(args.out, t)

    radar_topics = []
    if args.radar_topics != None:
        for t in args.radar_topics.split(","):
            radar_topics.append(t)
            topics.append(t)
            make_topic_dir(args.out, t)

    offset = 0
    for b in args.bags.split(","):
        print("start dumping bag: " +b)
        sys.stdout.flush()
        bag = rosbag.Bag(b)
        msg_it = iter(buffered_message_generator(bag, args.tolerance, topics))
        offset = msg_loop(bag, args.out, args.rate, args.tolerance, args.frame_limit, velo_topics, cam_topics, odom_topics, radar_topics, msg_it, args.resize_image, args.width, args.height, offset, args.last_only, args.rotate)
