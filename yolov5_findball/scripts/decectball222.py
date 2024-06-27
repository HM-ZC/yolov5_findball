                                                                                                                                                                                                                                                                                                                                             #!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import cv2

import rospy
import numpy as np
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov5_findball_msgs.msg import BoundingBox
from yolov5_findball_msgs.msg import BoundingBoxes
from geometry_msgs.msg import Point
class KalmanTracker:
    count = 0  # 全局计数器，用于分配ID
    
    def __init__(self, bbox):
        # 初始化卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],  # 状态转移矩阵
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],  # 观测矩阵
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.  # 观测噪声协方差矩阵
        self.kf.P[4:,4:] *= 1000.  # 初始状态协方差矩阵
        self.kf.P *= 10.  # 初始状态协方差矩阵
        self.kf.Q[-1,-1] *= 0.01  # 过程噪声协方差矩阵
        self.kf.Q[4:,4:] *= 0.01  # 过程噪声协方差矩阵
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)  # 初始状态
        self.id = KalmanTracker.count  # 分配ID
        KalmanTracker.count += 1
        self.hits = 1  # 命中次数
        self.no_losses = 0  # 未匹配次数
        self.bbox = bbox  # 当前边界框

    def update(self, bbox):
        self.hits += 1  # 增加命中次数
        self.no_losses = 0  # 重置未匹配次数
        self.kf.update(self.convert_bbox_to_z(bbox))  # 更新卡尔曼滤波器状态
        self.bbox = bbox  # 更新当前边界框

    def predict(self):
        self.kf.predict()  # 预测下一状态
        return self.convert_x_to_bbox(self.kf.x)  # 返回预测边界框

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)  # 返回当前状态

    def convert_bbox_to_z(self, bbox):
        # 将边界框转换为卡尔曼滤波器的观测向量
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x):
        # 将卡尔曼滤波器的状态向量转换为边界框
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))

class Yolo_Dect:
    def __init__(self):
        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '')
        weight_path = rospy.get_param('~weight_path', '')
        pub_topic = rospy.get_param('~pub_topic', '/camera1/ball_position')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf1 = rospy.get_param('~conf', '0.5')
        conf2 = rospy.get_param('~conf', '0.8')
        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')
        
        # Load wood frame model
        wood_frame_model_path = rospy.get_param('~wood_frame_model_path', '')
        self.wood_frame_model = torch.hub.load(yolov5_path, 'custom', path=wood_frame_model_path, source='local')

        # which device will be used
        if rospy.get_param('/use_cpu', 'false'):
            self.model.cpu()
            self.wood_frame_model.cpu()
        else:
            self.model.cuda()
            self.wood_frame_model.cuda()

        self.model.conf = conf1
        self.wood_frame_model.conf = conf2

        self.color_image = Image()
        self.img = Image()
        self.depth_image = Image()
        self.getImageStatus = False
        self.frame_counter = 0
        self.check_interval = 300
        self.frame_interval_when_detected = 1
        # Load class color
        self.classes_colors = {}
        # output publishers
        self.position_pub = rospy.Publisher(pub_topic, Point, queue_size=1)
        self.bounding_boxes_pub = rospy.Publisher('/yolov5/detected_boxes', BoundingBoxes, queue_size=1)

        # image subscribe
        self.color_sub = rospy.Subscriber("/camera1/undistorted_image", Image, self.image_callback, queue_size=1, buff_size=52428800)

        self.locked_ball = None  # 用于存储锁定的球的信息
        self.detected_wood_frame = None  # 用于存储检测到的木框信息
        self.trackers = []  # 跟踪器列表
        self.max_age = 1  # 最大未匹配次数
        self.min_hits = 3  # 最小命中次数
        # if no image messages
        while not self.getImageStatus:
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def image_callback(self, image):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(self.color_image, (320, 160))
        '''  
        if self.detected_wood_frame is not None:
            wood_frame_results = self.wood_frame_model(resized_image)
            wood_frames = wood_frame_results.pandas().xyxy[0].values
            self.detected_wood_frame = self.detect_wood_frame(wood_frames)
            self.frame_counter = self.frame_interval_when_detected
        elif self.frame_counter % self.check_interval == 0:
            wood_frame_results = self.wood_frame_model(resized_image)
            wood_frames = wood_frame_results.pandas().xyxy[0].values
            self.detected_wood_frame = self.detect_wood_frame(wood_frames)
            self.frame_counter = 0
            '''
        results = self.model(self.color_image)

        boxs = results.pandas().xyxy[0].values


        self.dectshow(self.color_image, boxs, image.height, image.width)
        cv2.waitKey(3)

    def detect_wood_frame(self, wood_frames):
        for frame in wood_frames:
            if frame[-1] == "box":
                return frame
        return None

    def dectshow(self, org_img, boxs, height, width):
        img = org_img.copy()
        count = 0
        for i in boxs:
            count += 1

        for box in boxs:
            boundingBox = BoundingBox()
            boundingBox.probability = np.float64(box[4])
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.num = np.int16(count)
            boundingBox.Class = box[-1]

            if box[-1] in self.classes_colors.keys():
                color = self.classes_colors[box[-1]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[-1]] = color

            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (int(color[0]), int(color[1]), int(color[2])), 2)

            if box[1] < 20:
                text_pos_y = box[1] + 30
            else:
                text_pos_y = box[1] - 10

            cv2.putText(img, box[-1], (int(box[0]), int(text_pos_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            self.boundingBoxes.bounding_boxes.append(boundingBox)
            self.bounding_boxes_pub.publish(self.boundingBoxes)

        # 绘制木框
        if self.detected_wood_frame is not None:
            x1, y1, x2, y2 = int(self.detected_wood_frame[0]), int(self.detected_wood_frame[1]), int(self.detected_wood_frame[2]), int(self.detected_wood_frame[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "wood_frame", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        self.track_balls(boxs, img)
        cv2.imshow('YOLOv5', img)

    def track_balls(self, boxs, img):
        detections = [box[:4] for box in boxs if box[-1] == "purple"]  # 获取紫色球的边界框

        if len(self.trackers) == 0:  # 如果没有跟踪器
            for i in range(len(detections)):
                tracker = KalmanTracker(detections[i])  # 为每个检测到的对象创建一个新的跟踪器
                self.trackers.append(tracker)

        trks = np.zeros((len(self.trackers), 4))  # 初始化跟踪器边界框数组
        to_del = []  # 要删除的跟踪器索引列表
        ret = []  # 返回的跟踪结果

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]  # 预测跟踪器的下一状态
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]  # 更新跟踪器边界框
            if np.any(np.isnan(pos)):  # 如果预测位置包含NaN
                to_del.append(t)  # 添加到删除列表中

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # 删除包含NaN的行

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, trks)  # 数据关联

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:  # 如果跟踪器有匹配的检测
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(detections[d, :][0])  # 更新跟踪器状态
            else:
                trk.no_losses += 1  # 增加未匹配次数
                if trk.no_losses > self.max_age:  # 如果未匹配次数超过阈值
                    to_del.append(t)  # 添加到删除列表中

        for i in reversed(to_del):
            self.trackers.pop(i)  # 删除超时未匹配的跟踪器

        for i in unmatched_dets:  # 为未匹配的检测创建新的跟踪器
            tracker = KalmanTracker(detections[i])
            self.trackers.append(tracker)

        for trk in self.trackers:
            d = trk.get_state()[0]  # 获取跟踪器状态
            ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))  # 添加跟踪结果
            x = (d[0] + d[2]) / 2
            y = (d[1] + d[3]) / 2
            ballcenter = (int(x), int(y))

            r = self.get_radius(d)  # 计算半径

            cv2.circle(img, ballcenter, int(r), (255, 255, 255), 2)  # 画圆
            cv2.rectangle(img, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255, 255, 255), 2)  # 画矩形框
            cv2.putText(img, f"ID: {trk.id}", (int(d[0]), int(d[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)  # 显示ID

            msg = Point()  # 初始化Point消息
            msg.y = r
            msg.x = x - 332
            msg.z = 0
            self.position_pub.publish(msg)  # 发布位置信息
            print(msg.x, msg.y)

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:  # 如果没有跟踪器
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 2), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)  # 初始化IOU矩阵

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det, trk)  # 计算IOU

        matched_indices = linear_sum_assignment(-iou_matrix)  # 匈牙利算法求解最优匹配

        matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))

        unmatched_detections = []  # 未匹配的检测
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []  # 未匹配的跟踪器
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []  # 最终匹配结果
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < 0.3:  # 过滤低IOU匹配
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m)

        return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)

    def iou(self, bb_test, bb_gt):
        # 计算两个边界框的IOU
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o
    def get_radius(self, ball):
        a = ball[2] - ball[0]
        b = ball[3] - ball[1]
        if a / b < 2.0 and b / a < 2.0:
            return (a + b) / 4
        elif a / b > 2.0:
            return a / 2
        elif b / a > 2.0:
            return b / 2

    def is_ball_in_wood_frame(self, ball, wood_frame):
        if wood_frame is None:
            return False
        # 判断球的下部是否在木框内
        ball_bottom_y = ball[1]
        wood_frame_top_y = wood_frame[1]
        wood_frame_bottom_y = wood_frame[3]
        wood_frame_left_x = wood_frame[0]
        wood_frame_right_x = wood_frame[2]
        
        ball_center_x = (ball[0] + ball[2]) / 2

        return wood_frame_left_x < ball_center_x < wood_frame_right_x and wood_frame_top_y < ball_bottom_y

def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()

if __name__ == "__main__":
    main()
