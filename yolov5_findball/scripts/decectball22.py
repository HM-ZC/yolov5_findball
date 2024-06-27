#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np

from std_msgs.msg import Header, Bool
from sensor_msgs.msg import Image
from yolov5_findball_msgs.msg import BoundingBox, BoundingBoxes
from geometry_msgs.msg import Point

import torch
model = torch.hub.load(r'/home/nuc/ws/src/yolov5_findball/yolov5', 'custom', path=r'/home/nuc/ws/src/yolov5_findball/weights/yolov5s.pt', source='local')


class Yolo_Dect:

    balls = []

    def __init__(self):
        yolov5_path = rospy.get_param('/yolov5_path', '')
        weight_path = rospy.get_param('~weight_path', '')
        pub_topic = rospy.get_param('~pub_topic', '/camera1/ball_position')

        self.camera_frame = rospy.get_param('~camera_frame', '')
        self.use_cpu = rospy.get_param('/use_cpu', 'false')
        self.conf = rospy.get_param('~conf', '0.5')

        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')

        if self.use_cpu:
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = self.conf
        self.getImageStatus = False

        self.classes_colors = {}
        self.position_pub = rospy.Publisher(pub_topic, Point, queue_size=1)
        self.color_sub1 = rospy.Subscriber("/camera/image_raw1", Image, self.image_callback1, queue_size=1, buff_size=52428800)
        self.color_sub2 = rospy.Subscriber("/camera/image_raw2", Image, self.image_callback2, queue_size=1, buff_size=52428800)
        self.detection_sub = rospy.Subscriber("/color_detection/purple_detected", Bool, self.detection_callback, queue_size=1)

        self.use_camera1 = True  # 初始使用第一个相机

        while not self.getImageStatus:
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def detection_callback(self, msg):
        self.use_camera1 = not msg.data  # 当接收到true时，使用第二个相机

    def image_callback1(self, image):
        if self.use_camera1:
            self.process_image(image)

    def image_callback2(self, image):
        if not self.use_camera1:
            self.process_image(image)

    def process_image(self, image):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(self.color_image)
        boxs = results.pandas().xyxy[0].values
        self.dectshow(self.color_image, boxs, image.height, image.width)
        cv2.waitKey(3)

    def dectshow(self, org_img, boxs, height, width):
        img = org_img.copy()
        count = 0
        for i in boxs:
            count += 1

        for box in boxs:
            if box[-1] == "blue":
                if (box[0] + box[2]) / 2 < 320 + 110 and (box[0] + box[2]) / 2 > 320 - 110:
                    self.balls.append(box)
                elif (box[0] + box[2]) / 2 < 220 and (box[0] + box[2]) / 2 > 0:
                    self.balls.append(box)
                elif (box[0] + box[2]) / 2 < 640 and (box[0] + box[2]) / 2 > 640 - 220:
                    self.balls.append(box)

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

            text_pos_y = box[1] + 30 if box[1] < 20 else box[1] - 10
            cv2.putText(img, box[-1], (int(box[0]), int(text_pos_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            self.boundingBoxes.bounding_boxes.append(boundingBox)

        if self.balls:
            biggest_ball = max(self.balls, key=lambda x: x[3])
            x = (biggest_ball[2] + biggest_ball[0]) / 2
            y = (biggest_ball[3] + biggest_ball[1]) / 2
            ballcenter = (int(x), int(y))
            r = self.get_radius(biggest_ball)
            cv2.circle(img, ballcenter, int(r), (255, 255, 255), 2)
            cv2.rectangle(img, (int(biggest_ball[0]), int(biggest_ball[1])), (int(biggest_ball[2]), int(biggest_ball[3])), (255, 255, 255), 2)

            msg = Point()
            msg.y = 1.0 * (45625.0 / r) * 0.788 + 290
            msg.x = 1.0 * (95.0 / r) * (x - 320)
            msg.z = 0
            self.position_pub.publish(msg)
            print(msg.x, msg.y)

        self.balls = []

    def get_radius(self, ball):
        a = ball[2] - ball[0]
        b = ball[3] - ball[1]
        if a / b < 2.0 and b / a < 2.0:
            return (a + b) / 4
        elif a / b > 2.0:
            return a / 2
        elif b / a > 2.0:
            return b / 2


def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":
    main()
