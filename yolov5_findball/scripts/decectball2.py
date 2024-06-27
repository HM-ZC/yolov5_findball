                                                                                                                                                                                                                                                                                                                                             #!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import torch
import rospy
import numpy as np
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
from torch.quantization import quantize_dynamic
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov5_findball_msgs.msg import BoundingBox
from yolov5_findball_msgs.msg import BoundingBoxes
from geometry_msgs.msg import Point
class Yolo_Dect:
    def __init__(self):
        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '')
        weight_path = rospy.get_param('~weight_path', '')
        pub_topic = rospy.get_param('~pub_topic', '/camera1/ball_position')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf1 = rospy.get_param('~conf', '0.5')
        
        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')

        # which device will be used
        if rospy.get_param('/use_cpu', 'false'):
            self.model.cpu()
            
        else:
            self.model.cuda()
            

        self.model.conf = conf1
       
        self.color_image = Image()
        self.img = Image()
        self.depth_image = Image()
        self.getImageStatus = False
        
        # Load class color
        self.classes_colors = {}
        # output publishers
        self.position_pub = rospy.Publisher(pub_topic, Point, queue_size=1)
        self.bounding_boxes_pub = rospy.Publisher('/yolov5/detected_boxes', BoundingBoxes, queue_size=1)

        # image subscribe
        self.color_sub = rospy.Subscriber("/camera1/undistorted_image", Image, self.image_callback, queue_size=1, buff_size=52428800)

        self.locked_ball = None  # 用于存储锁定的球的信息
        self.detected_wood_frame = None  # 用于存储检测到的木框信息

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
        self.model = quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        results = self.model(self.color_image)

        boxs = results.pandas().xyxy[0].values
        self.detected_wood_frame = self.detect_wood_frame(boxs)

        self.dectshow(self.color_image, boxs, image.height, image.width)
        cv2.waitKey(3)

    def detect_wood_frame(self, wood_frames):
        for frame in wood_frames:
            if frame[-1] == "woodframe":
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
            cv2.putText(img, "woodframe", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        self.track_ball(boxs, img)
        cv2.imshow('YOLOv5', img)

    def track_ball(self, boxs, img):
        filtered_boxs = [box for box in boxs if not self.is_ball_in_wood_frame(box, self.detected_wood_frame)]

        if self.locked_ball is None:
            # 如果没有锁定的球，选择一个新的球进行锁定
            max_size = 0
            for box in filtered_boxs:
                if box[-1] == "purple":
                    size = (box[2] - box[0]) * (box[3] - box[1])
                    if size > max_size:
                        max_size = size
                        self.locked_ball = box
        else:
            # 如果有锁定的球，检查它是否仍然存在
            ball_exists = False
            for box in filtered_boxs:
                if self.is_same_ball(self.locked_ball, box):
                    self.locked_ball = box
                    ball_exists = True
                    break

            if not ball_exists:
                # 如果锁定的球不再存在，清除锁定状态
                self.locked_ball = None

        if self.locked_ball is not None:
            # 绘制锁定的球
            x = (self.locked_ball[2] + self.locked_ball[0]) / 2
            y = (self.locked_ball[3] + self.locked_ball[1]) / 2
            ballcenter = (int(x), int(y))

            r = self.get_radius(self.locked_ball)

            cv2.circle(img, ballcenter, int(r), (255, 255, 255), 2)
            cv2.rectangle(img, (int(self.locked_ball[0]), int(self.locked_ball[1])), (int(self.locked_ball[2]), int(self.locked_ball[3])), (255, 255, 255), 2)

            msg = Point()
            msg.y = r
            msg.x = x - 322
            msg.z = 0

            self.position_pub.publish(msg)
            print(msg.x, msg.y)

    def is_same_ball(self, ball1, ball2):
        # 判断两个球是否是同一个球，这里可以根据坐标和大小来判断
        return abs(ball1[0] - ball2[0]) < 20 and abs(ball1[1] - ball2[1]) < 20 and abs(ball1[2] - ball2[2]) < 20 and abs(ball1[3] - ball2[3]) < 20

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
