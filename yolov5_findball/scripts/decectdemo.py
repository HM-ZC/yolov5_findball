#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
from cv_bridge import CvBridge

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov5_findball_msgs.msg import BoundingBox
from yolov5_findball_msgs.msg import BoundingBoxes
from yolov5_findball_msgs.msg import Silos
import torch
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)

# local_model_path = "models/yolov5"
# # 加载模型
# model = torch.hub.load(
#     "ultralytics/yolov5",
#     "custom",
#     path=local_model_path,
#     device='cpu',
#     force_reload=True,
#     _verbose=True,
# )

# import sys
# sys.path.insert(0, 'weights')
# model = torch.load('weights/yolov5s.pt')

# model= torch.hub.load(repo_or_dir="/root/.cache/torch/hub/ultralytics_yolov5_master/",
#                                   model="yolov5s",
#                                   force_reload=True,
#                                   source='local')

# 移植的话 改绝对路径
model = torch.hub.load(r'/home/robocon/ws/src/cameratest2/src/yolov5_findball/yolov5', 'custom', path=r'/home/robocon/ws/src/cameratest2/src/yolov5_findball/weights/yolov5s.pt', source='local')



class Yolo_Dect:

    silos = ["null","null","null","null","null"]
    silos_pub = ["null","null","null","null","null"]
    previous_data = ["null","null","null","null","null"]
    count = [0,0,0,0,0]
    silo_data3=["null","null","null","null","null"]
   
    def __init__(self):

        silo_data3=[]
        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '')

        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        pub_topic1 = rospy.get_param('~pub_topic', '/yolov5/silos')

        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')

        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom',
                                    path=weight_path, source='local')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageStatus = False

        # Load class color
        self.classes_colors = {}

        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)

        self.image_pub = rospy.Publisher(
            '/yolov5/detection_image',  Image, queue_size=1)

        self.silo_pub = rospy.Publisher(
            pub_topic1,  Silos, queue_size=1)

        # image subscribe
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1, buff_size=52428800)

        # if no image messages
        while (not self.getImageStatus) :
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.he

    def equare(self, x, y):
        if x == y:
            return 1
        else:
            return 0

    def image_callback(self, image):
        # self.boundingBoxes = BoundingBoxes()
        # self.boundingBoxes.header = image.header
        # self.boundingBoxes.image_header = image.header
        # self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(self.color_image)
        # xmin    ymin    xmax   ymax  confidence  class    name

        boxs = results.pandas().xyxy[0].values
        self.dectshow(self.color_image, boxs, image.height, image.width)
        cv2.waitKey(1)

    def dectshow(self, org_img, boxs, height, width):
        img = org_img.copy()
        silos_data = []
        silos_data1 = []
        silos_data2 = []
        silos_pub=[]
  
        


        count = 0
        for i in boxs:
            count += 1

        for box in boxs:

            x=[box[0],box[-1]]

            boundingBox = BoundingBox()
            boundingBox.probability =np.float64(box[4])
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

            cv2.rectangle(img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (int(color[0]),int(color[1]), int(color[2])), 2)

            if box[1] < 20:
                text_pos_y = box[1] + 30
            else:
                text_pos_y = box[1] - 10
                
            cv2.putText(img, box[-1],
                        (int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            # self.boundingBoxes.bounding_boxes.append(boundingBox)
            # self.position_pub.publish(self.boundingBoxes)
            silos_data1.append(x)

        for num in silos_data1 :
            x = 1000.0
            index = 0
            for i,silo_data in enumerate(silos_data1):
                if(x > silo_data[0]):
                    x=silo_data[0]
                    index = i
            silos_data2.append(silos_data1[index][1])
            silos_data1[index][0] = 1000.1
        # self.silos_data = sorted(self.silos_data, key=lambda x: x[0])
        silosdata=self.judge(silos_data2)
        print(self.silos) 
        
        # silo_pubdata=self.pubjudge(silosdata,2)
       
        # print(silo_pubdata)

        msg = Silos()
        msg.a = self.silos[0]
        msg.b = self.silos[1]
        msg.c = self.silos[2]
        msg.d = self.silos[3]
        msg.e = self.silos[4]
        self.silo_pub.publish(msg)

        ##检测内容输出
        # print(self.silos)
        
        # self.publish_image(img, height, width)
        ##检测框 上车后注释掉
        cv2.imshow('YOLOv5', img)



def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()

if __name__ == "__main__":

    main()