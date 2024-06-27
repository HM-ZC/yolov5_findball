import cv2
import math
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 鱼眼相机画面矫正
#     arg：mat frame
#     return：mat frame

def undistort(frame):

	k=np.array( [[478.3854 ,  0.3169   ,331.9190],
                 [  0.     ,  478.5842 ,226.7371],
                 [  0.     ,  0.       ,1.      ]])

	d=np.array([-0.4085 , 0.1536 ,6.157666674460974e-04 ,-0.0017 ,0  ])
	h,w=frame.shape[:2]
	mapx,mapy=cv2.initUndistortRectifyMap(k,d,None,k,(w,h),5)
	return cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

def main():
    rospy.init_node("camera_publisher2")
    pub = rospy.Publisher("/camera/image_raw2", Image, queue_size=10)
    dev = rospy.get_param('~dev', '')

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        rospy.logerr("Cannot open camera")
        return
    bridge = CvBridge()

    rate = rospy.Rate(30)  
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        frame=undistort(frame)
        if ret:
            image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            pub.publish(image_msg)
        rate.sleep()

    # 释放摄像头
    cap.release()

if __name__ == '__main__':
    try:
        main()    

    except rospy.ROSInterruptException:
        pass

