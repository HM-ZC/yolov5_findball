# Author: moon
# CreaTime: 2024/3/14
# FileName: save.py
import cv2
import numpy as np

def undistort(frame):

	k=np.array( [[478.3854 ,  0.3169   ,331.9190],
                 [  0.     ,  478.5842 ,226.7371],
                 [  0.     ,  0.       ,1.      ]])

	d=np.array([-0.4085 , 0.1536 ,6.157666674460974e-04 ,-0.0017 ,0  ])
	h,w=frame.shape[:2]
	mapx,mapy=cv2.initUndistortRectifyMap(k,d,None,k,(w,h),5)
	return cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

if __name__ == '__main__':
    n=1
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        retval,img=cap.read()
        img = undistort(img)
        cv2.imshow("img", img)
        if cv2.waitKey(25) == ord('q'):
            cv2.imwrite("/home/yuyu/workspace/robocon/camera/orddata/"+str(n)+".jpg",img)
            n=n+1
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     cap = cv2.VideoCapture(0)
#     width = int(cap.get(propId=cv2.CAP_PROP_FRAME_WIDTH) / 2)
#     height = int(cap.get(propId=cv2.CAP_PROP_FRAME_HEIGHT) / 2)
#     fourcc=cv2.VideoWriter.fourcc(*"mp4v")
#     output = cv2.VideoWriter("output.mp4",fourcc,20,(640,480))
#     if not cap.isOpened():
#             print("Cannot open camera")
#             exit()
#     while True:
#             retval, img = cap.read()
#             img = undistort(img)
#             output.write(img)
#             cv2.imshow("img", img)
#             if cv2.waitKey(20) == ord('q'):
#                 break
#     cap.release()
#     output.release()
#     cv2.destroyAllWindows()