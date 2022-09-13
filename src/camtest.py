from tkinter import S
import cv2
import math
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from scripts.predictor import predictor, predict_coord, SEG_CLASS_NAMES
import open3d as o3d
import numpy as np

class camera_node():
    '''
    class_n = number of classes without background
    searching_class : 
    1 : sauce
    2 : roll
    3 : snack
    '''
    def __init__(self):
        self.bridge = CvBridge()
    
    def get_rgb_depth_segmentation(self):
        '''
        get rgb, depth, segmentation image
        '''
        print("getting image from realsense")
        data = rospy.wait_for_message('/camera/color/image_raw',Image)
        data2 = rospy.wait_for_message('/camera/depth/image_raw',Image)   
        
        self.cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        self.cv_image_depth = self.bridge.imgmsg_to_cv2(data2,"32FC1") # original depth unit is milimeter

        # for test 
        print("cv_image.shape : ", self.cv_image.shape)
        print("cv_image_depth.shape : ", self.cv_image_depth.shape)

    def visualize(self):
        '''
        showing image, depth, segmentation
        '''
        demo_image = cv2.resize(self.cv_image, None,  fx = 0.5, fy = 0.5) 
        demo_image_depth = cv2.resize(self.cv_image_depth, None,  fx = 0.5, fy = 0.5) 
        demo_image_seg =  cv2.resize(self.image_mask, None,  fx = 0.5, fy = 0.5) 
        cv2.imshow('rgb', demo_image)
        cv2.imshow('depth', demo_image_depth)
        cv2.imshow('segmentation', demo_image_seg)
        return self.cv_image, self.cv_image_depth, self.image_mask

if __name__ == '__main__':
    # Ignore warnings in obj loader
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    
    cam = camera_node()
    rospy.init_node("segmentation_to_world")

    iter = 1


    while not rospy.is_shutdown():
        cam.get_rgb_depth_segmentation()
        rgb_img, depth_img, image_mask = cam.visualize()

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break 
        elif k == ord('s'):
            rgb_filename = "rs_image_%d.jpg" % (iter)
            cv2.imwrite(rgb_filename, rgb_img)
            depth_filename = "rs_depth_%d.jpg" % (iter)
            cv2.imwrite(depth_filename, depth_img)
            iter += 1
        
        rospy.sleep(0.5)
    cv2.destroyAllWindows()