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
    def __init__(self,path = '../js_ws/src/models/model_8_17_7_31_29'):
        self.bridge = CvBridge()
        self.predictor = predict_coord(path)
    
    def get_rgb_depth_segmentation(self):
        '''
        get rgb, depth, segmentation image
        '''
        data = rospy.wait_for_message('/camera/color/image_raw',Image)
        data2 = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw',Image)   
        
        self.cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        self.cv_image_depth = self.bridge.imgmsg_to_cv2(data2,"32FC1") # original depth unit is milimeter
        self.image_mask, self.rgbd_image_list = self.predictor.predict_seg(self.cv_image, self.cv_image_depth)

    def segmentation_to_pointcloud(self, searching_class = -1, class_n = 4):
        '''
        get pointcloud and center point
        '''
        self.pcd = []
        for i in range(class_n - 1):
            self.pcd.append(self.predictor.get_pointcloud(self.rgbd_image_list[i]))

        try:
            depth_scale = 1000 # mm
            min_x, min_y, min_z = np.min(self.pcd[searching_class - 1].points, axis=0) * depth_scale
            max_x, max_y, max_z = np.max(self.pcd[searching_class - 1].points, axis=0) * depth_scale
            cp = self.pcd[searching_class - 1].get_center() * depth_scale # world coordinate 기준
            recommended_cp = cp.copy()
            recommended_cp[2] = 50 # mm
            width = max_y - min_y
            height = max_z - min_z

            print("image's center's depth : ", self.cv_image_depth[320, 240], "mm", "at camera's coordinate")
            print("min xyz : ", min_x, min_y, min_z, "mm")
            print("max xyz : ", max_x, max_y, max_z, "mm")
            print("")
            print("                      depth(x)            horizontal(y)            vertical(z)")
            print("center point : ", cp.tolist(), "mm")
            print("recommended  : ", recommended_cp.tolist(), "mm")
            print("")
            print("width : ", width, "mm")
            print("height : ", height, "mm")
            print("==============================================")

            return {
                "cp": cp,
                "recommended_cp": recommended_cp,
                "width": width
            }

        except(ValueError):
            print("there is no segmentation or point cloud for searching_class : ", SEG_CLASS_NAMES[searching_class])
            print("==============================================")
        except(IndexError):
            print("give me index between class numbers...")
        except:
            print("Unknown Error...")

        return {
            "cp": None,
            "recommended_cp": None,
            "width": None
        }

    def visualize_pcd(self):
        '''
        showing pointcloud
        '''
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480)
        vis.add_geometry(self.pcd)
        o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.5)
        vis.run()
        vis.destroy_window()

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
    
    cam = camera_node('models/model_8_17_7_31_29')
    rospy.init_node("segmentation_to_world")

    iter = 1

    try:
        idx = input("put class number you want.")
        idx = int(idx)

        assert idx > 0 and idx <= 3, "No segmentation provided for background."
        print(SEG_CLASS_NAMES[idx])

    except(ValueError):
        print("try again, I only get integer.")
    
    while not rospy.is_shutdown():
        cam.get_rgb_depth_segmentation()
        pcd_dict = cam.segmentation_to_pointcloud(searching_class = idx, class_n = 4)
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