import cv2
import math
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from scripts.predictor import predictor, predict_coord
import open3d as o3d
import numpy as np

class camera_node():
    def __init__(self):
        self.bridge = CvBridge()
        self.predictor = predict_coord('./models/model_8_17_7_31_29')

    def get_img(self, class_n = 3, searching_class = 2, visualize = False, verbose = False):
        '''
        class_n = number of classes without background
        searching_class : 0, 1, 2. no background
        '''
        data = rospy.wait_for_message('/camera/color/image_raw',Image)
        data2 = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw',Image)   
        
        cv_image= self.bridge.imgmsg_to_cv2(data,"bgr8")
        cv_image_depth= self.bridge.imgmsg_to_cv2(data2,"32FC1") # original depth unit is milimeter
        image_mask, rgbd_image_list = self.predictor.predict_seg(cv_image, cv_image_depth)
        
        '''get pointcloud'''
        pcd = []
        for i in range(class_n):
            pcd.append(self.predictor.get_pointcloud(rgbd_image_list[i]))

        '''get center point for pointcloud'''
        if verbose == True:
            print("image's center's depth : ", cv_image_depth[320, 240], "mm", "at camera's coordinate")
            depth_scale = 1000 # mm
            try:
                min_x, min_y, min_z = np.min(pcd[searching_class].points, axis=0) * depth_scale
                max_x, max_y, max_z = np.max(pcd[searching_class].points, axis=0) * depth_scale
                cp = pcd[searching_class].get_center() * depth_scale # world coordinate 기준
                width = max_y - min_y
                height = max_z - min_z

                print("min xyz : ", min_x, min_y, min_z, "mm")
                print("max xyz : ", max_x, max_y, max_z, "mm")
                print("")
                print("                        depth            horizontal            vertical")
                print("center point : ", cp.tolist(), "mm")
                print("")
                print("width : ", width, "mm")
                print("height : ", height, "mm")
                print("==============================================")

            except(ValueError):
                print("there is segmentation or point cloud for searching_class : snack")
                print("==============================================")
            except(IndexError):
                print("give me index between class numbers...")
            except:
                print("Unknown Error...")


        if visualize:
            # '''showing pointcloud'''
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=640, height=480)
            # # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
            # # vis.add_geometry(axis_pcd)
            # vis.add_geometry(pcd1)
            # vis.add_geometry(pcd2)
            # vis.add_geometry(pcd3)
            # o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.5)
            # vis.run()
            # vis.destroy_window()

            '''showing image, depth, segmentation'''
            demo_image = cv2.resize(cv_image, None,  fx = 0.5, fy = 0.5) 
            demo_image_depth = cv2.resize(cv_image_depth, None,  fx = 0.5, fy = 0.5) 
            demo_image_seg =  cv2.resize(image_mask, None,  fx = 0.5, fy = 0.5) 
            cv2.imshow('rgb',demo_image)
            cv2.imshow('depth',demo_image_depth)
            cv2.imshow('segmentation',demo_image_seg)

        return cv_image, cv_image_depth, image_mask, pcd

if __name__ == '__main__':
    # Ignore warnings in obj loader
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    
    cam = camera_node()
    rospy.init_node("segmentation_to_world")

    iter = 1

    try:
        idx = input("put class number you want.")
        idx = int(idx)
    except(ValueError):
        print("try again, I only get integer.")
    
    while not rospy.is_shutdown():
        rgb_img, depth_img, _, _ = cam.get_img(searching_class = 2, visualize=True, verbose=True)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break 
        elif k == ord('s'):
            rgb_filename = "rs_image_%d.jpg" % (iter)
            cv2.imwrite(rgb_filename, rgb_img)
            depth_filename = "rs_depth_%d.jpg" % (iter)
            cv2.imwrite(depth_filename, depth_img)
            iter += 1
        
        rospy.sleep(0.2)
    cv2.destroyAllWindows()