import cv2
import math
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from predictor import predictor, predict_coord
import open3d as o3d


class camera_node():
    def __init__(self):
        self.bridge = CvBridge()
        self.predictor = predict_coord('/home/kkiruk/catkin_ws/src/js_ws/src/model_8_16_2_47_8')
        self.predictor.calibrate_camera()

    def get_img(self,visualize = False): # (480, 640, 3) (480, 848)
        data = rospy.wait_for_message('/camera/color/image_raw',Image)
        data2 = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw',Image)   
        cv_image= self.bridge.imgmsg_to_cv2(data,"bgr8")
        cv_image_depth= self.bridge.imgmsg_to_cv2(data2,"32FC1")
        cv_image_depth /= 1000 # original depth unit is milimeter
        image_mask, rgbd_image_list = self.predictor.predict_seg(cv_image, cv_image_depth)
        
        # get pointcloud
        pcd0 = self.predictor.get_pointcloud(rgbd_image_list[0])
        pcd1 = self.predictor.get_pointcloud(rgbd_image_list[1])
        pcd2 = self.predictor.get_pointcloud(rgbd_image_list[2])
        pcd3 = self.predictor.get_pointcloud(rgbd_image_list[3])

        # o3d.visualization.draw_geometries([pcd], zoom=0.5) # 안써도 됨
        print('pcd0 is :', pcd0) # 안써도 됨
        print('pcd0 has points :', pcd0.points) # 안써도 됨
        print('pcd1 is :', pcd1) # 안써도 됨
        print('pcd1 has points :', pcd1.points) # 안써도 됨
        print('pcd2 is :', pcd2) # 안써도 됨
        print('pcd2 has points :', pcd2.points) # 안써도 됨
        print('pcd3 is :', pcd3) # 안써도 됨
        print('pcd3 has points :', pcd3.points) # 안써도 됨
        
        # # testing shape and depth
        # print(cv_image.shape,cv_image_depth.shape)
        # print(cv_image_depth[240,320])

        if visualize:
            demo_image = cv2.resize(cv_image, None,  fx = 0.5, fy = 0.5) 
            demo_image_depth = cv2.resize(cv_image_depth, None,  fx = 0.5, fy = 0.5) 
            demo_image_seg =  cv2.resize(image_mask, None,  fx = 0.5, fy = 0.5) 
            cv2.imshow('rgb',demo_image)
            cv2.imshow('depth',demo_image_depth)
            cv2.imshow('segmentation',demo_image_seg)

        return cv_image, cv_image_depth, image_mask, (pcd0, pcd1, pcd2, pcd3)


if __name__ == '__main__':
    cam = camera_node()
    rospy.init_node("cam_test")

    iter = 1
    
    while not rospy.is_shutdown():
        rgb_img, depth_img, _, _ = cam.get_img(visualize=True)
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