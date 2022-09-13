from scripts.fcn import FCN18
import torch
import cv2
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from matplotlib import cm
import numpy as np
import cv2
import math
import open3d as o3d

SEG_CLASS_NAMES = ['bg','roll','sauce','snack']

class predictor():
  def __init__(self, path, device= 'cpu'):
    self.path = path
    self.device= device
    self.model = FCN18(4).to(self.device)

    checkpoint = torch.load(self.path)
    self.model.load_state_dict(checkpoint['model_state_dict'])

    self.model.eval()
    print('model evaluation start')

  def predict_seg(self, image, depth):
    '''
    image : PIL format, gonna be numpy array
    '''

    # image_np = np.array(image)
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2,0,1)
    image_torch = torch.from_numpy(image_np).to(torch.float)

    image_torch = torch.unsqueeze(image_torch, dim=0)

    test_transform = transforms.Compose([
        transforms.Normalize(mean=(0, 0, 0), std=(255., 255., 255.)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_seg = self.model(test_transform(image_torch).to(self.device))
    test_seg = test_seg.cpu()

    test_image_channel_idx = torch.argmax(torch.squeeze(test_seg, dim=0), dim=0) # final prediction
    test_image_mask = np.uint8(cm.gnuplot2(test_image_channel_idx.detach().numpy()*70)*255)

    # get RGBDImage
    test_image_channel_idx_stack = np.stack(np.array([test_image_channel_idx.detach().numpy(),\
                                              test_image_channel_idx.detach().numpy(), \
                                                test_image_channel_idx.detach().numpy()]), \
                                                  axis=0)
    
    # segmentation data to RGBDImage
    image_np_class1 = np.copy(image_np)
    image_np_class1[test_image_channel_idx_stack != 1] = 0
    image_np_class1 = image_np_class1.transpose(1,2,0) # HWC to CHW
    depth_np_class1 = np.copy(depth)
    depth_np_class1[test_image_channel_idx.detach().numpy() != 1] = 0
    image_np_class1 = np.asarray(image_np_class1, order='C')
    depth_np_class1 = np.asarray(depth_np_class1, order='C')
    image_o3d_class1 = o3d.geometry.Image(image_np_class1)
    depth_o3d_class1 = o3d.geometry.Image(depth_np_class1)
    
    image_np_class2 = np.copy(image_np)
    image_np_class2[test_image_channel_idx_stack != 2] = 0
    image_np_class2 = image_np_class2.transpose(1,2,0) # HWC to CHW
    depth_np_class2 = np.copy(depth)
    depth_np_class2[test_image_channel_idx.detach().numpy() != 2] = 0
    image_np_class2 = np.asarray(image_np_class2, order='C')
    depth_np_class2 = np.asarray(depth_np_class2, order='C')
    image_o3d_class2 = o3d.geometry.Image(image_np_class2)
    depth_o3d_class2 = o3d.geometry.Image(depth_np_class2)

    image_np_class3 = np.copy(image_np)
    image_np_class3[test_image_channel_idx_stack != 3] = 0
    image_np_class3 = image_np_class3.transpose(1,2,0) # HWC to CHW
    depth_np_class3 = np.copy(depth)
    depth_np_class3[test_image_channel_idx.detach().numpy() != 3] = 0
    image_np_class3 = np.asarray(image_np_class3, order='C')
    depth_np_class3 = np.asarray(depth_np_class3, order='C')
    image_o3d_class3 = o3d.geometry.Image(image_np_class3)
    depth_o3d_class3 = o3d.geometry.Image(depth_np_class3)

    rgbd_image_class0 = o3d.geometry.RGBDImage()
    rgbd_image_class1 = o3d.geometry.RGBDImage()
    rgbd_image_class2 = o3d.geometry.RGBDImage()
    rgbd_image_class3 = o3d.geometry.RGBDImage()
    
    # argument is for open3d.geometry.Image. devide by depth_scale = 1000.
    rgbd_out_class1 = rgbd_image_class1.create_from_color_and_depth(image_o3d_class1, depth_o3d_class1, convert_rgb_to_intensity=False)
    rgbd_out_class3 = rgbd_image_class3.create_from_color_and_depth(image_o3d_class3, depth_o3d_class3, convert_rgb_to_intensity=False)
    rgbd_out_class2 = rgbd_image_class2.create_from_color_and_depth(image_o3d_class2, depth_o3d_class2, convert_rgb_to_intensity=False)

    return test_image_mask, (rgbd_out_class1, rgbd_out_class2, rgbd_out_class3)

class predict_coord(predictor):
  def __init__(self, path):
    super(predict_coord, self).__init__(path)

    # by cv2.calibrateCamera
    self.intrinsic = o3d.camera.PinholeCameraIntrinsic()

    # self.intrinsic.intrinsic_matrix = np.array(
    #   [[512.10704929, 0., 320.], # theoritical value
    #   [0., 641.90915854, 240.],
    #   [0., 0., 1. ]]
    # )

    self.intrinsic.intrinsic_matrix = np.array(
      [[646.22925621,   0.,         299.50901429],
      [  0.,         611.69551895, 207.93689654],
      [  0.,           0.,           1.        ]]
    )

    # self.intrinsic.intrinsic_matrix = np.array([[622.90389931, 0., 325.17697045], 
    #                                             [0., 623.36739775, 250.94119723],
    #                                             [0., 0., 1. ]])

    r_mat = np.array(
      [[ 0.02181321, -0.99942881, -0.02581159],
       [-0.40214857,  0.01486566, -0.91545373],
       [ 0.91531454,  0.03034908, -0.4015946 ]]
    )

    c_mat = np.array([[-0.4], # 매니퓰레이터 원점에서부터 realsense 렌즈 중앙까지 직접 자로 측정(m) 46.5 5 37.5
                      [0.031],
                      [0.37]])
                      
    t_mat = -np.matmul(r_mat, c_mat)
    
    # extrinsic matrix define
    self.extrinsic_mat = np.concatenate((r_mat, t_mat), axis=1)
    self.extrinsic_mat = np.concatenate((self.extrinsic_mat, np.array([[0., 0., 0., 1.]])), axis=0)

    self.init_pointcloud = o3d.geometry.PointCloud()

  def get_pointcloud(self, rgbd_image):

    pcd = self.init_pointcloud.create_from_rgbd_image(rgbd_image, intrinsic=self.intrinsic, extrinsic=self.extrinsic_mat, project_valid_depth_only=True)
    pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # flip
    
    # height regularizer
    pcd_points = np.asarray(pcd.points)
    pcd_points[:,2] = np.clip(pcd_points[:,2], 0, 20)
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    
    # post-processing
    pcd = pcd.remove_non_finite_points()
    pcd, _ = pcd.remove_radius_outlier(nb_points=100, radius=0.01)
    
    return pcd
