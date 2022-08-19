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

class predictor():
  def __init__(self, path):
    self.path = path
    self.device= 'cuda'
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
    image_np_class1 = image_np_class1.transpose(1,2,0)
    depth_np_class1 = np.copy(depth)
    depth_np_class1[test_image_channel_idx.detach().numpy() != 1] = 0
    image_np_class1 = np.asarray(image_np_class1, order='C')
    depth_np_class1 = np.asarray(depth_np_class1, order='C')
    image_o3d_class1 = o3d.geometry.Image(image_np_class1)
    depth_o3d_class1 = o3d.geometry.Image(depth_np_class1)
    
    image_np_class2 = np.copy(image_np)
    image_np_class2[test_image_channel_idx_stack != 2] = 0
    image_np_class2 = image_np_class2.transpose(1,2,0)
    depth_np_class2 = np.copy(depth)
    depth_np_class2[test_image_channel_idx.detach().numpy() != 2] = 0
    image_np_class2 = np.asarray(image_np_class2, order='C')
    depth_np_class2 = np.asarray(depth_np_class2, order='C')
    image_o3d_class2 = o3d.geometry.Image(image_np_class2)
    depth_o3d_class2 = o3d.geometry.Image(depth_np_class2)

    image_np_class3 = np.copy(image_np)
    image_np_class3[test_image_channel_idx_stack != 3] = 0
    image_np_class3 = image_np_class3.transpose(1,2,0)
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
    
    # open3d.geometry.Image에 맞춰서 넣어줘야한다.
    rgbd_out_class1 = rgbd_image_class1.create_from_color_and_depth(image_o3d_class1, depth_o3d_class1, convert_rgb_to_intensity=False)
    rgbd_out_class3 = rgbd_image_class3.create_from_color_and_depth(image_o3d_class3, depth_o3d_class3, convert_rgb_to_intensity=False)
    rgbd_out_class2 = rgbd_image_class2.create_from_color_and_depth(image_o3d_class2, depth_o3d_class2, convert_rgb_to_intensity=False)

    return test_image_mask, (rgbd_out_class1, rgbd_out_class2, rgbd_out_class3) # RGBDImage

class predict_coord(predictor):
  def __init__(self, path):
    super(predict_coord, self).__init__(path)
    # intrinsic
    fov_x, fov_y = 69.4, 42.5 # color FOV. depth FOV is 86, 57
    width, height = 480, 640
    fx, fy = width / (2 * math.tan(fov_x / 2)), width / (2 * math.tan(fov_y / 2))
    cx, cy = width // 2, height // 2 # need to be fixed if so
    self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # unit vector from robot frame (world coordinate)
    # theta = 20 * math.pi / 180 # measured value. degrees to radian
    # r_mat = np.array([
    #   [0, -1 * math.sin(theta), 1 * math.cos(theta)],
    #   [-1, 0, 0],
    #   [0, -1 * math.cos(theta), -1 * math.sin(theta)]
    # ])

    p = -90 * math.pi / 180
    t = -20 * math.pi / 180 
    r_mat2 = np.array([
      [math.cos(p), math.sin(p), 0],
      [-math.sin(p) * math.sin(t), math.cos(p) * math.sin(t), -math.cos(t)],
      [-math.sin(p) * math.cos(t), math.cos(p) * math.cos(t), math.sin(t)]
    ])

    print("r mat : ", r_mat)
    print("")
    print("r_mat2 : ", r_mat2)
    print("")

    # translation
    c_mat = np.array([-0.35, 0., 0.4]) # robot to camera distance. 0.35, 0.4
    t_mat = np.matmul(r_mat, c_mat.T)

    # extrinsic matrix define
    self.extrinsic_mat = np.concatenate((r_mat, t_mat.reshape(-1,1)), axis=1)
    self.extrinsic_mat = np.concatenate((self.extrinsic_mat, np.array([[0., 0., 0., 1.]])), axis=0)
    self.init_pointcloud = o3d.geometry.PointCloud()

  def get_pointcloud(self, rgbd_image):

    pcd = self.init_pointcloud.create_from_rgbd_image(rgbd_image, intrinsic=self.intrinsic, extrinsic=self.extrinsic_mat, project_valid_depth_only=True)
    # pcd = self.init_pointcloud.create_from_rgbd_image(rgbd_image, intrinsic=self.intrinsic, project_valid_depth_only=True)
    pcd.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # flip horizontaly, and depth position
    
    return pcd