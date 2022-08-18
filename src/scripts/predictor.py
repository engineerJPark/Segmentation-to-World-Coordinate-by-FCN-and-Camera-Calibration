'''
Reference

# unit vector from robot frame (world coordinate)
r_mat1 = np.array([
  [0., -1., 0.],
  [1., 0., 0.],
  [0., 0., 1.]
])
r_mat2 = np.array([
  [1., 0., 0.],
  [0., 0., -1.],
  [0., 1., 0.]
])
theta = 20 * math.pi / 180 # measured value. degrees to radian
r_mat3 = np.array([
  [1., 0., 0.],
  [0., math.cos(theta), math.sin(theta)],
  [0., -math.sin(theta), math.cos(theta)]
])
r_mat_result_1 = np.matmul(np.matmul(r_mat1, r_mat2), r_mat3)

theta = 20 * math.pi / 180 # measured value. degrees to radian
r_mat_result = np.array([[0, -1 * math.cos(theta), -1 * math.sin(theta)],
                  [-1 * math.sin(theta), 0, -1 * math.cos(theta)],
                  [1, 0, 0]])

print(r_mat_result_1)
print(r_mat_result_2)

# translation
# c_mat = np.array([400, 0., 400]) # robot to camera distance. 0.35, 0.4. unit mm 기준
c_mat = np.array([-350, 0., 380]) # robot to camera distance. 0.35, 0.4. unit mm 기준
t_mat_result = np.matmul(r_mat_result, c_mat.T)

print(t_mat_result.shape)
self.extrinsic_mat = np.concatenate((r_mat_result, t_mat_result.reshape(-1,1)), axis=1)
self.extrinsic_mat = np.concatenate((self.extrinsic_mat, np.array([[0., 0., 0., 1.]])), axis=0)
'''

'''
camera calibration을 직접하지 않고, projection matrix로부터 분해해서 얻는 방법은???
world coordiante 기준 x : 48 cm, y : 0 cm, z : 13 cm
image plane 기준 : 맨위 35 cm, 양쪽 92 -> 41cm, 거리 48 cm, 아래쪽은 55cm 가로, 거리는 8cm

0, 0, 1 -> 480, 410, 350, 1
320, 0, 1 -> 480, 0, 350, 1
640, 0, 1 -> 480, -410, 350, 1
0, 480, 1 -> 80, 550, 0, 1
320, 480, 1 -> 80, 0, 0, 1
640, 480, 1 -> 80, -550, 0, 1

'''

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

    test_transform = transforms.Compose([ # interpolation=InterpolationMode.BILINEAR
        transforms.Normalize(mean=(0, 0, 0), std=(255., 255., 255.)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_seg = self.model(test_transform(image_torch).to(self.device))
    # test_seg[test_seg <= 8] = 0 # Thresholding
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

    theta = 20 * math.pi / 180 # measured value. degrees to radian
    A_mat = np.array([
      [480, 410, 350, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 480, 410, 350, 1, 0, 0, 0, 0],

      [480, 0, 350, 1, 0, 0, 0, 0, -320*480, 0, -320*350, -320],
      [0, 0, 0, 0, 480, 0, 350, 1, 0, 0, 0, 0],

      [480, -410, 350, 1, 0, 0, 0, 0, -640*480, 640*410, -640*350, -640],
      [0, 0, 0, 0, 480, -410, 350, 1, 0, 0, 0, 0],

      [80, 550, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 80, 550, 0, 1, -480*80, -480*550, 0, -480],

      [80, 0, 0, 1, 0, 0, 0, 0, -320*80, 0, 0, -320],
      [0, 0, 0, 0, 80, 0, 0, 1, -480*80, 0, 0, -480],

      [80, -550, 0, 1, 0, 0, 0, 0, -640*80, 640*550, 0, -640],
      [0, 0, 0, 0, 80, -550, 0, 1, -480*80, 480*550, 0, -480]
    ])
    eig_val, eig_vec = np.linalg.eig(np.matmul(A_mat.T, A_mat))
    idx = np.argmin(eig_val)
    P_flatten = eig_vec[:, idx]
    P_mat = P_flatten.reshape(3, -1)
    # print("P_mat's norm : ", np.linalg.norm(P_mat))
    
    self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    self.extrinsic_mat = np.matmul(np.linalg.inv(self.intrinsic.intrinsic_matrix), P_mat)
    self.extrinsic_mat = np.concatenate((self.extrinsic_mat, np.array([[0,0,0,1]])), axis=0)
    self.init_pointcloud = o3d.geometry.PointCloud()

  def get_pointcloud(self, rgbd_image):

    pcd = self.init_pointcloud.create_from_rgbd_image(rgbd_image, intrinsic=self.intrinsic, extrinsic=self.extrinsic_mat, project_valid_depth_only=True)
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # Flip it, otherwise the pointcloud will be upside down
    
    return pcd