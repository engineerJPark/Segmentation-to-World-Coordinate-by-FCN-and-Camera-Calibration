import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# for plotting
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 20.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
from matplotlib import cm

import collections
import PIL # turn to dataset
import os
import datetime

def predict(test_model, ):


test_model = model
test_model.eval()
print('model evaluation start')

# test_model = FCN18(21).to(device)

# ############################################################ fix this path
# PATH = 'fcn_model/model_9_2_22_52_53'

# checkpoint = torch.load(PATH)
# test_model.load_state_dict(checkpoint['model_state_dict'])

# test_model.eval()
# print('model evaluation start')

######################################################

# segmentation : plot image
with open(os.path.join(ROOT_DIR, "VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"), 'r') as f:
  lines = f.readlines()
for i in range(len(lines)):
  lines[i] =  lines[i].strip('\n')

idx = 1000
test_jpg_path = lines[idx] + '.jpg'
test_image = PIL.Image.open(os.path.join(ROOT_DIR, 'VOCdevkit/VOC2012', "JPEGImages", test_jpg_path))

# test image showing
plt.figure(figsize=(20, 40))
plt.subplot(1,2,1)
plt.imshow(test_image)

# test image transform & input to test model
test_image = np.array(test_image)
test_image = torch.from_numpy(test_image).to(torch.float).permute(2,0,1).to(device)
ori_x, ori_y = test_image.shape[1], test_image.shape[2]

test_image = torch.unsqueeze(test_image, dim=0)

test_transform = transforms.Compose([
    transforms.Normalize(mean=(0, 0, 0), std=(255., 255., 255.)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

test_seg = test_model(test_transform(test_image))

# model prediction
test_image_channel_idx = torch.argmax(torch.squeeze(test_seg, dim=0), dim=0).cpu()

# model prediction to PIL
test_image_PIL = PIL.Image.fromarray(
    np.uint8(cm.gist_ncar(test_image_channel_idx.detach().numpy()*10)*255)
    )

# predicted data showing
plt.subplot(1,2,2)
plt.imshow(test_image_PIL)
plt.show()

###############################################

# segmentation : plot image
with open(os.path.join(ROOT_DIR, "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"), 'r') as f:
  lines = f.readlines()
for i in range(len(lines)):
  lines[i] =  lines[i].strip('\n')

idx = 1000
test_jpg_path = lines[idx] + '.jpg'
test_image = PIL.Image.open(os.path.join(ROOT_DIR, 'VOCdevkit/VOC2012', "JPEGImages", test_jpg_path))

# test image showing
plt.figure(figsize=(20, 40))
plt.subplot(1,2,1)
plt.imshow(test_image)

# test image transform & input to test model
test_image = np.array(test_image)
test_image = torch.from_numpy(test_image).to(torch.float).permute(2,0,1).to(device)
ori_x, ori_y = test_image.shape[1], test_image.shape[2]

test_image = torch.unsqueeze(test_image, dim=0)

test_transform = transforms.Compose([
    transforms.Normalize(mean=(0, 0, 0), std=(255., 255., 255.)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

test_seg = test_model(test_transform(test_image))

# model prediction
test_image_channel_idx = torch.argmax(torch.squeeze(test_seg, dim=0), dim=0).cpu()

# model prediction to PIL
test_image_PIL = PIL.Image.fromarray(
    np.uint8(cm.gist_ncar(test_image_channel_idx.detach().numpy()*10)*255)
    )

# predicted data showing
plt.subplot(1,2,2)
plt.imshow(test_image_PIL)
plt.show()

##############################################################

'''
referenced from
https://stackoverflow.com/a/48383182
'''

# IoU function
def iou(pred, target, n_classes = 4):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    
    intersection = int((pred_inds * target_inds).sum().item())
    union = int((pred_inds + target_inds).sum().item())
    
    # print(intersection, union) # for test
    
    if int(target_inds.sum().item()) == 0 and int(pred_inds.sum().item()) == 0:
      continue
    
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(union)) # float(max(union, 1)))
    
  return np.array(ious), np.array(ious).mean()

################################################################

# for test data
with open(os.path.join(ROOT_DIR, "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"), 'r') as f:
  lines = f.readlines()
for i in range(len(lines)):
  lines[i] =  lines[i].strip('\n')

iter = 0
iou_stack = 0

for idx in range(len(lines)):
  test_jpg_path = lines[idx] + '.jpg'
  test_png_path = lines[idx] + '.png'
  test_image = PIL.Image.open(os.path.join(ROOT_DIR, 'VOCdevkit/VOC2012', "JPEGImages", test_jpg_path))
  test_gt_image = PIL.Image.open(os.path.join(ROOT_DIR, 'VOCdevkit/VOC2012', "SegmentationObject", test_png_path))

  # test image transform & input to test model
  test_image = np.array(test_image)
  test_image = torch.from_numpy(test_image).to(torch.float).permute(2,0,1).to(device)
  ori_x, ori_y = test_image.shape[1], test_image.shape[2]
  test_image = torch.unsqueeze(test_image, dim=0)

  test_transform = transforms.Compose([
      transforms.Normalize(mean=(0, 0, 0), std=(255., 255., 255.)),
      transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  ])
  # return_transform = transforms.Compose([
  #     transforms.Resize((ori_x, ori_y), interpolation=InterpolationMode.BILINEAR),
  # ])

  test_seg = test_model(test_transform(test_image))
  # test_seg = return_transform(test_seg)
  # test_seg[test_seg <= 8] = 0 # Thresholdings
  test_seg = torch.squeeze(test_seg, dim=0)

  # model prediction
  test_image_channel_idx = torch.argmax(test_seg, dim=0).cpu()

  # ground truth image getting
  test_gt_image = np.array(test_gt_image)
  test_gt_image = torch.from_numpy(test_gt_image).to(torch.int)

  iter += 1
  _, metric = iou(test_image_channel_idx, test_gt_image, 21)
  print("iou of %d th " % (iter), " : ", metric)
  iou_stack += metric

mean_iou = iou_stack / iter
print("mean_iou : ", mean_iou)

###############################################################

def foreground_pixel_acc(pred, gt, class_num):
  true_positive_stack = 0
  all_stack = 0
  for class_i in range(1, class_num+1):
    true_positive = (pred == class_i) * (gt == class_i)
    all = (gt == class_i)

    true_positive_stack += true_positive.sum()
    all_stack += all.sum()

  return true_positive_stack / all_stack

##################################################################

# foreground pixel accuracy

with open(os.path.join(ROOT_DIR, "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"), 'r') as f:
  lines = f.readlines()
for i in range(len(lines)):
  lines[i] =  lines[i].strip('\n')

iter = 0
acc_stack = 0

for idx in range(len(lines)):
  test_jpg_path = lines[idx] + '.jpg'
  test_png_path = lines[idx] + '.png'
  test_image = PIL.Image.open(os.path.join(ROOT_DIR, 'VOCdevkit/VOC2012', "JPEGImages", test_jpg_path))
  test_gt_image = PIL.Image.open(os.path.join(ROOT_DIR, 'VOCdevkit/VOC2012', "SegmentationObject", test_png_path))

  # test image transform & input to test model
  test_image = np.array(test_image)
  test_image = torch.from_numpy(test_image).to(torch.float).permute(2,0,1).to(device)
  ori_x, ori_y = test_image.shape[1], test_image.shape[2]
  test_image = torch.unsqueeze(test_image, dim=0)

  test_transform = transforms.Compose([
      transforms.Resize((320,320), interpolation=InterpolationMode.NEAREST),
      transforms.Normalize(mean=(0, 0, 0), std=(255., 255., 255.)),
      transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  ])
  return_transform = transforms.Compose([
      transforms.Resize((ori_x, ori_y), interpolation=InterpolationMode.NEAREST),
  ])

  test_seg = test_model(test_transform(test_image))
  test_seg = return_transform(test_seg)
  # test_seg[test_seg <= 8] = 0 # Thresholdings
  test_seg = torch.squeeze(test_seg, dim=0)

  # model prediction
  test_image_channel_idx = torch.argmax(test_seg, dim=0).cpu()

  # ground truth image getting
  test_gt_image = np.array(test_gt_image)
  test_gt_image = torch.from_numpy(test_gt_image).to(torch.int)

  iter += 1
  metric = foreground_pixel_acc(test_image_channel_idx, test_gt_image, 21)
  print("foreground pixel acc of %d th " % (iter), " : ", metric)
  acc_stack += metric

acc = acc_stack / iter
print("acc : ", acc.item())