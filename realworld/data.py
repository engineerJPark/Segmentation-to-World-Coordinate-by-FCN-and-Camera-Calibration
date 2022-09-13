import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import collections
import PIL


class VOCClassSegBase(torch.utils.data.Dataset):
  '''
  input size : (480, 640, 3) (480, 640)
  '''

  class_names = np.array([
      'background',
      'roll',
      'sauce',
      'snack',
  ])

  # def __init__(self, root='realworld', split='train', transform_tf=True):
  def __init__(self, root='realworld', transform_tf=True):
      self.root = root
      self.transform_tf = transform_tf
      self.transform = transforms.Compose([
          transforms.Normalize(mean=(0, 0, 0), std=(255., 255., 255.)),
          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
      ])

      file_names_rgb = os.listdir('realworld/real_dataset/rgb')
      file_names_rgb.sort()

      DATASET_DIR = os.path.join(self.root, 'real_dataset')
      # self.files = collections.defaultdict(list)
      # for split in ['train', 'val']:
      #   for i in range(len(file_names_rgb)):
      #       img_file = os.path.join(DATASET_DIR, 'rgb/rs_image_%d.jpg' % (i + 1))
      #       lbl_file = os.path.join(DATASET_DIR, 'gt/rs_gt_%d.png' % (i + 1))
      #       self.files[split].append({
      #           'img': img_file,
      #           'lbl': lbl_file,
      #       })
      self.files = []
      for i in range(len(file_names_rgb)):
          img_file = os.path.join(DATASET_DIR, 'rgb/rs_image_%d.jpg' % (i + 1))
          lbl_file = os.path.join(DATASET_DIR, 'gt/rs_gt_%d.png' % (i + 1))
          self.files.append({
              'img': img_file,
              'lbl': lbl_file,
          })

  def __len__(self):
      # return len(self.files[self.split])
      return len(self.files)

  def __getitem__(self, index):
      # data file
      # data_file = self.files[self.split][index]
      data_file = self.files[index]
      
      # load
      img_file = data_file['img']
      img = PIL.Image.open(img_file)
      img = torch.from_numpy(np.array(img)).to(torch.float)

      lbl_file = data_file['lbl']
      lbl = PIL.Image.open(lbl_file)
      lbl = torch.from_numpy(np.array(lbl)).to(torch.long)
      lbl = torch.unsqueeze(lbl, dim=0)

      # image preprocessing
      img = img.permute(2, 0, 1) # HWC -> CHW 
      lbl[lbl == 255] = -1

      # image transform
      if self.transform_tf == True:
          return self.transform(img), lbl
      else:
          return img, lbl