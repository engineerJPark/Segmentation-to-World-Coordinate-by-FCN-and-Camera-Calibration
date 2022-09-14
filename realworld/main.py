'''
TODO
check whether iou, pix acc is not wrong to control the voc data.
'''

from train import train
from fcn import FCN18
from utils import label_accuracy_score, seg_plot

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
import copy
import collections

if __name__ == '__main__': 
  if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(1)
  else:
    device = torch.device('cpu')
    torch.manual_seed(1)
  print(torch.__version__, device)

  model = FCN18(4)

  # resume training
  print("resume training ... ")
  PATH = 'voc_train/fcn_model/model_9_12_13_46_200'
  checkpoint = torch.load(PATH)
  state_dict = checkpoint['model_state_dict']
  new_state_dict = collections.OrderedDict()
  # voc pretrained to realworld
  with torch.no_grad():
    for key in state_dict:
      if key.split('.')[0][:-1] == 'downsample' \
        or key.split('.')[0][:-1] == 'fc':
        new_state_dict[key] = copy.deepcopy(state_dict[key])
  model.load_state_dict(new_state_dict, strict=False)
  
  epochs = 600
  lr = 1e-4
  weight_decay = 1e-5
  momentum = 0.99
  
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
  criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
  # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
  scheduler = None

  # train
  loss_history= train(model, optimizer, criterion, scheduler, epochs = epochs, device=device)
  print(loss_history)
  plt.plot(loss_history)
  plt.show()
  plt.savefig('voc_train/loss_history.png')

  acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(model, 4, device=device, verbose=True)
  
  seg_plot(model, 500, device=device)