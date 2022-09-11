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

if __name__ == '__main__': 
  if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(1)
  else:
    device = torch.device('cpu')
    torch.manual_seed(1)
  print(torch.__version__, device)

  model = FCN18(21)
  model.copy_params_from_vgg16(vgg16(weights=VGG16_Weights.DEFAULT))

  # resume training
  PATH = 'voc_train/fcn_model/model_9_11_5_51_100'
  checkpoint = torch.load(PATH)
  model.load_state_dict(checkpoint['model_state_dict'])
  
  epochs = 300
  lr = 1e-10
  weight_decay = 5e-4
  momentum = 0.99
  
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
  criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
  scheduler = None

  # train
  loss_history= train(model, optimizer, criterion, scheduler, epochs = epochs, device=device)
  print(loss_history)
  plt.plot(loss_history)
  plt.show()
  plt.savefig('voc_train/loss_history.jpg')

  acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(model, 21, device=device, verbose=True)
  
  seg_plot(model, 500, device=device)