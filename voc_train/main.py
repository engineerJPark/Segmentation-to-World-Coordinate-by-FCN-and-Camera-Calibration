from train import train
from predict import predict
from iou import mean_iou
from pix_acc import mean_foreground_pixel_acc

import matplotlib.pyplot as plt

if __name__ = '__main__': 
  if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(1)
  else:
    device = torch.device('cpu')
    torch.manual_seed(1)
  print(torch.__version__)
  print(device)

  model = FCN18(21).to(device)
  # PATH = 'voc_train/fcn_model/model_95_66'
  # checkpoint = torch.load(PATH)
  # model.load_state_dict(checkpoint['model_state_dict'])
  
  lr = 1e-6
  weight_decay = 1e-4
  momentum = 0.9

  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
  criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.5)
  # scheduler = None

  history = train(model, optimizer, criterion, scheduler, device=device, \
                  epochs = 2, lr = 1e-5, weight_decay = 1e-4, momentum = 0.9, verbos_iter=False)
  plt.plot(history)
  plt.show()  

  seg_plot(model, 0, device=device)
  mean_iou(model, device=device)
  mean_foreground_pixel_acc(model, device=device)