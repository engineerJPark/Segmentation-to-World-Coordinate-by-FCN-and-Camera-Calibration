from fcn import FCN18
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def train(model, epochs, optimizer, criterion, device='cpu', epochs = 100, lr = 1e-5, weight_decay = 1e-4, momentum = 0.9, pre_model=None, verbos_iter=True, verbos_epoch=True):
  model = FCN18(21).to(device)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
  criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.5)
  
  if pre_model is not None:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

  model.train()
  print('train mode start')

  loss_history =[]
  last_LOSS = 10 ** 9

  for epoch in range(epochs):
    running_loss = 0
    for iter, (train_img, train_gt_img) in enumerate(train_data_loader):
      train_img = train_img.to(device)
      train_gt_img = train_gt_img.squeeze(dim=1).to(device)

      # print(train_img.shape)
      # print(train_gt_img.shape)
      # print(train_img.permute(0,2,3,1).reshape(-1, 3).shape)
      # print(train_gt_img.reshape(-1, ).shape)

      # prediction
      score_img = model(train_img)
      score_img = score_img.permute(0,2,3,1)
      score_img = score_img.reshape(-1, score_img.shape[3]) # C H W 
      train_gt_img = train_gt_img.reshape(-1, )

      loss = criterion(score_img, train_gt_img)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      if verbos_iter == True:
        print("epoch %d, iteration: %d, loss : %f "%(epoch + 1, iter + 1, loss))

      running_loss += loss
    
    if verbos_epoch == True:
      print('======================================')
      print("epoch %d, loss : %f "%(epoch + 1, running_loss / len(train_data_loader)))

    now = datetime.datetime.now()
    EPOCH = epoch
    PATH = "fcn_model/model_%d%d_%d" % (now.month, now.day, epoch)
    LOSS = running_loss
    loss_history.append(LOSS.item())
    
    if last_LOSS > LOSS:
      torch.save({
                  'epoch': EPOCH,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': LOSS,
                  }, PATH)
      last_LOSS = LOSS
      
    scheduler.step()
  
  print("Training End")
  return loss_history