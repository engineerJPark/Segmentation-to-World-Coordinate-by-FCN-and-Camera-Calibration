from fcn import FCN18
from data import VOCClassSegBase
from torch.utils.data import DataLoader
import torch
import datetime


def train(model, optimizer, criterion, scheduler=None, epochs = 100, device='cpu',  verbos_iter=True, verbos_epoch=True):
  ROOT_DIR = 'voc_train/voc_data/'
  train_data = VOCClassSegBase(root=ROOT_DIR, split='train', transform_tf=True)
  train_data_loader = DataLoader(dataset=train_data, batch_size = 1, drop_last=True)

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
    PATH = "voc_train/fcn_model/model_%d%d_%d" % (now.month, now.day, epoch)
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

    if scheduler is not None:
      scheduler.step()
  
  print("Training End")
  return loss_history