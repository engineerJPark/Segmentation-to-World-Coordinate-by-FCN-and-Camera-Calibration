from fcn import FCN18
from data import VOCClassSegBase
from torch.utils.data import DataLoader
import torch
import datetime
from utils import label_accuracy_score


def train(model, optimizer, criterion, scheduler=None, epochs=300, device='cpu', verbose=False):
  ROOT_DIR = 'voc_train/voc_data/'
  train_data = VOCClassSegBase(root=ROOT_DIR, split='train', transform_tf=True)
  train_data_loader = DataLoader(dataset=train_data, batch_size = 1, drop_last=True)

  loss_history = []
  last_loss = 10 ** 9

  model.to(device)
  model.train()
  print('train mode start')

  # memory 누수
  for epoch in range(epochs):
    
    total_loss = 0
    for iter, (train_img, train_gt_img) in enumerate(train_data_loader):
      train_img = train_img.to(device)
      train_gt_img = train_gt_img.squeeze(dim=1).to(device)

      # prediction
      score_img = model(train_img)
      score_img = score_img.permute(0,2,3,1) # 1 H W C
      score_img = score_img.reshape(-1, score_img.shape[-1]) # 1 H W C
      train_gt_img = train_gt_img.reshape(-1, ) # H W

      optimizer.zero_grad()
      loss = criterion(score_img, train_gt_img)
      loss.backward()
      optimizer.step()
      total_loss += loss.data
    total_loss /= len(train_data_loader)

    print('====================================')
    print("epoch %d, loss : %f "%(epoch + 1, total_loss))
    loss_history.append(loss.item())

    
    if (epoch + 1) % 5 == 0 and last_loss > loss:
      now = datetime.datetime.now()
      PATH = "voc_train/fcn_model/model_%d_%d_%d_%d_%d" % (now.month, now.day, now.hour, now.minute, epoch + 1)
      torch.save({
                  'epoch': epoch + 1,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  # 'scheduler_state_dict': scheduler.state_dict(),
                  'loss': total_loss
                  }, PATH)
      last_loss = total_loss

    # if scheduler is not None:
    #   scheduler.step()
  
  print("Training End")
  return loss_history