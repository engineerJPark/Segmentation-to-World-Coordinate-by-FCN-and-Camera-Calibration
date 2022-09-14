from fcn import FCN18
from data import RealClassSegBase
from torch.utils.data import DataLoader
import torch
import datetime

def train(model, optimizer, criterion, scheduler=None, epochs=300, device='cpu', verbose=False):
  ROOT_DIR = 'realworld'
  train_data = RealClassSegBase(root=ROOT_DIR, transform_tf=True)
  train_data_loader = DataLoader(dataset=train_data, batch_size = 1, drop_last=True)

  loss_history = []
  last_loss = 10 ** 9

  model.to(device)
  model.train()
  print('train mode start')

  for epoch in range(epochs):
    
    total_loss = 0
    for iter, (train_img, train_gt_img) in enumerate(train_data_loader):
      print("iter : %d" % (iter + 1))

      # prediction
      train_img = model(train_img.to(device)).permute(0,2,3,1)
      train_img = train_img.reshape(-1, train_img.shape[-1]) # 1 H W C -> 1 H W C
      train_gt_img = train_gt_img.squeeze(dim=1).to(device).reshape(-1, ) # H W

      optimizer.zero_grad()
      loss = criterion(train_img, train_gt_img)
      loss.backward()
      optimizer.step()
      total_loss += float(loss)
    total_loss /= len(train_data_loader)

    print('====================================')
    print("epoch %d, loss : %f "%(epoch + 1, total_loss))
    loss_history.append(loss.item())

    
    if (epoch + 1) % 5 == 0 and last_loss > loss:
      now = datetime.datetime.now()
      PATH = "realworld/fcn_model/model_%d_%d_%d_%d_%d" % (now.month, now.day, now.hour, now.minute, epoch + 1)
      torch.save({
                  'epoch': epoch + 1,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': total_loss
                  }, PATH)
      last_loss = total_loss

    # if scheduler is not None:
    #   scheduler.step()
  
  print("Training End")
  return loss_history