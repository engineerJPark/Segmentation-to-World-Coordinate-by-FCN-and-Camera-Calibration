from torch.utils.data import DataLoader
from data import VOCClassSegBase
import torch

# for plotting
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (10.0, 20.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
from matplotlib import cm
import PIL

def _fast_hist(label_true, label_pred, n_class, device='cpu'):
  '''
  row = ground truth
  column = prediction
  '''
  mask = ((label_true >= 0) & (label_true < n_class)).to(device)
  hist = torch.bincount(
      n_class * label_true[mask].to(torch.int) +
      label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class).to(device)
  return hist

def label_accuracy_score(val_model, n_class, device='cpu', verbose=True):
  """Returns accuracy score evaluation result.
    - overall accuracy
    - mean accuracy
    - mean IU
    - fwavacc
    diag elements are true positive
    row is gt, column is prediction
  """
  val_model = val_model.to(device)
  val_model.eval()
  print('model evaluation start')

  # for test data
  ROOT_DIR = 'voc_train/voc_data/'
  val_data = VOCClassSegBase(root=ROOT_DIR, split='val', transform_tf=True)
  val_data_loader = DataLoader(dataset=val_data, batch_size = 1, drop_last=True)
  
  with torch.no_grad():
    hist = torch.zeros((n_class, n_class)).to(device)

    # model prediction
    for iter, (label_preds, label_trues) in enumerate(val_data_loader):

      label_preds = label_preds.to(device)
      label_trues = label_trues.squeeze(dim=0).squeeze(dim=0).to(device)
      
      label_seg = val_model(label_preds) # 1CHW
      label_seg = torch.squeeze(label_seg, dim=0) # CHW
      label_preds = torch.argmax(label_seg, dim=0) # HW
      hist += _fast_hist(label_trues.flatten(), label_preds.flatten(), n_class, device=device)


    acc = torch.diag(hist).sum() / hist.sum()
    acc_cls = torch.diag(hist) / hist.sum(dim=1)
    acc_cls = torch.nanmean(acc_cls)
    iu = torch.diag(hist) / (
        hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)
    )
    mean_iu = torch.nanmean(iu)
    freq = hist.sum(dim=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

  if verbose == True:
    print("acc : ", acc.item())
    print("acc_cls : ", acc_cls.item())
    print("mean_iu : ", mean_iu.item())
    print("fwavacc : ", fwavacc.item())
  return acc, acc_cls, mean_iu, fwavacc
