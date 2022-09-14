from torch.utils.data import DataLoader
from data import RealClassSegBase
import torch
import numpy as np

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
  print('model evaluation : metric mode start')

  # for test data
  ROOT_DIR = 'realworld'
  val_data = RealClassSegBase(root=ROOT_DIR, transform_tf=True)
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

  print('model evaluation : metric mode ended')
  return acc, acc_cls, mean_iu, fwavacc

def seg_plot(val_model, idx, device='cpu'):
  '''
  val_model = FCN18(21).to(device)
  PATH = 'fcn_model/model_9_2_22_52_53'
  checkpoint = torch.load(PATH)
  val_model.load_state_dict(checkpoint['model_state_dict'])
  '''
  val_model = val_model.to(device)
  val_model.eval()
  print('model evaluation : plot mode start')

  ROOT_DIR = 'realworld'
  val_data = RealClassSegBase(root=ROOT_DIR, transform_tf=True)
  val_data_loader = DataLoader(dataset=val_data, batch_size = 1, drop_last=True)

  for iter, (val_img, val_gt_img) in enumerate(val_data_loader):
    if idx == iter:
      break

  val_img = val_img.to(device)
  val_gt_img = val_gt_img.squeeze(dim=0).squeeze(dim=0).to(device)

  # test image showing
  plt.figure(figsize=(20, 40))
  plt.subplot(1,2,1)
  plt.imshow(val_gt_img.detach().cpu().numpy()) # ???

  # model prediction
  val_seg = val_model(val_img) # 1 C H W 
  val_img_class = torch.argmax(torch.squeeze(val_seg, dim=0), dim=0) # C H W -> HW

  # model prediction to PIL
  val_img_pil = PIL.Image.fromarray(np.uint8(cm.gist_ncar(val_img_class.detach().cpu().numpy()*10)*255))

  # predicted data showing
  plt.subplot(1,2,2)
  plt.imshow(val_img_pil)
  plt.show()
  plt.savefig('voc_train/segmentation_plot.png')

  print('model evaluation : plot mode ended')