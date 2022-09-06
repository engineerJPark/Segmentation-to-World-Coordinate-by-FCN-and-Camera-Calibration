import numpy as np

from torch.utils.data import DataLoader
from data import VOCClassSegBase
import torch
import numpy as np

# for plotting
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (10.0, 20.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
from matplotlib import cm
import PIL


def _fast_hist(label_true, label_pred, n_class):
  mask = (label_true >= 0) & (label_true < n_class)
  hist = np.bincount(
      n_class * label_true[mask].astype(int) +
      label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
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
  
  hist = np.zeros((n_class, n_class))

  # model prediction
  for iter, (label_preds, label_trues) in enumerate(val_data_loader):

    label_preds = label_preds.to(device)
    label_trues = label_trues.squeeze(dim=0).squeeze(dim=0).to(device)
    
    label_seg = val_model(label_preds) # 1CHW
    label_seg = torch.squeeze(label_seg, dim=0) # CHW
    label_preds = torch.argmax(label_seg, dim=0) # HW
    hist += _fast_hist(label_trues.detach().cpu().numpy().flatten(), label_preds.detach().cpu().numpy().flatten(), n_class)


  acc = np.diag(hist).sum() / hist.sum()
  with np.errstate(divide='ignore', invalid='ignore'):
      acc_cls = np.diag(hist) / hist.sum(axis=1)
  acc_cls = np.nanmean(acc_cls)
  with np.errstate(divide='ignore', invalid='ignore'):
      iu = np.diag(hist) / (
          hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
      )
  mean_iu = np.nanmean(iu)
  freq = hist.sum(axis=1) / hist.sum()
  fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
  if verbose == True:
    print("acc : ", acc.item())
    print("acc_cls : ", acc_cls.item())
    print("mean_iu : ", mean_iu.item())
    print("fwavacc : ", fwavacc.item())
  return acc, acc_cls, mean_iu, fwavacc


def foreground_pixel_acc(pred, gt, class_num):
  true_positive_stack = 0
  all_stack = 0

  # Ignore IoU for background class ("0")
  # This goes from 1:n_classes-1 -> class "0" is ignored
  for class_i in range(1, class_num+1):
    true_positive = (pred == class_i) * (gt == class_i)
    all = (gt == class_i)

    true_positive_stack += true_positive.sum()
    all_stack += all.sum()

  return true_positive_stack / all_stack


def mean_foreground_pixel_acc(val_model, device='cpu', verbose=True):
  val_model = val_model.to(device)
  val_model.eval()
  print('model evaluation start')

  # for test data
  ROOT_DIR = 'voc_train/voc_data/'
  val_data = VOCClassSegBase(root=ROOT_DIR, split='val', transform_tf=True)
  val_data_loader = DataLoader(dataset=val_data, batch_size = 1, drop_last=True)
  
  acc_stack = 0

  # model prediction
  for iter, (val_img, val_gt_img) in enumerate(val_data_loader):

    val_img = val_img.to(device)
    val_gt_img = val_gt_img.squeeze(dim=0).squeeze(dim=0).to(device)
    
    val_seg = val_model(val_img) # 1CHW
    val_seg = torch.squeeze(val_seg, dim=0) # CHW
    val_img_class = torch.argmax(val_seg, dim=0) # HW

    # print("val_img_class.shape : ", val_img_class.shape) # test

    metric = foreground_pixel_acc(val_img_class, val_gt_img, 21)
    acc_stack += metric
    if verbose==True:
      print("acc of %d th " % (iter + 1), " : ", metric.item())

  acc = acc_stack / (iter + 1)
  print("pixel acc : ", acc.item())


'''
referenced from
https://stackoverflow.com/a/48383182
'''

# IoU function
def iou(pred, target, n_classes):
  ious = []
  pred = pred.reshape(-1, )
  target = target.reshape(-1, )

  # Ignore IoU for background class ("0")
  # This goes from 1:n_classes-1 -> class "0" is ignored
  for cls in range(1, n_classes):  
    pred_inds = pred == cls
    target_inds = target == cls
    
    intersection = float((pred_inds * target_inds).sum().item())
    union = float((pred_inds + target_inds).sum().item()) 
    if union <= 1 : union = 1 # prevent divided by 0
    
    try:
      ious.append(intersection / union)
    except:
      continue
    
  return np.array(ious), np.array(ious).mean() 


def mean_iou(val_model, device='cpu', verbose=True):
  val_model = val_model.to(device)
  val_model.eval()
  print('model evaluation start')

  # for test data
  ROOT_DIR = 'voc_train/voc_data/'
  val_data = VOCClassSegBase(root=ROOT_DIR, split='val', transform_tf=True)
  val_data_loader = DataLoader(dataset=val_data, batch_size = 1, drop_last=True)
  
  iou_stack = 0

  # model prediction
  for iter, (val_img, val_gt_img) in enumerate(val_data_loader):

    val_img = val_img.to(device)
    val_gt_img = val_gt_img.squeeze(dim=0).squeeze(dim=0).to(device)

    val_seg = val_model(val_img) # 1CHW
    val_seg = torch.squeeze(val_seg, dim=0) # CHW
    val_img_class = torch.argmax(val_seg, dim=0) # HW

    # print("val_img_class.shape : ", val_img_class.shape) # test

    _, metric = iou(val_img_class, val_gt_img, 21)
    iou_stack += metric
    if verbose==True:
      print("iou of %d th " % (iter + 1), " : ", metric.item())

  mean_iou = iou_stack / (iter + 1)
  print("mean_iou : ", mean_iou.item())


def seg_plot(val_model, idx, device='cpu'):
  '''
  val_model = FCN18(21).to(device)
  PATH = 'fcn_model/model_9_2_22_52_53'
  checkpoint = torch.load(PATH)
  val_model.load_state_dict(checkpoint['model_state_dict'])
  '''
  val_model = val_model.to(device)
  val_model.eval()
  print('model evaluation start')

  ROOT_DIR = 'voc_train/voc_data/'
  val_data = VOCClassSegBase(root=ROOT_DIR, split='val', transform_tf=True)
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

  # print("val_img.shape : ", val_img.shape)
  # print("val_gt_img.shape : ", val_gt_img.shape)

  # model prediction
  val_seg = val_model(val_img) # 1 C H W 
  val_img_class = torch.argmax(torch.squeeze(val_seg, dim=0), dim=0) # C H W -> HW

  # model prediction to PIL
  val_img_pil = PIL.Image.fromarray(
      np.uint8(cm.gist_ncar(val_img_class.detach().cpu().numpy()*10)*255)
      )

  # predicted data showing
  plt.subplot(1,2,2)
  plt.imshow(val_img_pil)
  plt.show()
