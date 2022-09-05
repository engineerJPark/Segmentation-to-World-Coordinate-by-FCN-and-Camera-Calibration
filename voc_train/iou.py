'''
referenced from
https://stackoverflow.com/a/48383182
'''

from torch.utils.data import DataLoader
from data import VOCClassSegBase

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
    union = float(torch.max((pred_inds + target_inds).sum().item(), 1)[0]) # prevent divided by 0
    
    try:
      ious.append(intersection / union)
    except:
      continue
    
  return np.array(ious), np.array(ious).mean() 


def mean_iou(val_model, device='cpu'):
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
    
    val_seg = val_model(val_img) # 1CHW
    val_seg = torch.squeeze(val_seg, dim=0) # CHW
    val_img_class = torch.argmax(val_seg, dim=0) # HW

    print("val_img_class.shape : ", val_img_class.shape) # test

    _, metric = iou(val_img_class, val_gt_img, 21)
    print("iou of %d th " % (iter + 1), " : ", metric)
    iou_stack += metric

  mean_iou = iou_stack / (iter + 1)
  print("mean_iou : ", mean_iou)