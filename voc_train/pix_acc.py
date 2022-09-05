from torch.utils.data import DataLoader
from data import VOCClassSegBase

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


def mean_foreground_pixel_acc(val_model, device='cpu'):
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
    
    val_seg = val_model(val_img) # 1CHW
    val_seg = torch.squeeze(val_seg, dim=0) # CHW
    val_img_class = torch.argmax(val_seg, dim=0) # HW

    print("val_img_class.shape : ", val_img_class.shape) # test

    _, metric = foreground_pixel_acc(val_img_class, val_gt_img, 21)
    print("iou of %d th " % (iter + 1), " : ", metric)
    acc_stack += metric

  acc = acc_stack / (iter + 1)
  print("pixel acc : ", acc)