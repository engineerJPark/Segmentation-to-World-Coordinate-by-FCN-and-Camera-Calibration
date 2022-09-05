import torch
from torch.utils.data import DataLoader
from data import VOCClassSegBase

# for plotting
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (10.0, 20.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
from matplotlib import cm
import PIL

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
  plt.imshow(val_gt_img.numpy().cpu()) # ???

  # print("val_img.shape : ", val_img.shape)
  # print("val_gt_img.shape : ", val_gt_img.shape)

  # model prediction
  val_seg = val_model(val_img) # 1 C H W 
  val_img_class = torch.argmax(torch.squeeze(val_seg, dim=0), dim=0).cpu() # C H W -> HW

  # model prediction to PIL
  val_img_pil = PIL.Image.fromarray(
      np.uint8(cm.gist_ncar(val_img_class.detach().numpy()*10)*255)
      )

  # predicted data showing
  plt.subplot(1,2,2)
  plt.imshow(val_img_pil)
  plt.show()
