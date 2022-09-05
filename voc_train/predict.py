import torch
from torchvision import transforms

# for plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 20.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
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

  val_data = VOCClassSegBase(root=ROOT_DIR, split='val', transform_tf=True)
  val_data_loader = DataLoader(dataset=val_data, batch_size = 1, drop_last=True)

  val_img, val_gt_img = val_data_loader[idx]

  # test image showing
  plt.figure(figsize=(20, 40))
  plt.subplot(1,2,1)
  plt.imshow(val_gt_img)

  print("val_img.shape : ", val_img.shape)
  print("val_gt_img.shape : ", val_gt_img.shape)

  # test image transform & input to test model
  # test_image = np.array(test_image)
  # test_image = torch.from_numpy(test_image).to(torch.float).permute(2,0,1).to(device)
  # test_image = torch.unsqueeze(test_image, dim=0)

  val_transform = transforms.Compose([
      transforms.Normalize(mean=(0, 0, 0), std=(255., 255., 255.)),
      transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  ])

  # model prediction
  val_seg = val_model(val_transform(val_img))
  val_img_class = torch.argmax(torch.squeeze(val_seg, dim=0), dim=0).cpu()

  # model prediction to PIL
  val_img_pil = PIL.Image.fromarray(
      np.uint8(cm.gist_ncar(val_img_class.detach().numpy()*10)*255)
      )

  # predicted data showing
  plt.subplot(1,2,2)
  plt.imshow(val_img_pil)
  plt.show()
