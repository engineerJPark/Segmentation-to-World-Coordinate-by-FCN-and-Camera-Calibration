import torch
import torch.nn as nn

# Bilinear weights deconvolution Algorithm
def bilinear_kernel_init(Cin, Cout, kernel_size, device):
  factor = (kernel_size + 1) // 2
  if kernel_size % 2 == 1:
    center = factor - 1
  else:
    center = factor - 0.5

  og = (torch.arange(kernel_size).reshape(-1,1), torch.arange(kernel_size).reshape(1,-1))
  filter = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)

  weight = torch.zeros((Cin, Cout, kernel_size, kernel_size))
  weight[range(Cin), range(Cout), :, :] = filter
  return weight.clone().to(device)

class FCN18(nn.Module):
  def __init__(self, class_n, device='cpu'): # class 20 + 1(bakcground)
    super().__init__()
    self.downsample1 = nn.Sequential(
      nn.Conv2d(3,64,3,padding=100),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64,64,3,padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True)
    ).to(device)
    self.downsample2 = nn.Sequential(
      nn.Conv2d(64,128,3,padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128,128,3,padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True)
    ).to(device)
    self.downsample3 = nn.Sequential(
      nn.Conv2d(128,256,3,padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256,256,3,padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256,256,3,padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True)
    ).to(device)
    self.downsample4 = nn.Sequential(
      nn.Conv2d(256,512,3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True)
    ).to(device)
    self.downsample5 = nn.Sequential(
      nn.Conv2d(512,512,3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True)
    ).to(device)

    # fc layers
    self.fc6 = nn.Sequential(
      nn.Conv2d(512, 4096, kernel_size=7, bias=False), 
      nn.ReLU(),
      nn.Dropout2d()
    ).to(device)
    nn.init.xavier_normal_(self.fc6[0].weight)
    self.fc7 = nn.Sequential(
      nn.Conv2d(4096, 4096, kernel_size=1, bias=False), 
      nn.ReLU(),
      nn.Dropout2d()
    ).to(device)
    nn.init.xavier_normal_(self.fc7[0].weight)

    # fc before upsample : to class_n
    self.score_pool3 = nn.Conv2d(256, class_n, kernel_size=1).to(device)
    nn.init.xavier_normal_(self.score_pool3.weight)

    self.score_pool4 = nn.Conv2d(512, class_n, kernel_size=1).to(device)
    nn.init.xavier_normal_(self.score_pool4.weight)

    self.score_final = nn.Conv2d(4096, class_n, kernel_size=1).to(device)
    nn.init.xavier_normal_(self.score_final.weight)
    
    # stride s, padding s/2, kernelsize 2s -> 2 times upsampling for images
    self.upsample_make_16s = nn.ConvTranspose2d(class_n, class_n, kernel_size=4, stride=2, bias=False).to(device) # to 1/16 padding=1,
    self.upsample_make_16s.weight.data.copy_(bilinear_kernel_init(class_n, class_n, 4, device))

    self.upsample_make_8s = nn.ConvTranspose2d(class_n, class_n, kernel_size=4,  stride=2, bias=False).to(device) # to 1/8 padding=1,
    self.upsample_make_8s.weight.data.copy_(bilinear_kernel_init(class_n, class_n, 4, device))
    
    self.upsample_to_score = nn.ConvTranspose2d(class_n, class_n, kernel_size=16,  stride=8, bias=False).to(device) # to 1 padding=4,
    self.upsample_to_score.weight.data.copy_(bilinear_kernel_init(class_n, class_n, 16, device))
    # for param in self.upsample_to_score.parameters(): # freeze the last layer
    #   param.requires_grad = False

  def crop_(self, crop_obj, base_obj, crop=True):
      if crop:
          c = (crop_obj.size()[2] - base_obj.size()[2]) // 2
          crop_obj = crop_obj[:,:,c:c+base_obj.shape[2],c:c+base_obj.shape[3]]
          # crop_obj = F.pad(crop_obj, (-c, -c, -c, -c))
          # crop_obj = torchvision.transforms.CenterCrop((base_obj.shape[2], base_obj.shape[3]))
      return crop_obj

  def forward(self, x):
    input_x = x.clone()
    # print("input shape : ", x.shape)
    
    x = self.downsample1(x) # 1/2
    x = self.downsample2(x) # 1/4
    x = self.downsample3(x) # 1/8
    pool3 = x
    x = self.downsample4(x) # 1/16
    pool4 = x
    x = self.downsample5(x) # 1/32

    x = self.fc6(x) # 1/32
    x = self.fc7(x) # 1/32

    x = self.score_final(x)
    x = self.upsample_make_16s(x)
    score_fc_16s = x # 1/16

    x = self.score_pool4(pool4) # 1/16 
    x = self.crop_(x, score_fc_16s)
    score_pool4c = x # 1/16

    x = score_fc_16s + score_pool4c # 1/16
    x = self.upsample_make_8s(x)
    score_fc_8s = x # 1/8

    x = self.score_pool3(pool3)
    x = self.crop_(x, score_fc_8s)
    score_pool3c = x # 1/8

    x = score_fc_8s + score_pool3c # 1/8
    x = self.upsample_to_score(x)
    x = self.crop_(x, input_x)

    out = x
    # print("out shape : ", out.shape)
    return out