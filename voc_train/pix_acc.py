  def foreground_pixel_acc(pred, gt, class_num):
    true_positive_stack = 0
    all_stack = 0
    for class_i in range(1, class_num+1):
      true_positive = (pred == class_i) * (gt == class_i)
      all = (gt == class_i)

      true_positive_stack += true_positive.sum()
      all_stack += all.sum()

    return true_positive_stack / all_stack

  ##################################################################

  # foreground pixel accuracy

  with open(os.path.join(ROOT_DIR, "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"), 'r') as f:
    lines = f.readlines()
  for i in range(len(lines)):
    lines[i] =  lines[i].strip('\n')

  iter = 0
  acc_stack = 0

  for idx in range(len(lines)):
    test_jpg_path = lines[idx] + '.jpg'
    test_png_path = lines[idx] + '.png'
    test_image = PIL.Image.open(os.path.join(ROOT_DIR, 'VOCdevkit/VOC2012', "JPEGImages", test_jpg_path))
    test_gt_image = PIL.Image.open(os.path.join(ROOT_DIR, 'VOCdevkit/VOC2012', "SegmentationObject", test_png_path))

    # test image transform & input to test model
    test_image = np.array(test_image)
    test_image = torch.from_numpy(test_image).to(torch.float).permute(2,0,1).to(device)
    ori_x, ori_y = test_image.shape[1], test_image.shape[2]
    test_image = torch.unsqueeze(test_image, dim=0)

    test_transform = transforms.Compose([
        transforms.Resize((320,320), interpolation=InterpolationMode.NEAREST),
        transforms.Normalize(mean=(0, 0, 0), std=(255., 255., 255.)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return_transform = transforms.Compose([
        transforms.Resize((ori_x, ori_y), interpolation=InterpolationMode.NEAREST),
    ])

    test_seg = test_model(test_transform(test_image))
    test_seg = return_transform(test_seg)
    # test_seg[test_seg <= 8] = 0 # Thresholdings
    test_seg = torch.squeeze(test_seg, dim=0)

    # model prediction
    test_image_channel_idx = torch.argmax(test_seg, dim=0).cpu()

    # ground truth image getting
    test_gt_image = np.array(test_gt_image)
    test_gt_image = torch.from_numpy(test_gt_image).to(torch.int)

    iter += 1
    metric = foreground_pixel_acc(test_image_channel_idx, test_gt_image, 21)
    print("foreground pixel acc of %d th " % (iter), " : ", metric)
    acc_stack += metric

  acc = acc_stack / iter
  print("acc : ", acc.item())