# FCN Implementation for 2022 Summer KAsimov Grasping PJT

[Paper Link](https://arxiv.org/abs/1411.4038)

- Backbone : ResNet 18 pretrained on ImageNet
- Tested on PASCAL VOC 2012
- Need to get train set from OpenSet V6 by [this repo](https://github.com/engineerJPark/OpenImageSet2VOC)

# ROS commands to run this program

```
ros_1
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

```
export PYTHONPATH="."
ros_1
python3 test/cam.py
```

# TODO

- extrinsic matrix 보정
- point cloud 출력 위치 정확한지 check
- model training 다시 수행