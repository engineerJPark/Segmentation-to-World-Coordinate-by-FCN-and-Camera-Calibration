# FCN Implementation for 2022 Summer KAsimov Grasping PJT

[Paper Link](https://arxiv.org/abs/1411.4038)

- Backbone : ResNet 18 pretrained on ImageNet
- Pretrained on PASCAL VOC 2012, 21-class
- 

# ROS commands to run this program

realsense dependency : Installation link
ros dependency : Installation link

```
source /opt/ros/noetic/setup.bash && source ~/catkin_ws/devel/setup.bash
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

```
source /opt/ros/noetic/setup.bash && source ~/catkin_ws/devel/setup.bash
cd src
python3 cam.py
```

# Process




# Reference

Intrinsic Extrinsic calibration link

