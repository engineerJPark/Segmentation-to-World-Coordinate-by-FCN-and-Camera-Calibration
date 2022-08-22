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
cd /home/kkiruk/catkin_ws/src/js_ws/src
export PYTHONPATH="."
ros_1
python3 test/cam.py
```

# TODO

## open3d Unprojection Algorithm
- https://github.com/isl-org/Open3D/blob/534b04716004f04d5d9e419444c11748f4c21dec/cpp/open3d/t/geometry/PointCloud.cpp#L738
- https://github.com/isl-org/Open3D/blob/ee0fda9bbe8a2981ed04215115a6996df517745d/cpp/open3d/t/geometry/kernel/PointCloud.cpp#L42
- https://github.com/isl-org/Open3D/blob/534b04716004f04d5d9e419444c11748f4c21dec/cpp/open3d/t/geometry/kernel/PointCloudImpl.h#L58
- http://www.open3d.org/docs/0.12.0/cpp_api/classopen3d_1_1core_1_1kernel_1_1_transform_indexer.html#af6039b7c93c60c64dcad9e3deb5ab75a

## April Tag and CheckerBoard
- https://www.google.com/search?q=apriltag&oq=apriltag&aqs=chrome..69i57j0i512l9.3217j0j7&sourceid=chrome&ie=UTF-8
- https://www.google.com/search?q=checkerboard+depth&oq=checkerboard+depth&aqs=chrome..69i57j33i160l3.11338j0j7&sourceid=chrome&ie=UTF-8

## RealSense Py
- https://dev.intelrealsense.com/docs/python2
- https://github.com/IntelRealSense/librealsense    

## other camera calibration method
- https://www.youtube.com/results?search_query=+Nicolai+Nielsen+open3d
- https://github.com/isl-org/Open3D/issues/791
- https://stackoverflow.com/questions/65231665/how-to-do-projection-from-world-coordinates-to-pixel-coordinates
