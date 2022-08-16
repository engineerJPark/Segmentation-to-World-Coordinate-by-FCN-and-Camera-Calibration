```
ros_1
roslaunch realsense2_camera rs_camera.launch align_depth:=true

export PYTHONPATH="."
ros_1
python3 test/cam.py
```

# TODO

- extrinsic matrix 보정
- point cloud 출력 위치 정확한지 check
- model training 다시 수행