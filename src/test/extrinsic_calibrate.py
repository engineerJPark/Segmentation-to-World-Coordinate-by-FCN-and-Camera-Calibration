'''
from
~~~~~~~~~~~~~
'''


import cv2
import numpy as np
import os
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 


# Defining the world coordinates for 3D points
objpoints = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objpoints[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objpoints = objpoints * 2
objpoints[0, :, 1] = objpoints[0, :, 1] * (-1)

#####################################################

# Extracting path of individual image stored in a given directory
img = cv2.imread('/home/kkiruk/catkin_ws/src/js_ws/src/dataset/calibration/extrinsic/rs_image_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

if ret == True:
    imgpoints = cv2.cornerSubPix(gray, corners, (5,5),(-1,-1), criteria)
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, imgpoints, ret)
    imgpoints = np.squeeze(imgpoints, axis=1)
        
print("objpoints : \n")
print(objpoints)

print("imgpoints shape : \n")
print(imgpoints.shape)
print("imgpoints : \n")
print(imgpoints)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ######################################################

# camera intrinsic
cameraMatrix = np.array(
      [
        [623.31476768, 0., 269.87277202],
        [0., 613.62125703 ,237.91605748],
        [0., 0., 1.]
      ]
    )

# distortion parameter
dist_coeffs = np.array([-0.07379347, 0.66942174, -0.00238366, -0.02229801, -1.27933461])

# getting extrinsic parameter
retval, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, cameraMatrix, dist_coeffs, rvec=None, tvec=None, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE)

R = cv2.Rodrigues(rvec)
t= tvec

# print(rvec)
print(R)
print("\n")
print(t)