import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import pickle

def camera_calibration(nx, ny, path, show=True):
    images = glob.glob(path)

    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in the image plane

    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) #x,y coordinates

    for fname in images:
        img = mpimg.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            # draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            if show:
                plt.figure(figsize=(10,10))
                fig = plt.figure()
                plt.imshow(img)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    calib_dict = {}
    calib_dict['mtx'] = mtx
    calib_dict['dist'] = dist
    calib_dict['rvecs'] = rvecs
    calib_dict['tvecs'] = tvecs
    return calib_dict

def undistort(img,  mtx, dist):
    """
    Computes the ideal point coordinates from the observed point coordinates.
    """

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

if __name__ == '__main__':
    calib_dict = camera_calibration(nx=9, ny=6, path="camera_cal/calibration*.jpg")
    
    with open('calibrate_camera.p', 'wb') as f:
        pickle.dump(calib_dict, f)
