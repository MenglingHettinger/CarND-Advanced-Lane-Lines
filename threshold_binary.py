import numpy as np
import cv2

from distortion_correction import *

def hls_thres(img, thres=(0, 255), channel='s'):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 's':
        chan = hls[:,:,2]
    elif channel == 'l':
        chan = hls[:,:,1]
    elif channel == 'h':
        chan = hls[:,:,0]
    chan_binary = np.zeros_like(chan)
    chan_binary[(chan > thres[0]) & (chan <= thres[1])] = 1
    return chan_binary

def hsv_thres(img, thres=(0, 255), channel='s'):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if channel == 'v':
        chan = hsv[:,:,2]
    elif channel == 's':
        chan = hsv[:,:,1]
    elif channel == 'h':
        chan = hsv[:,:,0]
    chan_binary = np.zeros_like(chan)
    chan_binary[(chan > thres[0]) & (chan <= thres[1])] = 1
    return chan_binary

def bgr_thres(img, thres=(0, 255), channel='r'):
    if channel == 'r':
        chan = img[:,:,2]
    elif channel == 'g':
        chan = img[:,:,1]
    elif channel == 'b':
        chan = img[:,:,0]
    chan_binary = np.zeros_like(chan)
    chan_binary[(chan > thres[0]) & (chan <= thres[1])] = 1
    return chan_binary

def abs_sobel_thres(img, orient='x', sobel_kernel=3, thres=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel / np.max(abs_sobel))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thres[0]) & (scaled_sobel <= thres[1])] = 1
    return sobel_binary
    
def dir_thres(img, sobel_kernel=3, thres=(0, np.pi/2)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    # Mask based on the threshold
    binary_output = np.zeros_like(dir_grad)
    binary_output[(dir_grad >= thres[0]) & (dir_grad <= thres[1])] = 1
   
    return binary_output

def mag_thres(img, sobel_kernel=3, thres=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thres[0]) & (gradmag <= thres[1])] = 1

    # Return the binary image
    return binary_output

def corners_warp(img):
    """
    Applies a perspective transformation to an image.
    """

    src = np.float32([[800,510],[1150,700],[200,700],[510,510]])
    dst = np.float32([[1000,0],[1000,700],[350,700],[350,0]])

    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    #inverse 
    Minv = cv2.getPerspectiveTransform(dst, src)
    #create a warped image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    unpersp = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)
    return M, Minv, warped, unpersp

def combined_thres(img, mtx, dist):
    #read image
    undist = undistort(img,  mtx, dist)
    undist = cv2.GaussianBlur(undist, (3, 3), 0)
    # Extract lines using gradient thresholding
    sobel_x_binary = abs_sobel_thres(undist,  orient='x', sobel_kernel=3,  thres=(12, 250))
    sobel_y_binary = abs_sobel_thres(undist,  orient='y', sobel_kernel=3,  thres=(12, 250))
    sobel_dir_binary = dir_thres(undist, sobel_kernel=3, thres=(0.7, 1.3))
    mag_binary = mag_thres(undist, sobel_kernel=3,  thres=(50, 250))
    # Extract lines using colors
    s_binary = hls_thres(undist, thres=(100, 255), channel='s')
    v_binary = hsv_thres(undist, thres=(50, 255), channel='v')
    r_binary = bgr_thres(undist, thres=(40, 120), channel='r')

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.uint8(np.dstack((sobel_x_binary, sobel_y_binary, s_binary, mag_binary))*255)
    
    # Combine the two binary thresholds    
    combined_binary = np.zeros_like(sobel_x_binary)
    combined_binary[((sobel_x_binary == 1) & (sobel_y_binary ==1)) | ((s_binary == 1) & (v_binary == 1))] = 1
    #combined_binary[(sobel_x_binary == 1) | (sobel_y_binary == 1) | (r_binary == 1)] = 1
    # Apply perspective transform
    # Define points
    M, Minv, warped, unpersp = corners_warp(combined_binary)
    
    return undist, sobel_x_binary, sobel_y_binary, sobel_dir_binary, s_binary, v_binary, mag_binary, combined_binary, warped, Minv

