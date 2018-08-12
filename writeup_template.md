## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test5.jpg "Road Transformed"
[image3]: ./examples/binary_combined.png "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/lane.png "Output"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained `distortion_correction.py`. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

This can also been seen in `advanced_lane_detection.ipynb`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color (hls and hsv) and gradient thresholds (abs_sobel in x and y direction) to generate a binary image (thresholding steps at lines 6 through 130 in `threshold_binary.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

This can also been seen in `advanced_lane_detection.ipynb`.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:




This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 800, 510      | 1000, 0       | 
| 1150, 700     | 1000, 700     |
| 200, 700      | 350, 700      |
| 510, 510      | 350, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I did this in lines 5 through 91 in my code in `line_fit.py`

Firstly, I took a histogram of the bottom half of the image, and find the peaks for the left half and right half. This is the starting point for the left and right lines. 

Then x and y positions were identified of all the nonzero pixels in the image, and and the peak positions then are used to update for each window.  

for each sliding window, repeat the above process. 

After getting all the points from different sliding windows, we use polynomial to fit the points. 


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 136 through 171 in my code in `line_fit.py`

First of all, we define the y value where we want radius of curvature. Meters per pixel in horizontal and vertical dimensions are calculated. 

Next, define the left and right lanes in pixels using the polynomial fitting parameters. 

Then, left curve radius and right curve radius were calculated using the formula illustrated in the lecture. Lane center is also caluclated using the bottom of the left and right lanes. 


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in section 3a. Line Fitting for images in my code in `advanced_lane_detection.ipynb`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a sample of video (./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first problem is that the line detection algorithm may not work well for different types of lines, for example, different thresholds for gradients and colors are required for yellow lines and white lines.

Secondly, when this pipeline is applied to videos, it is not stable for some of the cases. For instances, when the road has a lot of sharp turns, it tends to jump around. In addition, the lighting conditions also seem to affect the effectiveness of the pipeline.




