import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from threshold_binary import combined_thres, corners_warp
from line_fit import *
from moviepy.editor import VideoFileClip

class Line():
	def __init__(self, n):
		"""
		n is the window size of the moving average
		"""
		self.n = n
		self.detected = False

		# Polynomial coefficients: x = A*y^2 + B*y + C
		# Each of A, B, C is a "list-queue" with max length n
		self.A = []
		self.B = []
		self.C = []
		# Average of above
		self.A_avg = 0.
		self.B_avg = 0.
		self.C_avg = 0.

	def get_fit(self):
		return (self.A_avg, self.B_avg, self.C_avg)

	def add_fit(self, fit_coeffs):
		"""
		Gets most recent line fit coefficients and updates internal smoothed coefficients
		fit_coeffs is a 3-element list of 2nd-order polynomial coefficients
		"""
		# Coefficient queue full?
		q_full = len(self.A) >= self.n

		# Append line fit coefficients
		self.A.append(fit_coeffs[0])
		self.B.append(fit_coeffs[1])
		self.C.append(fit_coeffs[2])

		# Pop from index 0 if full
		if q_full:
			_ = self.A.pop(0)
			_ = self.B.pop(0)
			_ = self.C.pop(0)

		# Simple average of line coefficients
		self.A_avg = np.mean(self.A)
		self.B_avg = np.mean(self.B)
		self.C_avg = np.mean(self.C)

		return (self.A_avg, self.B_avg, self.C_avg)






# MoviePy video annotation will call this function
def annotate_image(img):
	"""
	Annotate the input image with lane line markings
	Returns annotated image
	"""
	with open('calibrate_camera.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']
	window_size = 5  # how many frames for line smoothing
	left_line = Line(n=window_size)
	right_line = Line(n=window_size)
	detected = False  # did the fast line fit detect the lines?
	left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
	left_lane_inds, right_lane_inds = None, None  # for calculating curvature

	# Undistort, threshold, perspective transform
	undist, sobel_x_binary, sobel_y_binary, sobel_dir_binary, s_binary, v_binary, mag_binary, combined_binary, warped, Minv = combined_thres(img, mtx, dist)
	# Perform polynomial fit
	if not detected:
		# Slow line fit
		ret = fit_lines_first_img(warped)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		# Get moving average of line fit coefficients
		left_fit = left_line.add_fit(left_fit)
		right_fit = right_line.add_fit(right_fit)

		# Calculate curvature
		left_curve, right_curve, center = cal_curv(left_fit, right_fit, warped)

		detected = True  # slow line fit always detects the line

	else:  # implies detected == True
		# Fast line fit
		left_fit = left_line.get_fit()
		right_fit = right_line.get_fit()
		ret = fit_lines_next_img(warped, left_fit, right_fit)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		# Only make updates if we detected lines in current frame
		if ret is not None:
			left_fit = ret['left_fit']
			right_fit = ret['right_fit']
			nonzerox = ret['nonzerox']
			nonzeroy = ret['nonzeroy']
			left_lane_inds = ret['left_lane_inds']
			right_lane_inds = ret['right_lane_inds']

			left_fit = left_line.add_fit(left_fit)
			right_fit = right_line.add_fit(right_fit)
			left_curve, right_curve, center = cal_curv(left_fit, right_fit, warped)
		else:
			detected = False

	# Perform final visualization on top of original undistorted image
	result =  plot_lines(undist, left_fit, right_fit, Minv, left_curve, right_curve)


	return result


def annotate_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':
	# Annotate the video
	annotate_video('project_video.mp4', 'project_video_out.mp4')
