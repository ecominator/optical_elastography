#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:24:35 2022

@author: ecominator & Klavdiia & hubernikus
"""

import numpy as np
import cv2 as cv
#import scipy
#from scipy import signal
#import matplotlib
import matplotlib.pyplot as plt
import os
#import shutil
#from skimage.morphology import skeletonize
#import skimage.io
#from sklearn.mixture import GaussianMixture as GMM

from functions import crop_images
from functions import cross_correlation
from functions import elasticity_HT


from wave_detection.line_detection_gmm import WaveDetector


folder_for_frames = 'sim_05perc/'
folder_for_cropped_frames = 'sim_05perc_crop/'
#xcorr_images = 'sim_1perc_xcorr_images/'

os.mkdir(folder_for_frames)           #comment after the first run
os.mkdir(folder_for_cropped_frames)   #comment after the first run

# extracting frames from the video of COMSOL simulated wave
video_name = 'test05percent_G200_cut'
folder_name = 'data/'
path_to_source = folder_name + video_name + '.avi'
cap = cv.VideoCapture(path_to_source)
num_frames = 30
l=0
while l <= num_frames:
    ret, frame = cap.read()
    if not ret: 
        print('No frames grabbed!')
        break
    cv.imwrite(folder_for_frames + f'{l}.jpg', frame)
    l+=1
    
# crop images for faster calculation 
data_path = folder_for_frames
data_save = folder_for_cropped_frames
x_start = 40     #starting from pixel x_start
x_end =90        #ending at pixel x_end
y_start = 40     #starting from pixel y_start
y_end = 90       #ending at pixel y_end
crop_images(data_path, data_save, x_start, x_end, y_start, y_end, color=False)

# choosing ONE pixel for check
pixel = [0, 10]

kernel = 1
threshold_bin = 0
image_1 = cv.imread(data_save  + '0.jpg', cv.IMREAD_GRAYSCALE)
slopes = np.zeros([image_1.shape[0], image_1.shape[1]])

plt.imshow(image_1) #just for debugging
# breakpoint()

for i in range(image_1.shape[0]):
    for j in range(image_1.shape[1]):
        pixel=[i, j]
        print(pixel)
        _, corr_full_y = cross_correlation(num_frames, folder_for_cropped_frames, pixel, kernel, x_disp=False, y_disp=True)
        my_detector = WaveDetector(corr_image=corr_full_y)
        slope = my_detector.do_slope_estimation()
        slopes[i, j] = slope
        
# compute the angles
angs = 180 - np.arctan(slopes)*180/np.pi
angs

mean_slope = np.mean(slopes)
print(mean_slope)

median_slope = np.median(slopes)
print(median_slope)

# plot the result
slopes_norm = cv.normalize(slopes, None, 1, 256, cv.NORM_MINMAX)
plt.imshow(slopes)
plt.title('Slopes for the wave')
plt.colorbar()
plt.show()

# elasticity calculation
dens = 997
# #conversion for experimental videos
# conv_pix = 1/3
# conv_fr = 1/100000

#conversion for simulated videos
conv_pix = (1/1.28)*1e-6
conv_fr = (1/(12500*85))*(85/30)

mu = elasticity_HT(slopes, dens, conv_pix, conv_fr)
mean_elasticity = np.mean(mu)
print(mean_elasticity)