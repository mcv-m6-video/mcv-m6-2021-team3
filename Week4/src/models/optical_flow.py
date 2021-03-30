import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import ssd
from tqdm.auto import tqdm

def block_matching(img1, img2, window_size, shift, stride):
    """
    Block matching method to compute Optical Flow for two consecutive frames.
    :params img1, img2: First and second consecutive frames
    :param window_size: Size of the window to consider around each pixel
    :param shift: Displacement of the window in the other frame
    :param stride: Step size between two estimations
    :return: Optical flow for each direction x,y
    """
    
    # Initialize the matrices.
    vx = np.zeros((img2.shape[:2]))
    '''np.zeros((int((img2.shape[0] - window_size)/stride+1), 
                int((img2.shape[1] - window_size)/stride+1)))'''
    vy = np.zeros((img2.shape[:2]))
    '''np.zeros((int((img2.shape[0] - window_size)/stride+1), 
                int((img2.shape[1] - window_size)/stride+1)))'''
    wh = int(window_size / 2)
    
    # Go through all the blocks.
    tx, ty = 0, 0
    for x in tqdm(np.arange(wh, img2.shape[0] - wh - 1, stride), 'Computing pixel Optical Flow'):
        for y in np.arange(wh, img2.shape[1] - wh - 1, stride):
            nm = img2[x-wh:x+wh+1, y-wh:y+wh+1].flatten()
            
            min_dist = None
            flowx, flowy = 0, 0
            # Compare each block of the next frame to each block from a greater
            # region with the same center in the previous frame.
            for i in np.arange(max(x - shift, wh), min(x + shift + 1, img1.shape[0] - wh - 1)):
                for j in np.arange(max(y - shift, wh), min(y + shift + 1, img1.shape[1] - wh - 1)):
                    om = img1[i-wh:i+wh+1, j-wh:j+wh+1].flatten()
                    
                    # Compute the distance and update minimum.
                    dist = ssd(nm, om)
                    if not min_dist or dist < min_dist:
                        min_dist = dist
                        flowx, flowy = x - i, y - j
            
            # Update the flow field. Note the negative tx and the reversal of
            # flowx and flowy. This is done to provide proper quiver plots, but
            # should be reconsidered when using it.
            #vx[-tx,ty] = flowy
            #vy[-tx,ty] = flowx
            vx[int(x-stride/2):int(x+stride/2), int(y-stride/2):int(y+stride/2)] = flowx
            vy[int(x-stride/2):int(x+stride/2), int(y-stride/2):int(y+stride/2)] = flowy
            
            ty += 1
        tx += 1
        ty = 0
    
    return vx, vy

def movingAverage(curve, radius): 
    window_size = 2 * radius + 1
    # Define the filter 
    f = np.ones(window_size)/window_size 
    # Add padding to the boundaries 
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
    # Apply convolution 
    curve_smoothed = np.convolve(curve_pad, f, mode='same') 
    # Remove padding 
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smoothing(trajectory, smoothing_radius): 
    smoothed_trajectory = np.copy(trajectory) 
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=smoothing_radius)

    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame
