import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import dist_func, bilateral_weights
from tqdm.auto import tqdm

def block_matching(img1, img2, window_size, shift, stride, metric='ssd', fw_bw='fw', bilateral=None, cv2_method=True):
    """
    Block matching method to compute Optical Flow for two consecutive frames.
    :params img1, img2: First and second consecutive frames
    :param window_size: Size of the window to consider around each pixel
    :param shift: Displacement of the window in the other frame
    :param stride: Step size between two estimations
    :param metric: Metric measurement of difference between windows
    :param fw_bw: Forward or backward computation ('fw', 'bw')
    :return: Optical flow for each direction x,y
    """
    if fw_bw in 'bw':
        img1, img2 = img2, img1
    if bilateral is None:
        weights = None
    
    # Initialize the matrices.
    vx = np.zeros((img2.shape[:2]))
    vy = np.zeros((img2.shape[:2]))
    
    wh = int(window_size / 2)

    #plt.figure()
    
    # Go through all the blocks.
    for x in tqdm(np.arange(wh, img2.shape[0] - wh - 1, stride), 'Computing pixel Optical Flow'):
        for y in np.arange(wh, img2.shape[1] - wh - 1, stride):
            nm = img2[x-wh:x+wh+1, y-wh:y+wh+1]
            
            # Compare each block of the next frame to each block from a greater
            # region with the same center in the previous frame.
            if cv2_method is not None:
                method = eval(cv2_method)

                xmin, xmax = max(x-wh-shift, 0), min(x+wh+1+shift, img1.shape[0]-1)
                ymin, ymax = max(y-wh-shift, 0), min(y+wh+1+shift, img1.shape[1]-1)

                om = img1[xmin:xmax, ymin:ymax]

                # Apply template Matching
                res = cv2.matchTemplate(om,nm,method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                top_left = (ymin+top_left[0], xmin+top_left[1])
                center = (top_left[0] + wh, top_left[1] + wh)
                bottom_right = (top_left[0] + window_size, top_left[1] + window_size)

                flowx, flowy = x - center[1], y - center[0]

                '''plt.subplot(121)
                img2rect = cv2.rectangle(np.expand_dims(img2.copy(),2).repeat(3,2),(y-wh,x-wh), (y+wh+1,x+wh+1), (51,153,255), 2)
                plt.imshow(img2rect)
                plt.subplot(122)
                img1rect = cv2.rectangle(np.expand_dims(img1.copy(),2).repeat(3,2), (ymin,xmin), (ymax,xmax), (255,255,0), 1)
                img1rect = cv2.rectangle(img1rect, top_left, bottom_right, (0,255,0), 1)
                img1rect = cv2.rectangle(img1rect, (y-wh,x-wh), (y+wh+1,x+wh+1), (51,153,255), 2)
                plt.imshow(img1rect)
                plt.pause(.01)'''

            else:
                if bilateral is not None:
                    weights = bilateral_weights(nm,bilateral['gamma_col'],bilateral['gamma_pos'])
                nm = nm.flatten()
                
                min_dist = np.inf
                if metric in 'ncc':
                    min_dist=0
                flowx, flowy = 0, 0

                for i in np.arange(max(x - shift, wh), min(x + shift + 1, img1.shape[0] - wh - 1)):
                    for j in np.arange(max(y - shift, wh), min(y + shift + 1, img1.shape[1] - wh - 1)):
                        om = img1[i-wh:i+wh+1, j-wh:j+wh+1].flatten()
                        '''plt.subplot(121)
                        img2rect = cv2.rectangle(np.expand_dims((img2.copy()[:200,:200]),2).repeat(3,2),(x-wh,y-wh), (x+wh+1,y+wh+1), (51,153,255), 1)
                        plt.imshow(img2rect)
                        plt.subplot(122)
                        img1rect = cv2.rectangle(np.expand_dims((img1.copy()[:200,:200]),2).repeat(3,2),(x-wh,y-wh), (x+wh+1,y+wh+1), (51,153,255), 1)
                        img1rect = cv2.rectangle(img1rect,(i-wh,j-wh), (i+wh+1,j+wh+1), (0,255,0), 1)
                        plt.imshow(img1rect)
                        plt.pause(.1)'''
                        
                        # Compute the distance and update minimum.
                        dist = dist_func(nm, om, metric, weights)
                        if (dist > min_dist if metric in 'ncc' else dist < min_dist):
                            min_dist = dist
                            flowx, flowy = x - i, y - j
            
            # Update the flow field.
            vx[int(x-stride/2):int(x+stride/2), int(y-stride/2):int(y+stride/2)] = flowy
            vy[int(x-stride/2):int(x+stride/2), int(y-stride/2):int(y+stride/2)] = flowx
    
    if fw_bw in 'fw':
        return np.concatenate((vx[..., None], vy[..., None], np.ones((vx.shape[0],vx.shape[1],1))), axis=2)
    elif fw_bw in 'bw':
        return np.concatenate((vx[..., None]*-1, vy[..., None]*-1, np.ones((vx.shape[0],vx.shape[1],1))), axis=2)

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
