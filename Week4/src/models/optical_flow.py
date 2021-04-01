import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.insert(1, '../')
from utils.metrics import dist_func, bilateral_weights

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



import os
import yaml
import numpy as np
import mxnet as mx

from config.mask_flownet_config import MaskFlownetConfig
sys.path.insert(1, './MaskFlownet')
from MaskFlownet.network import get_pipeline
from MaskFlownet import path, network
from MaskFlownet.network import config

# Functions from predict_new_data.py on MaskFlownet folder
def find_checkpoint(args):
    # find checkpoint
    steps = 0
    checkpoint_str = args.checkpoint

    if checkpoint_str is not None:
    	if ':' in checkpoint_str:
    		prefix, steps = checkpoint_str.split(':')
    	else:
    		prefix = checkpoint_str
    		steps = None
    	log_file, run_id = path.find_log(prefix)	
    	if steps is None:
    		checkpoint, steps = path.find_checkpoints(run_id)[-1]
    	else:
    		checkpoints = path.find_checkpoints(run_id)
    		try:
    			checkpoint, steps = next(filter(lambda t : t[1] == steps, checkpoints))
    		except StopIteration:
    			print('The steps not found in checkpoints', steps, checkpoints)
    			sys.stdout.flush()
    			raise StopIteration
    	steps = int(steps)
    	if args.clear_steps:
    		steps = 0
    	else:
    		_, exp_info = path.read_log(log_file)
    		exp_info = exp_info[-1]
    		for k in args.__dict__:
    			if k in exp_info and k in ('tag',):
    				setattr(args, k, eval(exp_info[k]))
    				print('{}={}, '.format(k, exp_info[k]), end='')
    		print()
    	sys.stdout.flush()
    return checkpoint, steps


def load_model(config_str):
    # load network configuration
    with open(os.path.join('./MaskFlownet', 'network', 'config', config_str)) as f:
    	config =  network.config.Reader(yaml.load(f))
    return config


def instantiate_model(args, gpu_device, config):
    ctx = [mx.cpu()] if gpu_device == '' else [mx.gpu(gpu_id) for gpu_id in map(int, gpu_device.split(','))]
    # initiate
    pipe = get_pipeline(args.network, ctx=ctx, config=config)
    return pipe
    

def load_checkpoint(pipe, config, checkpoint):
    # load parameters from given checkpoint
    print('Load Checkpoint {}'.format(checkpoint))
    sys.stdout.flush()
    network_class = getattr(config.network, 'class').get()
    print('load the weight for the network')
    pipe.load(checkpoint)
    if network_class == 'MaskFlownet':
   		print('fix the weight for the head network')
   		pipe.fix_head()
    sys.stdout.flush()
    return pipe


def predict_image_pair_flow(img1, img2, pipe, resize=None):
    for result in pipe.predict([img1], [img2], batch_size = 1, resize=None):
        flow, occ_mask, warped = result

    return flow, occ_mask, warped
    

def create_video_clip_from_frames(frame_list, fps):
    """ Function takes a list of video frames and puts them together in a sequence"""
    visual_clip = ImageSequenceClip(frame_list, fps=fps) #put frames together using moviepy
    return visual_clip #return the ImageSequenceClip


def predict_video_flow(video_filename, batch_size, resize=None):
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_frames = []
    new_frames = []
    has_frames, frame = cap.read()
    prev_frames.append(frame)
    while True:
        has_frames, frame = cap.read()
        if not has_frames:
            cap.release()
            break
        new_frames.append(frame)
        prev_frames.append(frame)
    del prev_frames[-1] #delete the last frame of the video from prev_frames
    flow_video = [flow for flow, occ_mask, warped in pipe.predict(prev_frames, new_frames, batch_size=batch_size, resize=resize)]
    
    return flow_video, fps


class MaskFlownetOF:

    def __init__(self):
        args = MaskFlownetConfig().get_args()

        checkpoint, steps = find_checkpoint(args)
        config = load_model(args.config)
        pipe = instantiate_model(args, args.gpu_device, config)
        pipe = load_checkpoint(pipe, config, checkpoint)

        self.config = config
        self.pipe = pipe
        self.checkpoint = checkpoint
        self.steps = steps

    def get_optical_flow(self, img1, img2, get_only_flow=True):
        """
            This function uses MaskFlownet to get the 

        """
        
        for result in self.pipe.predict([img1], [img2], batch_size = 1, resize=None):
            flow, occ_mask, warped = result

        if get_only_flow:
            return flow
        else:
            return [flow, occ_mask, warped]


# For testing purposes
if __name__ == '__main__':
    sys.path.insert(1, '../')
    flownet = MaskFlownetOF()