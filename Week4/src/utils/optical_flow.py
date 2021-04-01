"""

This script contains the necessary functions to get the optical flow 
using the MaskFlownet deep neural network.

"""

import os
import sys
import yaml
import numpy as np
import mxnet as mx

sys.path.insert(1, '../')
from config.config import Config
args = Config().get_args()

sys.path.insert(1, args.mask_flownet_path)
from MaskFlownet.network import get_pipeline
from MaskFlownet import path, network
from MaskFlownet.network import config

del args

# Functions from predict_new_data.py on MaskFlownet folder
def find_checkpoint(checkpoint_str):
    # find checkpoint
    steps = 0
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
    with open(os.path.join('../MaskFlownet', 'network', 'config', config_str)) as f:
    	config =  network.config.Reader(yaml.load(f))
    return config


def instantiate_model(gpu_device, config):
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
    for result in pipe.predict([img1], [img2], batch_size = 1, resize=resize):
        flow, occ_mask, warped = result
        print(type(flow))
        print(flow.shape)

        with open('./flow_output/u_v.pkl', 'wb') as f:
            pickle.dump(flow, f)

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

    def __init__(self, args):
        checkpoint, steps = find_checkpoint(args.checkpoint)
        config = load_model(args.config)
        pipe = instantiate_model(args.gpu_device, config)
        pipe = load_checkpoint(pipe, config, checkpoint)

        self.config = config
        self.pipe = pipe
        self.checkpoint = checkpoint
        self.steps = steps

    def get_optical_flow(self, img1, img2, get_only_flow=True):
        """
            This function uses MaskFlownet to get the 

        """
        
        for result in self.pipe.predict([img1], [img2], batch_size = 1, resize=resize):
            flow, occ_mask, warped = result

        if get_only_flow:
            return flow
        else:
            return [flow, occ_mask, warped]


# For testing purposes
if __name__ == '__main__':
    sys.path.insert(1, '../')
    from config.config import Config
    args = Config().get_args()
    flownet = MaskFlownetOF(args)