import cv2
import png
import numpy as np
import glob
from tqdm.auto import tqdm
from PIL import Image
import os
from os.path import join, exists
from models.optical_flow import block_matching, smoothing, fixBorder
from utils.metrics import compute_MSEN_PEPN
from utils.visualize import OF_hsv_visualize, OF_quiver_visualize
from utils.utils import write_png_flow, pol2cart
import pyflow.pyflow as pyflow

def read_kitti_OF(flow_file):
    """
    Read from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    print("Reading %d x %d flow file in .png format" % (h, w))
    flow = np.zeros((h, w, 3), dtype=np.float64)

    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0

    return flow

class KITTI():
    def __init__(self, data_path, mode, args):
        """
        Init of the KITTI class

        :param args: configuration for the current estimation
        """
        # INPUT PARAMETERS
        self.data_path = data_path
        self.seq_paths = [join(data_path,'data_stereo_flow','training','image_0','000045_{}.png'.format(i)) for i in ['10','11']]
        self.GTOF = read_kitti_OF(join(data_path,'data_stereo_flow','training','flow_noc','000045_10.png'))

        # OF ESTIMATION PARAMETERS
        self.mode = mode
        # For block matching estimation
        self.block_matching = dict(
            window_size = args.window_size,
            shift = args.shift,
            stride = args.stride)
        # For pyflow
        self.pyflow = dict(
            alpha=args.alpha, 
            ratio=args.ratio, 
            minWidth=args.minWidth, 
            nOuterFPIterations=args.nOuterFPIterations,
            nInnerFPIterations=args.nInnerFPIterations,
            nSORIterations=args.nSORIterations,
            colType=args.colType)
       
        # Stabilization BM
        self.frames_paths = glob.glob(join(self.data_path,'video_stabilization','flowers','flowers_01',"*." + args.extension))
        self.frames_paths.sort()
        self.stabilizationBM = args.modelStab

        # Output path to results
        self.output_path = args.output_path
        save_path = join(args.output_path,mode)
        os.makedirs(save_path,exist_ok=True)
        if self.mode in 'block_matching':
            self.png_name = join(save_path,'_'.join(('ws-'+str(args.window_size),
                                                'shift-'+str(args.shift),
                                                'stride-'+str(args.stride)+'.png')))
        elif self.mode in 'pyflow':
            self.png_name = join(save_path,'_'.join((str(args.alpha), str(args.ratio), str(args.minWidth), 
                                                     str(args.nOuterFPIterations), str(args.nInnerFPIterations), 
                                                     str(args.nSORIterations), str(args.colType)+'.png')))


    def estimate_OF(self):
        if exists(self.png_name):
            self.pred_OF = read_kitti_OF(self.png_name)
        else:
            if self.mode in 'block_matching':
                img1, img2 = [cv2.imread(img_path) for img_path in self.seq_paths]
                self.pred_OF = block_matching(img1, img2, self.block_matching['window_size'], 
                                            self.block_matching['shift'], self.block_matching['stride'])
                
            elif self.mode in 'pyflow':
                img1, img2 = [np.array(Image.open(img_path.replace('image','colored'))) for img_path in self.seq_paths]
                img1 = img1.astype(float) / 255.
                img2 = img2.astype(float) / 255.

                u, v, im2W = pyflow.coarse2fine_flow(img1, img2, self.pyflow['alpha'], self.pyflow['ratio'], self.pyflow['minWidth'], 
                                                    self.pyflow['nOuterFPIterations'], self.pyflow['nInnerFPIterations'], 
                                                    self.pyflow['nSORIterations'], self.pyflow['colType'])
                self.pred_OF = np.concatenate((u[..., None], v[..., None], np.ones((u.shape[0],u.shape[1],1))), axis=2)
        
            write_png_flow(self.pred_OF,self.png_name)
    
    def seq_stabilization_BM(self):

        if self.stabilizationBM in 'task21':
        
            #resize to sped up OF
            H, W, C = cv2.imread(self.frames_paths[0]).shape
            dsize = (int(W*0.25), int(H*0.25))

            for f_id, path in enumerate(tqdm(self.frames_paths[0:-1], 'Stabilization in progress')):
                #load pair of frames
                img1 = cv2.imread(self.frames_paths[f_id])
                img1 = cv2.resize(img1, dsize)
                img2 = cv2.imread(self.frames_paths[f_id+1]) 
                img2 = cv2.resize(img2, dsize)
                #OF
                pred_OF = block_matching(img1, img2, self.block_matching['window_size'], self.block_matching['shift'], self.block_matching['stride'])
                mag, ang = cv2.cartToPolar(np.array(pred_OF[0]), np.array(pred_OF[1]))
                #keep the values which is foudn the most for mag and ang
                uniques, counts = np.unique(mag, return_counts=True)
                mc_mag = uniques[counts.argmax()]
                uniques, counts = np.unique(ang, return_counts=True)
                mc_ang = uniques[counts.argmax()]
                u, v = pol2cart(mc_mag, mc_ang)
                #Create an affine transformation for v and u
                affine_H = np.float32([[1, 0, -v],[0,1,-u]])
                #Compute affine transforamtion
                img2_stabilized = cv2.warpAffine(img2,affine_H,(img2.shape[1],img2.shape[0]))
                cv2.imwrite(join(self.output_path,'seq_stabilization','%04d' % f_id +'.png'),img2_stabilized)

        elif self.stabilizationBM in 'task22':
                
            n_frames = int(len(self.frames_paths))
            H, W, C = cv2.imread(self.frames_paths[0]).shape
            dsize = (int(W*0.25), int(H*0.25))
            transforms = np.zeros((n_frames-1, 3), np.float32) 
            prev = cv2.imread(self.frames_paths[0])
            prev = cv2.resize(prev, dsize)
            prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

            for f_id, path in enumerate(tqdm(self.frames_paths[0:-1], 'OF and transformation computation')):

                #load pair of frames
                curr = cv2.imread(self.frames_paths[f_id+1]) 
                curr = cv2.resize(curr, dsize) 
                curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                #OF computation
                prev_pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev, curr, prev_pts, None)
                idx = np.where(status==1)[0]
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]
                assert prev_pts.shape == curr_pts.shape
                    
                #Tranformation estimation
                m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) 
                dx = m[0,2]
                dy = m[1,2]
                da = np.arctan2(m[1,0], m[0,0]) # Extract rotation angle
                transforms[f_id] = [dx,dy,da] 

                prev = curr
                
            # Find the cumulative sum of tranform matrix for each dx,dy and da
            trajectory = np.cumsum(transforms, axis=0) 

            smoothed_trajectory = smoothing(trajectory, 50)
            difference = smoothed_trajectory - trajectory
            transforms_smooth = transforms + difference

            for f_id, path in enumerate(tqdm(self.frames_paths[0:-1], 'Stabilization in process')):

                frame = cv2.imread(self.frames_paths[f_id]) 
                frame = cv2.resize(frame, dsize)
                # Extract transformations from the new transformation array
                dx = transforms_smooth[f_id,0]
                dy = transforms_smooth[f_id,1]
                da = transforms_smooth[f_id,2]
                # Reconstruct transformation matrix accordingly to new values
                m = np.zeros((2,3), np.float32)
                m[0,0] = np.cos(da)
                m[0,1] = -np.sin(da)
                m[1,0] = np.sin(da)
                m[1,1] = np.cos(da)
                m[0,2] = dx
                m[1,2] = dy
                # Apply affine wrapping to the given frame
                frame_stabilized = cv2.warpAffine(frame, m, (H,W))

                # Fix border artifacts
                frame_stabilized = fixBorder(frame_stabilized)
                cv2.imshow('Frame', frame_stabilized)
                cv2.waitKey(1)
                cv2.imwrite(join(self.output_path,'seq_stabilization','%04d' % f_id +'.png'),frame_stabilized)


    def get_MSEN_PEPN(self):
        return compute_MSEN_PEPN(self.GTOF,self.pred_OF)
    
    def visualize(self):
        OF_hsv_visualize(self.pred_OF, self.png_name.replace('.png','_hsv.png'))
        OF_quiver_visualize(cv2.imread(self.seq_paths[1]),self.pred_OF,15,self.png_name.replace('.png','_quiver.png'))
