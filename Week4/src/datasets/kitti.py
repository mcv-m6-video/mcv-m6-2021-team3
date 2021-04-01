import cv2
import numpy as np
import glob
from tqdm.auto import tqdm
from PIL import Image
import os
from os.path import join, exists
from models.optical_flow import block_matching, smoothing, fixBorder
from utils.metrics import compute_MSEN_PEPN
from utils.visualize import OF_hsv_visualize, OF_quiver_visualize, OF_plot_metrics
from utils.utils import write_png_flow, pol2cart, read_kitti_OF
import pyflow.pyflow as pyflow

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
            stride = args.stride,
            cv2_method = args.cv2_method)
        self.metric = args.dist_func
        if args.bilateral[0] is not None:
            self.bilateral = dict(
                gamma_col = args.bilateral[0],
                gamma_pos = args.bilateral[1])
        else:
            self.bilateral = None

        # For pyflow
        self.pyflow = dict(
            alpha=args.alpha, 
            ratio=args.ratio, 
            minWidth=args.minWidth, 
            nOuterFPIterations=args.nOuterFPIterations,
            nInnerFPIterations=args.nInnerFPIterations,
            nSORIterations=args.nSORIterations,
            colType=args.colType)

        # Output path to results
        self.output_path = args.output_path
        save_path = join(args.output_path,mode)
        os.makedirs(save_path,exist_ok=True)
        if self.mode in 'block_matching':
            self.png_name = join(save_path,'_'.join((args.dist_func, str(args.bilateral[0]), str(args.bilateral[1]),
                                                     'ws-'+str(args.window_size),
                                                     'shift-'+str(args.shift),
                                                     'stride-'+str(args.stride),
                                                     str(args.cv2_method)+'.png')))
        elif self.mode in 'pyflow':
            self.png_name = join(save_path,'_'.join((str(args.alpha), str(args.ratio), str(args.minWidth), 
                                                     str(args.nOuterFPIterations), str(args.nInnerFPIterations), 
                                                     str(args.nSORIterations), str(args.colType)+'.png')))

        elif self.mode in 'mask_flownet':
            self.png_name = join(save_path,'_mask_flownet_'+'.png')


    def estimate_OF(self):
        if exists(self.png_name):
            self.pred_OF = read_kitti_OF(self.png_name)
            self.pred_OF[:,:,:2] = self.pred_OF[:,:,:2]
        else:
            if self.mode in 'block_matching':
                img1, img2 = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) for img_path in self.seq_paths]
                self.pred_OF = block_matching(img1, img2, self.block_matching['window_size'], 
                                            self.block_matching['shift'], self.block_matching['stride'],
                                            metric=self.metric,bilateral=self.bilateral, cv2_method=self.block_matching['cv2_method'])
                
            elif self.mode in 'pyflow':
                img1, img2 = [np.array(Image.open(img_path.replace('image','colored'))) for img_path in self.seq_paths]
                img1 = img1.astype(float) / 255.
                img2 = img2.astype(float) / 255.

                u, v, im2W = pyflow.coarse2fine_flow(img1, img2, self.pyflow['alpha'], self.pyflow['ratio'], self.pyflow['minWidth'], 
                                                    self.pyflow['nOuterFPIterations'], self.pyflow['nInnerFPIterations'], 
                                                    self.pyflow['nSORIterations'], self.pyflow['colType'])
                self.pred_OF = np.concatenate((u[..., None], v[..., None], np.ones((u.shape[0],u.shape[1],1))), axis=2)
            
            write_png_flow(self.pred_OF,self.png_name)

    def get_MSEN_PEPN(self):
        msen, pepn, error = compute_MSEN_PEPN(self.GTOF,self.pred_OF)
        self.vec_error = error
        return msen, pepn
    
    def visualize(self):
        occluded_idx = self.GTOF[:, :, 2] == 0
        self.GTOF[occluded_idx, :] = 0
        OF_hsv_visualize(self.pred_OF, self.png_name.replace('.png','_hsv.png'))
        OF_quiver_visualize(cv2.imread(self.seq_paths[0]),self.pred_OF,15,self.png_name.replace('.png','_quiver.png'))
        OF_plot_metrics(self.vec_error,self.png_name.replace('.png','_error.png'))
