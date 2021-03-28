import cv2
import png
import numpy as np
from PIL import Image
import os
from os.path import join, exists
from models.optical_flow import block_matching
from utils.metrics import compute_MSEN_PEPN
from utils.visualize import OF_hsv_visualize, OF_quiver_visualize
from utils.utils import write_png_flow
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

        # Output path to results
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
    
    def get_MSEN_PEPN(self):
        return compute_MSEN_PEPN(self.GTOF,self.pred_OF)
    
    def visualize(self):
        OF_hsv_visualize(self.pred_OF, self.png_name.replace('.png','_hsv.png'))
        OF_quiver_visualize(cv2.imread(self.seq_paths[1]),self.pred_OF,15,self.png_name.replace('.png','_quiver.png'))
