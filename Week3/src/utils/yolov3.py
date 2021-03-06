import os
from PIL import Image
from os.path import join, basename
import glob
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import torch
import cv2

from utils.utils import get_weights

# Import YOLOv3 libraries
from yolov3.models.experimental import attempt_load
from yolov3.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from yolov3.utils.plots import plot_one_box
from yolov3.utils.torch_utils import select_device
from yolov3.utils.datasets import letterbox
from yolov3.train import main as train_yolov3

class UltralyricsYolo():
    def __init__(self,
                 weights=None,
                 device='0',
                 agnostic_nms=False,
                 args=None):
        """ 
        Class initializer
        :param weights: path to weights in .pt
        :param device: device to execute model (cpu or num of gpu)
        :param args: argsparse of options
        :param agnostic_nms: agnostic non-maximum suppresion
        """

        # Initialize
        if weights is None:
            weights=args.model
        set_logging()
        weights = get_weights(weights,'ultralytics')
        
        # Define class position for car
        if args.mode in 'inference':
            classes=[2]
        elif args.mode in 'eval':
            classes=[0]

        if args.mode in 'inference' or args.mode in 'eval':
            self.device = select_device(device)
        
            # Load model
            self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
            self.img_size = check_img_size(args.img_size[0], s=self.model.stride.max())  # check img_size
                
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

            self.conf_thres = args.conf_thres
            self.iou_thres = args.iou_thres
            self.classes = classes
            self.agnostic_nms = agnostic_nms
            
            img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
            _ = self.model(img) if self.device.type != 'cpu' else None  # run once

        elif args.mode in 'train':
            self.weights = weights
            self.args = args

    def predict(self, img_path):
        """
            Passes the image either through a pre-trained net on the COCO network or a fine-tuned
            image on the AICity dataset.
            :param img_path: image path
            :return list of bboxes [(xmin, ymin, xmax, ymax), confidence]
        """

        img = cv2.imread(img_path)
        img0 = img.copy()
        
        #This happens inside datasets
        # Convert
        img = letterbox(img, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        #this happens on detect
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        pred = [d.cpu().detach().numpy() for d in pred if d is not None]
        pred = pred[0] if len(pred) else pred
        
        pred = [[[x1, y1, x2, y2],conf] for x1, y1, x2, y2, conf, clss in pred]

        return pred

    def train(self, kfold=None):
        """
            Train model.
            :param kfold: number of folder used for training. If None, no cross-validation.
        """        
        if kfold is not None:
            train_yolov3(self.weights, self.args, kfold)
        else:
            train_yolov3(self.weights, self.args)       


def gt_multi_txt(path, bboxes):
    """
        Convert bboxes in AICity format to YOLOv3 utralytics format.
        :param bboxes: list of bboxes in format (xmin, ymin, xmax, ymax)
    """    
    
    W, H = Image.open(path).size

    lines_out=[]
    for obj_info in bboxes:
        label = 0 #obj_info['name']
        xmin, ymin, xmax, ymax = obj_info['bbox']

        cx = '%.3f' % np.clip(((xmax+xmin)/2)/W,0,1)
        cy = '%.3f' % np.clip(((ymax+ymin)/2)/H,0,1)
        w = '%.3f' % np.clip((xmax-xmin)/W ,0,1)
        h = '%.3f' % np.clip((ymax-ymin)/H ,0,1)

        lines_out.append(' '.join([str(label),cx,cy,w,h,'\n']))

    return lines_out


def to_yolov3(data, gt_bboxes, mode, save_path='yolov3_data'):
    """
        Convert AICity data format to YOLOv3 utralytics format.
        :param data: paths to train and validation files
        :param gt_bboxes: dict of ground truth detections
        :param mode: sorted or shuffled data
        :param save_path: path to store yolov3 ultralytic data format
    """        
    save_path = join(save_path,mode)
    data_path = join(os.getcwd(),save_path,'data')
    
    if len(data)==1:    
        if os.path.exists(data_path):
            if len(glob.glob(data_path+'/*.*')) == 2*sum([len(d) for _,d in data.items()]):
                print('Data already in YOLOv3 format!')
                return

        os.makedirs(data_path,exist_ok=True)

        for split, split_data in data[0].items():
            files = []
            for path in tqdm(split_data,'Preparing '+split+' data for YOLOv3'):
                # Convert to yolov3 format
                frame_id = basename(path).split('.')[0]
                lines_out = gt_multi_txt(path, gt_bboxes[frame_id])

                # Write/save files
                file_out = open(join(data_path,frame_id+'.txt'), 'w')
                file_out.writelines(lines_out)
                new_path = join(data_path,frame_id+'.jpg')
                files.append(new_path+'\n')
                copyfile(path, new_path)

            split_txt = open(join(os.getcwd(),save_path,split+'.txt'), 'w')
            split_txt.writelines(files)
    else:
        for k, fold in enumerate(data):
            for split, split_data in fold.items():
                files = []
                for path in tqdm(split_data,'Preparing '+split+' data for YOLOv3'):
                    # Convert to yolov3 format
                    frame_id = basename(path).split('.')[0]
                    new_path = join(data_path,frame_id+'.jpg')
                    files.append(new_path+'\n')
                    
                # Write files
                os.makedirs(join(save_path,str(len(data))),exist_ok=True)
                split_txt = open(join(save_path,str(len(data)),split+'_'+str(k)+'.txt'), 'w')
                split_txt.writelines(files)

