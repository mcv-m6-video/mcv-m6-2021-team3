from __future__ import print_function
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
#matplotlib inline
from IPython import display as dp
import numpy as np
import cv2
import json
import os
from skimage import io
from os.path import join, basename, exists
import glob
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

from utils.sort import Sort

from utils.metrics import voc_eval, compute_iou, compute_centroid, compute_total_miou, interpolate_bb
from utils.utils import write_json_file, read_json_file, frame_id, dict_to_list_tracking
from utils.visualize import visualize_background_iou

#from utils.detect2 import Detect2, to_detectron2
from utils.tf_models import TFModel
from utils.yolov3 import UltralyricsYolo, to_yolov3




def load_text(text_dir, text_name):
    """
    Parses an annotations TXT file
    :param xml_dir: dir where the file is stored
    :param xml_name: name of the file to parse
    :return: a dictionary with the data parsed
    """
    with open(join(text_dir, text_name), 'r') as f:
        txt = f.readlines()

    annot = {}
    for frame in txt:
        frame_id, _, xmin, ymin, width, height, conf, _, _, _ = list(map(float, (frame.split('\n')[0]).split(',')))
        update_data(annot, frame_id, xmin, ymin, xmin + width, ymin + height, conf)
    return annot


def load_xml(xml_dir, xml_name, ignore_parked=True):
    """
    Parses an annotations XML file
    :param xml_dir: dir where the file is stored
    :param xml_name: name of the file to parse
    :return: a dictionary with the data parsed
    """
    tree = ET.parse(join(xml_dir, xml_name))
    root = tree.getroot()
    annot = {}

    for child in root:
        if child.tag in 'track':
            if child.attrib['label'] not in 'car':
                continue
            obj_id = int(child.attrib['id'])
            for bbox in child.getchildren():
                '''if bbox.getchildren()[0].text in 'true':
                    continue'''
                frame_id, xmin, ymin, xmax, ymax, _, _, _ = list(map(float, ([v for k, v in bbox.attrib.items()])))
                update_data(annot, int(frame_id) + 1, xmin, ymin, xmax, ymax, 1., obj_id)

    return annot

def load_annot(annot_dir, name, ignore_parked=True):
    """
    Loads annotations in XML format or TXT
    :param annot_dir: dir containing the annotations
    :param name: name of the file to load
    :return: the loaded annotations
    """
    if name.endswith('txt'):
        annot = load_text(annot_dir, name)
    elif name.endswith('xml'):
        annot = load_xml(annot_dir, name, ignore_parked)
    else:
        assert 'Not supported annotation format ' + name.split('.')[-1]

    return annot

def update_data(annot, frame_id, xmin, ymin, xmax, ymax, conf, obj_id=0):
    """
    Updates the annotations dict with by adding the desired data to it
    :param annot: annotation dict
    :param frame_id: id of the framed added
    :param xmin: min position on the x axis of the bbox
    :param ymin: min position on the y axis of the bbox
    :param xmax: max position on the x axis of the bbox
    :param ymax: max position on the y axis of the bbox
    :param conf: confidence
    :return: the updated dictionary
    """

    frame_name = '%04d' % int(frame_id)
    obj_info = dict(
        name='car',
        obj_id=obj_id,
        bbox=list(map(float, [xmin, ymin, xmax, ymax])),
        confidence=float(conf)
    )

    if frame_name not in annot.keys():
        annot.update({frame_name: [obj_info]})
    else:
        annot[frame_name].append(obj_info)

    return annot

class AICity:
    """
    This class contains all the logic of the background estimation process for the AICity dataset
    """

    def __init__(self, args):
        """
        Init of the AICity class

        :param args: configuration for the current estimation
        """
        self.options = args

        # INPUT PARAMETERS
        self.data_path = args.data_path
        self.img_size = args.img_size
        self.split = args.split
        self.task = args.task
        self.model = args.model
        self.framework = args.framework
        self.mode = args.mode

        # Load detections
        self.gt_bboxes = load_annot(args.gt_path, 'ai_challenge_s03_c010-full_annotation.xml')
        infer_path = join(self.options.output_path,self.mode+'/') +'_'.join((self.model, self.framework+'.json'))
        if exists(infer_path):
            self.det_bboxes = read_json_file(infer_path)
        else:
            self.det_bboxes = {}

        # Load frame paths and filter by gt
        self.frames_paths = glob.glob(join(self.data_path, "*." + args.extension))
        self.frames_paths = [path for frame_id,_ in self.gt_bboxes.items() for path in self.frames_paths if frame_id in path]
        self.frames_paths.sort()
        '''
        if args.test_mode:
            self.frames_paths = self.frames_paths[0:int(len(self.frames_paths) / 10)]
        '''
        self.data = {'train':[], 'val':[]}

        # OUTPUT PARAMETERS
        self.output_path = args.output_path
        self.save_json = args.save_json
        self.view_img = args.view_img
        self.save_img = args.save_img

        
    def __len__(self):
        return len(self.frames_paths)

    def train_val_split(self):
        """
        Apply random split to specific propotion of the dataset (split).

        """
        if self.split[0] in 'rand':
            train, val, _, _ = train_test_split(np.array(self.frames_paths), np.empty((len(self),)),
                                                test_size=1-self.split[1], random_state=0)
            self.data['train'] = train.tolist()
            self.data['val'] = val.tolist()

        elif self.split[0] in 'first_frames':
            self.data['train'] = self.frames_paths[:int(len(self)*self.split[1])]
            self.data['val'] = self.frames_paths[int(len(self)*self.split[1]):]
    
    def data_to_model(self):
        if self.framework in 'ultralytics':
            to_yolov3(self.data, self.gt_bboxes, self.split[0])
        '''elif self.framework in 'detectron2':
            to_detectron2(self.data, self.gt_bboxes)'''

    
    def inference(self, weights=None):
        if self.framework in 'ultralytics':
            model = UltralyricsYolo(weights, args=self.options)
        
        elif self.framework in 'tensorflow':
            model = TFModel(self.options, self.model)

        '''elif self.framework in 'detectron2':
            model = Detect2(self.model)'''
                
        for file_name in tqdm(self.frames_paths, 'Model predictions ({}, {})'.format(self.model, self.framework)):
            pred = model.predict(file_name)
            frame_id = file_name[-8:-4]
            for (bbox), conf in pred:
                self.det_bboxes = update_data(self.det_bboxes, frame_id, *bbox, conf)
        
        if self.save_json:
            save_path = join(self.options.output_path, self.mode+'/')
            os.makedirs(save_path, exist_ok=True)
            write_json_file(self.det_bboxes,save_path+'_'.join((self.model, self.framework+'.json')))

    def train_split(self, split=0):
        """
        Apply random split to specific propotion of the train set.

        Args:
            split (float): proportion of the train set that will be used.
        """
        self.train_dataset = random.choices(self.dataset_train,k=int(len(self.dataset_train)*split))
   

    def get_mAP(self, test=False):
        """
        Estimats the mAP using the VOC evaluation

        :return: map of all estimated frames
        """
        if self.mode == 'eval':
            mAP50 = voc_eval(self.gt_bboxes, self.data['val'], self.det_bboxes)[2]
            mAP70 = voc_eval(self.gt_bboxes, self.data['val'], self.det_bboxes, use_07_metric=True)[2]
        else:
            mAP50 = voc_eval(self.gt_bboxes, self.frames_paths, self.det_bboxes)[2]
            mAP70 = voc_eval(self.gt_bboxes, self.frames_paths, self.det_bboxes, use_07_metric=True)[2]
        return mAP50, mAP70

    def get_mIoU(self):
        """
        Estimates the mIoU

        :return: iou of all estimated frames
        """
        return \
            compute_total_miou(self.gt_bboxes, self.det_bboxes, self.frames_paths)

    def save_results(self, name_json):
        """
        Saves results to a JSON file

        :param name_json: name of the json file
        """
        write_json_file(self.det_bboxes, name_json)

    def visualize_task(self):
        """
        Creates plots for a given frame and bbox estimation
        """
        visualize_background_iou(self.data, None, self.gt_bboxes, self.det_bboxes, self.framework, self.model, self.options.output_path, self.mode)
  
    def return_bb(self, frame, bb_id):
        for bbox in self.det_bboxes[frame_id(frame)]:
            if bbox['obj_id'] == bb_id:
                return bbox['bbox']
        return None

    
    def compute_tracking_overlapping(self, threshold = 0.5, interpolate = True, remove_noise = True):
       
        id_seq = {}
        #not assuming any order
        start_frame = int(min(self.det_bboxes.keys()))
        num_frames = int(max(self.det_bboxes.keys())) - start_frame + 1

        #init the tracking by  using the first frame 
        for value, detection in enumerate(self.det_bboxes[frame_id(start_frame)]):
            detection['obj_id'] = value
            id_seq.update({value: True})
        #now, frame by frame, no assuming order nor continuity
        for i in range(start_frame, num_frames):
            #init
            id_seq = {frame_id: False for frame_id in id_seq}
            
            for detection in self.det_bboxes[frame_id(i+1)]:
                active_frame = i 
                bbox_matched = False
                #if there is no good match on previous frame, check n-1 up to n=5
                while (bbox_matched == False) and (active_frame >= start_frame) and ((i - active_frame)<5):
                    candidates = [candidate['bbox'] for candidate in self.det_bboxes[frame_id(active_frame)]]               
                    #compare with all detections in previous frame
                    #best match
                    iou = compute_iou(np.array(candidates), np.array(detection['bbox']))
                    while np.max(iou) > threshold:
                        #candidate found, check if free
                        matching_id = self.det_bboxes[frame_id(active_frame)][np.argmax(iou)]['obj_id']
                        if id_seq[matching_id] == False:
                            detection['obj_id'] = matching_id
                            bbox_matched = True
                            #interpolate bboxes 
                            if i != active_frame and interpolate:
                                frames_skip = i - active_frame
                                for j in range(frames_skip):
                                    new_bb = interpolate_bb(self.return_bb((active_frame+j), matching_id), detection['bbox'],frames_skip-j+1)
                                    update_data(self.det_bboxes, (active_frame+1+j),*new_bb,0,matching_id)
                            break
                        else: #try next best match
                            iou[np.argmax(iou)] = 0
                    active_frame = active_frame - 1

                if not bbox_matched:
                    #new object
                    detection['obj_id'] = max(id_seq.keys())+1

                id_seq.update({detection['obj_id']: True})
        
        # filter by number of ocurrences
        if remove_noise:
            id_ocurrence = {}
            # Count ocurrences
            for i in range(start_frame, num_frames):
                for detection in self.det_bboxes[frame_id(i)]:
                    objt_id = detection['obj_id']
                    if objt_id in id_ocurrence:
                        id_ocurrence[objt_id] += 1
                    else:
                        id_ocurrence[objt_id] = 1
            # detectiosn to be removed
            ids_to_remove = [id_obj for id_obj in id_ocurrence if id_ocurrence[id_obj]<4]
            for i in range(start_frame, num_frames):
                for idx, detection in enumerate(self.det_bboxes[frame_id(i)]):
                    if detection['obj_id'] in ids_to_remove:
                        self.det_bboxes[frame_id(i)].pop(idx)

    def compute_tracking_kalman(self, display=False): 
        
        data_list = np.array(dict_to_list_tracking(self.det_bboxes))

        total_time = 0.0
        total_frames = 0
        out = []
        idx_frame = []
        colours = np.random.rand(32,3) #used only for display

        mot_tracker = Sort() #create instance of the SORT tracker

        for idx, frame in enumerate(self.det_bboxes): # all frames in the sequence
            
            idx = int(idx)
            
            colors = []

            dets = data_list[data_list[:,0]==idx,1:6]
            im = io.imread(join(self.data_path,frame)+'.png')

            start_time = time.time()
            trackers = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            out.append(trackers)
            if display:
                for d in trackers:
                    d = d.astype(np.uint32)
                    ec=colours[d[4]%32,:]
                    colors.append(ec)
                
                image,centers = draw_bboxes(im,trackers,colors)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imshow('Image', image)
                cv2.waitKey(1)
                cv2.imwrite('Tracking_'+frame+'.png',image)
            
            n_bboxes = len(out[idx])
            for track in out[idx]:
                self.det_bboxes[frame][n_bboxes-1]['obj_id'] = track[4]
                print("frame:",frame)
                print("index:",n_bboxes-1)
                n_bboxes = n_bboxes-1

        idx_frame.append(frame)
        print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
        return(self.det_bboxes)    


            
 

        

        
        
