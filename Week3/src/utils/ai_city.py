import numpy as np
import cv2
from PIL import Image
import os
from os.path import join, basename
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from utils.metrics import voc_eval
from utils.utils import write_json_file
from shutil import copyfile
from detectron2.structures import BoxMode

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
            for bbox in child.getchildren():
                '''if bbox.getchildren()[0].text in 'true':
                    continue'''
                frame_id, xmin, ymin, xmax, ymax, _, _, _ = list(map(float, ([v for k, v in bbox.attrib.items()])))
                update_data(annot, int(frame_id) + 1, xmin, ymin, xmax, ymax, 1.)

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


def update_data(annot, frame_id, xmin, ymin, xmax, ymax, conf):
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
        bbox=list(map(float, [xmin, ymin, xmax, ymax])),
        confidence=conf
    )

    if frame_name not in annot.keys():
        annot.update({frame_name: [obj_info]})
    else:
        annot[frame_name].append(obj_info)

    return annot

def gt_multi_txt(path, bboxes):
    
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


def to_yolov3(data, gt_bboxes, save_path='yolov3_data'):
    
    data_path = join(os.getcwd(),save_path,'data')
    if os.path.exists(data_path):
        if len(glob.glob(data_path+'/*.*')) == 2*sum([len(d) for _,d in data.items()]):
            print('Data already in YOLOv3 format!')
            return

    os.makedirs(data_path,exist_ok=True)

    for split, split_data in data.items():
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

def add_record(image_id, filename, bboxes):
    record = {}
        
    width, height = Image.open(filename).size
    
    record["file_name"] = filename
    record["image_id"] = int(image_id)
    record["height"] = height
    record["width"] = width

    objs = []
    for obj_info in bboxes:
        label = 0 #obj_info['name']
        xmin, ymin, xmax, ymax = obj_info['bbox']

        obj = {
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,
        }
        objs.append(obj)
    record["annotations"] = objs
    
    return record


def to_detectron2(data, gt_bboxes):
    
    datasets_dicts = {}
    for split, split_data in data.items():
        dataset_dicts = []
        for path in tqdm(split_data,'Preparing '+split+' data for detectron2'):
            frame_id = basename(path).split('.')[0]
            dataset_dicts.append(add_record(frame_id, path, gt_bboxes[frame_id]))
        
        datasets_dicts.update({split:dataset_dicts})

    return datasets_dicts

class AICity:
    """
    This class contains all the logic of the background estimation process for the AICity dataset
    """

    def __init__(self, args):
        """
        Init of the AICity class

        :param args: configuration for the current estimation
        """

        # INPUT PARAMETERS
        self.data_path = args.data_path
        self.img_size = args.img_size        

        self.gt_bboxes = load_annot(args.gt_path, 'ai_challenge_s03_c010-full_annotation.xml')
        self.det_bboxes = {}

        # Load frame paths and filter by gt
        self.frames_paths = glob.glob(join(self.data_path, "*." + args.extension))
        self.frames_paths.sort()
        self.frames_paths = [path for frame_id,_ in self.gt_bboxes.items() for path in self.frames_paths if frame_id in path]
        '''
        if args.test_mode:
            self.frames_paths = self.frames_paths[0:int(len(self.frames_paths) / 10)]
        '''
        self.data = {'train':[], 'val':[]}

        self.split_factor = args.split_factor
        self.task = args.task
        self.model = args.model

        # OUTPUT PARAMETERS
        self.output_path = args.output_path
        self.save_json = args.save_json
        self.view_img = args.view_img
        self.save_img = args.save_img

        
    
    def __len__(self):
        return len(self.frames_paths)

    def train_val_split(self):
        """
        Apply random split to specific propotion of the dataset (split_factor).

        """
        train, val, _, _ = train_test_split(np.array(self.frames_paths), np.empty((len(self),)),
                                            test_size=self.split_factor, random_state=0)

        self.data['train'] = train.tolist()  
        self.data['val'] = val.tolist()
    
    def data_to_model(self):
        if self.model in 'yolov3':
            to_yolov3(self.data, self.gt_bboxes)
        elif self.model in ['faster_rcnn', 'retinanet']:
            to_detectron2(self.data, self.gt_bboxes)
    
    '''def detect(self):
        if self.model in ['Faster_RCNN', 'Mask_RCNN', 'RetinaNet']:
            detect2'''


    def train_split(self, split=0):
        """
        Apply random split to specific propotion of the train set.

        Args:
            split (float): proportion of the train set that will be used.
        """
        self.train_dataset = random.choices(self.dataset_train,k=int(len(self.dataset_train)*split))

    def read_frames(self):
        """
        Reds all the frames from disk

        :return: np.array of frames
        """

        images = []
        for file_name in tqdm(self.bg_modeling_frames_paths, 'Reading frames'):
            images.append(self.read_frame(file_name,
                                          laplacian=self.options.laplacian,
                                          pre_denoise=self.options.pre_denoise)[1])

        return np.asarray(images)
    

    def get_mAP(self):
        """
        Estimats the mAP using the VOC evaluation

        :return: map of all estimated frames
        """
        return \
        voc_eval(self.gt_bboxes, self.bg_frames_paths, self.det_bboxes, resize_factor=self.options.resize_factor)[2]

    def save_results(self, name_json):
        """
        Saves results to a JSON file

        :param name_json: name of the json file
        """
        write_json_file(self.det_bboxes, name_json)

    def visualize_task(self, frame, bg, frame_id):
        """
        Creates plots for a given frame and background estimation

        :param frame: original image
        :param bg: estimated background/foreground
        :param frame_id: id of the given frame
        """
        self.miou, self.std_iou = np.empty(0, ), np.empty(0, )
        self.miou, self.std_iou, self.xaxis = visualize_background_iou(self.miou, self.std_iou, self.xaxis,
                                                                       frame, frame_id, bg, self.gt_bboxes,
                                                                       self.det_bboxes, self.options)
