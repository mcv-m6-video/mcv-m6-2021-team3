import cv2
import glob
import xml.etree.ElementTree as ET
from os.path import join
import png
import numpy as np


def update_data(annot,frame_id,xmin,ymin,xmax,ymax,conf):

    frame_name = '%04d' % int(frame_id)
    obj_info = dict(
        name = 'car',
        bbox = [xmin, ymin, xmax, ymax],
        confidence = conf
    )

    if frame_name not in annot.keys():
        annot.update({frame_name:[obj_info]})
    else:
        annot[frame_name].append(obj_info)

    return annot

def load_text(text_dir,text_name):
    with open(join(text_dir,text_name),'r') as f:
        txt = f.readlines()

    annot = {}
    for frame in txt:
        frame_id,_,xmin,ymin,width,height, conf,_,_,_ = list(map(float,(frame.split('\n')[0]).split(',')))
        update_data(annot,frame_id,xmin, ymin, xmin+width, ymin+height, conf)
    return annot

def load_xml(xml_dir,xml_name):
    tree = ET.parse(join(xml_dir,xml_name))
    root = tree.getroot()
    annot = {}
    for child in root:
        if child.tag in 'track':
            if child.attrib['label'] not in 'car':
                continue
            for bbox in child.getchildren():
                frame_id, xmin, ymin, xmax, ymax,_,_,_ = list(map(float,([v for k,v in bbox.attrib.items()])))
                update_data(annot,int(frame_id)-1,xmin, ymin, xmax, ymax, 1.)
                
    return annot

def load_annot(annot_dir, name):
    if name.endswith('txt'):
        annot = load_text(annot_dir, name)
    elif name.endswith('xml'):
        annot = load_xml(annot_dir, name)
    else:
        assert 'Not supported annotation format '+name.split('.')[-1]
    
    return annot

def load_frames(vdo_dir):
    img_dirs = glob.glob(join(vdo_dir,'*.png'))
    img_dirs.sort()
    return img_dirs

def read_kitti_OF(flow_file):
    '''
    Read from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    '''
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
