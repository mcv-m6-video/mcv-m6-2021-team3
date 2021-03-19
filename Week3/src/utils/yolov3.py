import os
from PIL import Image
from os.path import join, basename
import glob
import numpy as np
from tqdm import tqdm
from shutil import copyfile

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