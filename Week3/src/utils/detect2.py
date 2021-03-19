# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from PIL import Image
from utils.utils import get_weights

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

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


def predict(img_path, model):
    
    if model in 'faster_rcnn':
        cfg_file = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'        
    elif model in 'mask_rcnn':
        cfg_file = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    elif model in 'retinanet':
        cfg_file = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'

    weights = get_weights(model)

    # load image
    img = cv2.imread(img_path)

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = weights
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)

    # Visualize
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_img = out.get_image()

    # filter only cars
    output = [box.cpu().numpy().tolist() for label, box in zip(outputs["instances"].pred_classes,outputs["instances"].pred_boxes) if label == 2]

    return output



