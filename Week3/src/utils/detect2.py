# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from utils.utils import get_weights

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

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
    output = [[label.cpu().numpy().tolist(), box.cpu().numpy().tolist()] for label, box in zip(outputs["instances"].pred_classes,outputs["instances"].pred_boxes) if label == 2]

    return output



