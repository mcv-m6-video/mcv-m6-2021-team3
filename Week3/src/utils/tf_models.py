import os
import pathlib
import glob
import time
import yaml

import matplotlib
import matplotlib.pyplot as plt

import io
import cv2
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops


class TFModel():

    def __init__(self, args, model):
        physical_devices = tf.config.list_physical_devices('GPU')

        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        self.models = yaml.load(open('./data/tfm_models/models.yaml','r'), Loader=yaml.FullLoader)
        self.model = hub.load(self.models[model])
        self.threshold = args.threshold
        self.path_to_labels = './models/research/object_detection/data/mscoco_label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(self.path_to_labels, use_display_name=True)


    def predict(self, image):
        image_np = cv2.imread(image)
        height, width, n_channels = image_np.shape
        image = image_np.copy()
        image_np = image_np.reshape(1, height, width, n_channels).astype(np.uint8)

        # running inference
        results = self.model(image_np)

        # different object detection models have additional results
        # all of them are explained in the documentation
        result = {key:value.numpy() for key,value in results.items()}

        label_id_offset = 0
        image_np_with_detections = image_np.copy()

        # Use keypoints if available in detections
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in result:
                keypoints = result['detection_keypoints'][0]
                keypoint_scores = result['detection_keypoint_scores'][0]

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections[0],
                result['detection_boxes'][0],
                (result['detection_classes'][0] + label_id_offset).astype(int),
                result['detection_scores'][0],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=self.threshold,
                agnostic_mode=False)

        #cv2.imshow("detection", image_np_with_detections[0])

        detection_boxes = results['detection_boxes'][0].numpy()
        detection_scores = results['detection_scores'][0].numpy()
        detection_classes = results['detection_classes'][0].numpy()

        idx = tf.image.non_max_suppression(
                    result['detection_boxes'][0], result['detection_scores'][0], 50, iou_threshold=self.threshold,
                    score_threshold=float('-inf'), name=None)
        
        detection_classes = detection_classes[idx.numpy()]
        detection_boxes = detection_boxes[idx.numpy()]
        detection_scores = detection_scores[idx.numpy()]

        for idx, bbox in enumerate(detection_boxes):
            ymin, xmin, ymax, xmax = bbox

            detection_boxes[idx][1] = ymin * height
            detection_boxes[idx][0] = xmin * width
            detection_boxes[idx][3] = ymax * height
            detection_boxes[idx][2] = xmax * width
        
        # filter only cars
        output = [[box, score] for label, box, score in zip(detection_classes,
                    detection_boxes, detection_scores) if label == 3]

        return output
    
    def train(self):
        pass
