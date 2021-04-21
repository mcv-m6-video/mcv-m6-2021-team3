import os
import pathlib
import glob
import time
import yaml
from tqdm import tqdm
from os.path import join, dirname

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
from object_detection.utils import dataset_util, config_util
from object_detection.builders import model_builder
from object_detection.dataset_tools import create_coco_tf_record 

global gt # Temporary fix

class TFModel():

    def __init__(self, model, weights, iou_thres=0.5, coco_model=False):
        """ 
        Class initializer

        :param model: string defining the model to use
        :param iou_thres: non-max suppresion iou threshold
        :param coco_model: use or not coco model

        """

        physical_devices = tf.config.list_physical_devices('GPU')

        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
        
        self.iou_thres = iou_thres
        self.coco_model = coco_model
        
        if self.coco_model:
            self.models = yaml.load(open('/home/josep/shared_dir/Week3bis/src/data/tfm_models/models.yaml','r'), Loader=yaml.FullLoader)
            self.model = hub.load(self.models[model])
            self.path_to_labels = '/home/josep/shared_dir/Week3bis/src/models/research/object_detection/data/mscoco_label_map.pbtxt'
        else:
            self.path_to_labels = 'data/tf2_finetune/label_map.pbtxt'
            self.checkpoints_path = weights
            self.config_file = join(dirname(self.checkpoints_path), 'pipeline.config')
            print(self.config_file)
            print(self.checkpoints_path)
            configs = config_util.get_configs_from_pipeline_file(self.config_file)
            model_config = configs['model']
            self.model = model_builder.build(model_config=model_config, is_training=False)
            ckpt = tf.compat.v2.train.Checkpoint(model=self.model)
            ckpt.restore(self.checkpoints_path).expect_partial()

        self.category_index = label_map_util.create_category_index_from_labelmap(self.path_to_labels, 
                                                                                 use_display_name=True)

    def get_model_detection_function(self, model):
        """
        Get a tf.function for detection.
        """

        #@tf.function
        def detect_fn(image):
            """
            Detect objects in image.
            """

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)

            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    def predict(self, filename):
        """
            Passes the image either through a pre-trained net on the COCO network or a fine-tuned
            image on the AICity dataset
        """

        # running inference
        '''if not self.coco_model:
            with tf.io.gfile.GFile(filename, 'rb') as fid:
                encoded_image = fid.read()
            image = Image.open(BytesIO(encoded_image))
            (width, height) = image.size
            image_np = np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

            detect_fn = self.get_model_detection_function(self.model)
            detections, _, _ = detect_fn(input_tensor)

            detection_boxes = detections['detection_boxes'][0].numpy()
            detection_scores = detections['detection_scores'][0].numpy()
            detection_classes = detections['detection_classes'][0].numpy()
        else:'''
        image_np = cv2.imread(filename)
        height, width, n_channels = image_np.shape
        image = image_np.copy()
        image_np = image_np.reshape(1, height, width, n_channels).astype(np.uint8)
        input_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
        results = self.model(input_tensor)

        detection_boxes = results['detection_boxes'][0].numpy()
        detection_scores = results['detection_scores'][0].numpy()
        detection_classes = results['detection_classes'][0].numpy()

        idx = tf.image.non_max_suppression(
                    detection_boxes, detection_scores, 50, iou_threshold=self.iou_thres,
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
        if not self.coco_model:
            output = [[box, score] for label, box, score in zip(detection_classes,
                    detection_boxes, detection_scores) if label == 0]
        else:
            output = [[box, score] for label, box, score in zip(detection_classes,
                        detection_boxes, detection_scores) if label == 3]

        return output


def create_tf_example(filename, data):    
    """
    This creates a TF record for a single image

    :param filename: path to the image
    :param data: ground truth data for the particular frame

    :return: TFRecord 
    """

    height = 1080 # Image height
    width = 1920 # Image width
    filename = filename.encode('utf8') # Filename of the image. Empty if image is not from file
    
    with tf.io.gfile.GFile(filename, 'rb') as fid:
        encoded_image = fid.read()

    image_format = b'png' # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for entry in data:
        if entry['name'] == 'car':
            xmin, ymin, xmax, ymax = entry['bbox']
            xmins.append(xmin / width)
            xmaxs.append(xmax / width)
            ymins.append(ymin / height)
            ymaxs.append(ymax / height)
            classes_text.append('car'.encode('utf8'))
            classes.append(1)
    
    tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_label_and_data


def to_tf_record(data, gt, writer):
    '''paths = {
        'train': os.path.join(tf_records_path, 'train'),
        'val': os.path.join(tf_records_path, 'val')
    }'''
    
    for dataset in ['train', 'val', 'test']:
        #writer = tf.io.TFRecordWriter(paths[dataset] + '.tfrecord')
        for cam, frames in data[dataset].items():
            for img in tqdm(frames, 'Creating TF Record'):
                frame_id = img.split('/')[-1].split('.')[0]
                if frame_id in gt[cam].keys():
                    record = create_tf_example(img, gt[cam][frame_id])
                    writer[dataset].write(record.SerializeToString())
        
        #print("Saving TFRecords file. This can take a while...")
        #writer.close()
    return writer
