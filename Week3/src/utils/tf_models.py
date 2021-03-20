import os
import pathlib
import glob
import time
import yaml

import matplotlib
import matplotlib.pyplot as plt

import io
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

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
    pass


class TFModel():
    def __init__(self, args):
        self.convert_image_to_grayscale = False
        self.flip_image_horizontally = False
        self.save_imgs = False
        self.save_txt = False

        self.models = yaml.load(open('./data/tfm_models/models.yaml','r'), Loader=yaml.FullLoader)
        self.model = hub.load(self.models[args.model])

        self.path_to_labels = './models/research/object_detection/data/mscoco_label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        self.threshold = 0.5

    def load_image_into_numpy_array(path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
            path: the file path to the image

        Returns:
            uint8 numpy array with shape (img_height, img_width, 3)
        """
        image = None
        if(path.startswith('http')):
            response = urlopen(path)
            image_data = response.read()
            image_data = BytesIO(image_data)
            image = Image.open(image_data)
        else:
            image_data = tf.io.gfile.GFile(path, 'rb').read()
            image = Image.open(BytesIO(image_data))

        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (1, im_height, im_width, 3)).astype(np.uint8)


    def predict(self, image):
        image_np = load_image_into_numpy_array(image)

		# Flip horizontally
		if(self.flip_image_horizontally):
			image_np[0] = np.fliplr(image_np[0]).copy()

		# Convert image to grayscale
		if(self.convert_image_to_grayscale):
			image_np[0] = np.tile(
			np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

			plt.figure(figsize=(24,32))
			plt.imshow(image_np[0])
			plt.show()

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
				category_index,
				use_normalized_coordinates=True,
				max_boxes_to_draw=200,
				min_score_thresh=.30,
				agnostic_mode=False)

		detection_boxes = results['detection_boxes'].numpy()
		detection_scores = results['detection_scores'].numpy()

		if self.save_txt:
			for img_number, score in enumerate(detection_scores[0]):
				if score >= threshold:
					f.writelines(str(idx) + " None " + str(detection_boxes[0][img_number][0]) + " "  +\
								str(detection_boxes[0][img_number][1]) + " "  +\
								str(detection_boxes[0][img_number][2]) + " "  +\
								str(detection_boxes[0][img_number][3]) + " " + str(score) + "\n")

		if self.save_imgs:
            plt.figure(figsize=(24,32))
            plt.imshow(image_np_with_detections[0])
            plt.show()
            plt.imsave(str(idx) + ".png", image_np_with_detections[0])
            plt.close()

        return None