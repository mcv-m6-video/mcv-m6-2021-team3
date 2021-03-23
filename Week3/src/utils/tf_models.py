import os
import pathlib
import glob
import time
import yaml
from tqdm import tqdm

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

    def __init__(self, args, model):
        """ 
        Class initializer

        :param args: argsparse of options
        :param model: string defining the model to use

        """

        physical_devices = tf.config.list_physical_devices('GPU')

        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
        
        self.options = args
        
        if args.coco_model:
            self.models = yaml.load(open('/home/josep/shared_dir/Week3bis/src/data/tfm_models/models.yaml','r'), Loader=yaml.FullLoader)
            self.model = hub.load(self.models[model])
            self.path_to_labels = '/home/josep/shared_dir/Week3bis/src/models/research/object_detection/data/mscoco_label_map.pbtxt'
        else:
            self.path_to_labels = '/home/josep/shared_dir/Week3bis/src/data/finetune/tf_records/label_map.pbtxt'
            self.checkpoints_path = os.path.join('/home/josep/shared_dir/Week3bis/src/models/research/object_detection/checkpoints/',
                                                 args.trained_model)
            self.config_file = os.path.join(self.checkpoints_path, 'pipeline.config')
            print(self.config_file)
            print(self.checkpoints_path)
            configs = config_util.get_configs_from_pipeline_file(self.config_file)
            model_config = configs['model']
            self.model = model_builder.build(model_config=model_config, is_training=False)
            ckpt = tf.compat.v2.train.Checkpoint(model=self.model)
            ckpt.restore(os.path.join(self.checkpoints_path, 'trained', 'ckpt-0')).expect_partial()

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
        if not self.options.coco_model:
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
        else:
            image_np = cv2.imread(filename)
            height, width, n_channels = image_np.shape
            image = image_np.copy()
            image_np = image_np.reshape(1, height, width, n_channels).astype(np.uint8)
            results = self.model(image_np)

            detection_boxes = results['detection_boxes'][0].numpy()
            detection_scores = results['detection_scores'][0].numpy()
            detection_classes = results['detection_classes'][0].numpy()

        idx = tf.image.non_max_suppression(
                    detection_boxes, detection_scores, 50, iou_threshold=self.options.iou_threshold,
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
        if not self.options.coco_model:
            output = [[box, score] for label, box, score in zip(detection_classes,
                    detection_boxes, detection_scores) if label == 0]
        else:
            output = [[box, score] for label, box, score in zip(detection_classes,
                        detection_boxes, detection_scores) if label == 3]

        return output

    # @tf.function
    # def train_step(self,
    #                 groundtruth_boxes_list,
    #                 groundtruth_classes_list,
    #                 image_tensors, 
    #                 optimizer, 
    #                 vars_to_fine_tune,
    #                 image_size):
    #     """A single training iteration.

    #     Args:
    #     image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
    #         Note that the height and width can vary across images, as they are
    #         reshaped within this function to be 640x640.
    #     groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
    #         tf.float32 representing groundtruth boxes for each image in the batch.
    #     groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
    #         with type tf.float32 representing groundtruth boxes for each image in
    #         the batch.

    #     Returns:
    #     A scalar tensor representing the total loss for the input batch.
    #     """

    #     shapes = tf.constant(self.options.batch_size * [[image_size, image_size, 3]], dtype=tf.int32)
    #     self.detection_model.provide_groundtruth(
    #         groundtruth_boxes_list=groundtruth_boxes_list,
    #         groundtruth_classes_list=groundtruth_classes_list)

    #     with tf.GradientTape() as tape:
    #         preprocessed_images = tf.concat([self.detection_model.preprocess(image_tensor)[0] for image_tensor in image_tensors], axis=0)
    #         prediction_dict = self.detection_model.predict(preprocessed_images, shapes)
    #         losses_dict = self.detection_model.loss(prediction_dict, shapes)
    #         total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
    #         gradients = tape.gradient(total_loss, vars_to_fine_tune)
    #         optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        
    #     return total_loss

    # def dataset_imgs(self, img):
    #     global gt

    #     img = bytes.decode(img.numpy())
    #     img_id = img.split('/')[-1].split('.')[0]
    #     image = cv2.imread(img)
    #     height, width, n_channels = image.shape

    #     return image

    # def dataset_bbox(self, img):
    #     global gt

    #     img = bytes.decode(img.numpy())
    #     img_id = img.split('/')[-1].split('.')[0]

    #     xmins, ymins, xmaxs, ymaxs = [], [], [], []
    #     bbox = []
    #     for obj_id, obj in enumerate(gt[img_id]):
    #         xmin, ymin, xmax, ymax = obj['bbox']
    #         xmins.append(xmin)
    #         ymins.append(ymin)
    #         xmaxs.append(xmax)
    #         ymaxs.append(ymax)

    #     bbox.append(xmins)
    #     bbox.append(ymins)
    #     bbox.append(xmaxs)
    #     bbox.append(ymaxs)

    #     return bbox

    # def dataset_classes(self, img):
    #     global gt

    #     img = bytes.decode(img.numpy())
    #     img_id = img.split('/')[-1].split('.')[0]
        
    #     classes = []

    #     for obj_id, obj in enumerate(gt[img_id]):
    #         classes.append(3)

    #     return classes

    # def build_dataset(self, args, data):
    #     """
        
    #     """

    #     imgs = tf.data.Dataset.from_tensor_slices((data))
    #     imgs = imgs.map(lambda x: tf.py_function(self.dataset_imgs, [x], [tf.float32]))
    #     imgs = imgs.batch(batch_size=self.options.batch_size,
    #                             drop_remainder=True)
    #     imgs = imgs.prefetch(buffer_size=self.options.batch_size)

    #     bbox = tf.data.Dataset.from_tensor_slices((data))
    #     bbox = bbox.map(lambda x: tf.py_function(self.dataset_bbox, [x], [tf.float32]))
    #     bbox = bbox.batch(batch_size=self.options.batch_size,
    #                             drop_remainder=True)
    #     bbox = bbox.prefetch(buffer_size=self.options.batch_size)

    #     classes = tf.data.Dataset.from_tensor_slices((data))
    #     classes = classes.map(lambda x: tf.py_function(self.dataset_classes, [x], [tf.float32]))
    #     classes = classes.batch(batch_size=self.options.batch_size,
    #                             drop_remainder=True)
    #     classes = classes.prefetch(buffer_size=self.options.batch_size)
        
    #     return imgs, bbox, classes


    # ""def train(self, data, ground_truth):
    #     """
    #     This performs a custom training to a TF model for object detection.

    #     :param data: training and validation images.
    #     :param ground_truth: ground truth data
    #     """
    #     global gt
    #     gt = ground_truth

    #     tf.keras.backend.clear_session() # Just in case something is there from previous trains / inferences

    #     train_data = data['train']
    #     val_data = data['val']

    #     train_imgs, train_bbox, train_classes = self.build_dataset(self.options, train_data)
    #     val_imgs, val_bbox, val_classes = self.build_dataset(self.options, val_data)

    #     model = self.options.model_conf_file
    #     configs_path = os.path.join('./models/research/object_detection/configs/tf2', model)
    #     checkpoint_path = os.path.join('./models/research/object_detection/checkpoints', self.options.model, 'checkpoint/ckpt-0')
    #     num_classes = 1

    #     config = config_util.get_configs_from_pipeline_file(configs_path)
    #     model_config = config['model']
    #     model_config.ssd.num_classes = num_classes
    #     model_config.ssd.freeze_batchnorm = True
    #     self.detection_model = model_builder.build(model_config=model_config, is_training=True)

    #     # LOAD WEIGHTS
    #     # Run model through a dummy image so that variables are created
    #     fake_box_predictor = tf.compat.v2.train.Checkpoint(
    #         _base_tower_layers_for_heads=self.detection_model._box_predictor._base_tower_layers_for_heads,
    #         # _prediction_heads=detection_model._box_predictor._prediction_heads,
    #         #    (i.e., the classification head that we *will not* restore)
    #         _box_prediction_head=self.detection_model._box_predictor._box_prediction_head,
    #         )
    #     fake_model = tf.compat.v2.train.Checkpoint(
    #             _feature_extractor=self.detection_model._feature_extractor,
    #             _box_predictor=fake_box_predictor)
    #     ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    #     ckpt.restore(checkpoint_path).expect_partial()

    #     # Run model through a dummy image so that variables are created
    #     image, shapes = self.detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    #     prediction_dict = self.detection_model.predict(image, shapes)
    #     _ = self.detection_model.postprocess(prediction_dict, shapes)

    #     print('Start fine-tuning!', flush=True)
    #     # Define variables to train
    #     # Select variables in top layers to fine-tune.
    #     trainable_variables = self.detection_model.trainable_variables
    #     to_fine_tune = []
    #     prefixes_to_train = ['WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
    #                          'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']

    #     for var in trainable_variables:
    #         if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
    #             to_fine_tune.append(var)

    #     # Define optimizer
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=self.options.learning_rate)

    #     for epoch in range(self.options.epochs):
    #         idx = 1
    #         for imgs, bboxs, classes in zip(train_imgs, train_bbox, train_classes):                
    #             # Training step (forward pass + backwards pass)
    #             total_loss = self.train_step(bboxs,
    #                                          classes,
    #                                          imgs, 
    #                                         optimizer, 
    #                                         to_fine_tune,
    #                                         self.options.img_size[0])

    #             if idx % 10 == 0:
    #                 print('batch ' + str(idx) + ' of ' + str(self.options.num_batches)
    #                 + ', loss=' +  str(total_loss.numpy()), flush=True)

    #             idx += 1
            
    #         idx = 0""


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


def to_tf_record(args, data, gt):
    paths = {
        'train': os.path.join(args.tf_records_path, 'train'),
        'val': os.path.join(args.tf_records_path, 'val')
    }

    train_data = data['train']
    val_data = data['val']
    
    for dataset in ['train', 'val']:
        writer = tf.io.TFRecordWriter(paths[dataset] + '.tfrecord')

        for idx, img in tqdm(enumerate(data[dataset]), 'Creating TF Record'):
            record = create_tf_example(img, gt[img.split('/')[-1].split('.')[0]])
            writer.write(record.SerializeToString())
        
        print("Saving TFRecords file. This can take a while...")
        writer.close()
