import numpy as np
import cv2
from os.path import join
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET
from utils.refinement import get_single_objs, filter_noise
from utils.metrics import voc_eval
from utils.utils import write_json_file

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
                if bbox.getchildren()[0].text in 'true':
                    continue
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
        bbox=[xmin, ymin, xmax, ymax],
        confidence=conf
    )

    if frame_name not in annot.keys():
        annot.update({frame_name: [obj_info]})
    else:
        annot[frame_name].append(obj_info)

    return annot



class AICity:
    """

    """

    def __init__(self, data_path, gt_path, options):
        """

        """

        # PARAMETERS
        self.data_path = data_path
        self.frames_paths = glob.glob(join(self.data_path, "*." + options['extension']))
        self.frames_paths.sort()
        self.options = options
        self.gt_bboxes = load_annot(gt_path,'ai_challenge_s03_c010-full_annotation.xml')
        self.det_bboxes = {}

        # FUNCTIONS
        self.split_data()

    def split_data(self):
        """

        """
        if self.options['test_mode']:
            self.frames_paths = self.frames_paths[0:int(len(self.frames_paths) / 10)]

        self.bg_modeling_frames_paths = self.frames_paths[
                                        :int(len(self.frames_paths) * self.options['split_factor'])]  # 535 frames
        self.bg_frames_paths = self.frames_paths[int(len(self.frames_paths) * self.options['split_factor']):]
        self.bg_frames_paths = self.bg_frames_paths[:300]
        # 1606 frames

    def create_background_model(self):
        """

        """
        bg_modeling_frames = self.read_frames()

        if self.options['colorspace'] == 'gray':
            n_frames, height, width = bg_modeling_frames.shape
            gaussian = np.zeros((height, width, 2))
            gaussian[:, :, 0] = np.mean(bg_modeling_frames, axis=0)
            gaussian[:, :, 1] = np.std(bg_modeling_frames, axis=0)

            del bg_modeling_frames

            self.background_model = gaussian

        else:
            if self.options['colorspace'] == "LAB":
                n_frames, height, width, n_channels = bg_modeling_frames.shape
                gaussian = np.zeros((height, width, 4))

                # Channel A
                gaussian[:, :, 0] = np.mean(bg_modeling_frames[:, :, :, 0], axis=0)
                gaussian[:, :, 1] = np.std(bg_modeling_frames[:, :, :, 0], axis=0)

                # Channel B
                gaussian[:, :, 2] = np.mean(bg_modeling_frames[:, :, :, 1], axis=0)
                gaussian[:, :, 3] = np.std(bg_modeling_frames[:, :, :, 1], axis=0)

                self.background_model = gaussian

            elif self.options['colorspace'] == "HSV":
                n_frames, height, width = bg_modeling_frames.shape

                gaussian = np.zeros((height, width, 2))

                # Channel H
                gaussian[:, :, 0] = np.mean(bg_modeling_frames, axis=0)
                gaussian[:, :, 1] = np.std(bg_modeling_frames, axis=0)

                self.background_model = gaussian

    def get_frame_background(self, frame):
        """
        :param frame:
        :return:
        """

        if self.options['colorspace'] == 'gray':
            bg = np.zeros_like(frame)

            diff = frame - self.background_model[:, :, 0]
            foreground_idx = np.where(abs(diff) > self.options['alpha'] * (2 + self.background_model[:, :, 1]))

            bg[foreground_idx[0], foreground_idx[1]] = 255

        elif self.options['colorspace'] == "LAB" or self.options['colorspace'] == "YCbCr":
            bg = np.zeros((frame.shape[0], frame.shape[1]))

            diff_ch1 = frame[:, :, 0] - self.background_model[:, :, 0]
            diff_ch2 = frame[:, :, 1] - self.background_model[:, :, 2]

            foreground_ch1_idx = np.where(abs(diff_ch1) > self.options['alpha'] * (2 + self.background_model[:, :, 1]))
            foreground_ch2_idx = np.where(abs(diff_ch2) > self.options['alpha'] * (2 + self.background_model[:, :, 3]))

            bg[foreground_ch1_idx[0], foreground_ch1_idx[1]] = 255
            bg[foreground_ch2_idx[0], foreground_ch2_idx[1]] = 255

        elif self.options['colorspace'] == "HSV":
            bg = np.zeros_like(frame)

            diff = frame - self.background_model[:, :, 0]

            foreground_idx = np.where(abs(diff) > self.options['alpha'] * (2 + self.background_model[:, :, 1]))

            bg[foreground_idx[0], foreground_idx[1]] = 255

        bg = bg.astype(np.uint8)

        if self.options['return_bboxes']:
            bg, bboxes = get_single_objs(bg, self.options['noise_filter'], self.options['fill'])
            return bg, bboxes

        elif not self.options['return_bboxes'] and self.options['noise_filter']:
            # Filter noise
            bg, _ = filter_noise(bg, self.options['noise_filter'])
        
        return bg, None

    def get_frames_background(self):
        for frame_id, frame_path in tqdm(enumerate(self.bg_frames_paths), 'Predicting background'):
            img, frame = self.read_frame(frame_path, colorspace=self.options['colorspace'],
                                    laplacian=self.options['laplacian'], pre_denoise=self.options['pre_denoise'])

            bg, bboxes = self.get_frame_background(frame)
            bg = np.repeat(np.expand_dims(bg,axis=2),3,axis=2)

            if self.options['adaptive_model']:
                self.update_gaussian(frame, bg)
            
            if self.options['return_bboxes']:                
                for x,y,w,h in bboxes:
                    self.det_bboxes = update_data(self.det_bboxes, frame_path[-8:-4], x, y, x+w, y+h, 1.)
                    bg = cv2.rectangle(bg,(x,y),(x+w,y+h),(255,0,0),2)
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            img = cv2.hconcat((bg, img))
            #cv2.imwrite('result/{}.jpg'.format(frame_id),img)
            #cv2.imshow("Background", img)
            #cv2.waitKey(100)
            
            # Free memory
            del img, frame, bg
        
        print('DONE!')

    def read_frames(self):
        """
        :return:
        """

        images = []
        for file_name in tqdm(self.bg_modeling_frames_paths, 'Reading frames'):
            images.append(self.read_frame(file_name, colorspace=self.options['colorspace'],
                                          laplacian=self.options['laplacian'],
                                          pre_denoise=self.options['pre_denoise'])[1])

        return np.asarray(images)

    def read_frame(self, path, colorspace='gray', laplacian=False, pre_denoise=False):
        """

        """
        img0 = cv2.imread(path)

        if self.options['resize_factor'] < 1.0:
            img = cv2.resize(img0,
                                (int(img0.shape[1] * self.options['resize_factor']),
                                int(img0.shape[0] * self.options['resize_factor'])),
                                cv2.INTER_CUBIC)
        if pre_denoise:
            img = cv2.fastNlMeansDenoising(img, templateWindowSize=7)
        if laplacian:
            img = cv2.Laplacian(img, cv2.CV_8U)

        if self.options['colorspace'] == 'gray':
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        elif self.options['colorspace'] == "LAB":
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)[:, :, 1:]

        elif self.options['colorspace'] == "YCbCr":
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2YCrCb)[:, :, 1:]

        elif self.options['colorspace'] == "HSV":
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)[:, :, 0]

        if self.options['bilateral_filter']:            
            img[:, :, 0] = cv2.bilateralFilter(img[:, :, 0], 9, 75, 75)
            img[:, :, 1] = cv2.bilateralFilter(img[:, :, 1], 9, 75, 75)
        
        if self.options['median_filter']:
            filter_size = 5
            kernel = np.ones((filter_size, filter_size)) / filter_size ** 2
            img = cv2.filter2D(img, -1, kernel)

        return img0, img

    def update_gaussian(self, frame, bg):
        """

        """

        if self.options['colorspace'] == 'gray':
            [x, y] = np.where(bg == 0)
            # update mean
            self.background_model[x, y, 0] = self.options['rho'] * frame[x, y] + (1 - self.options['rho']) * self.background_model[x, y, 0]
            # update std
            self.background_model[x, y, 1] = self.options['rho'] * np.square(frame[x, y] - self.background_model[x, y, 0]) + (
                        1 - self.options['rho']) * self.background_model[x, y, 1]
        else:
            pass
    
    def get_mAP(self):
        return voc_eval(self.gt_bboxes, self.bg_frames_paths, self.det_bboxes)[2]

    def save_results(self, name_json):
        write_json_file(self.det_bboxes, name_json)
