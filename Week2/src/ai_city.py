import numpy as np
import cv2
from os.path import join
import glob
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.ndimage.morphology import binary_fill_holes


def filter_noise(bg, noise_filter='base', min_area=0.0003):
    """

    """
    if noise_filter in 'morph filter':
        # Opening
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN,  np.ones((1,1)))  
        # Closing
        bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((3,3)))

    # Connected components
    num_lab, labels = cv2.connectedComponents(bg)

    # Filter by area
    rm_labels = [u for u in np.unique(labels) if np.sum(labels == u) < min_area*bg.size]
    for label in rm_labels:
        bg[np.where(labels == label)] = 0
        labels[np.where(labels == label)] = 0    
    
    return bg, labels

def generate_bbox(label_mask):
    """
    
    """
    pos = np.where(label_mask==255)

    if sum(pos).size < 1:
        return 0,0,0,0

    y = np.min(pos[0])
    x = np.min(pos[1])
    h = np.max(pos[0]) - np.min(pos[0])
    w = np.max(pos[1]) - np.min(pos[1])

    return x,y,w,h

def fill_and_get_bbox(labels, fill=True, mask=None):
    """

    """
    bg_idx = np.argmax([np.sum(labels == u) for u in np.unique(labels)])
    fg_idx = np.delete(np.unique(labels),bg_idx)
    
    bg = np.zeros(labels.shape, dtype=np.uint8)

    bboxes = []

    for label in fg_idx:
        aux = np.zeros(labels.shape, dtype=np.uint8)
        aux[labels==label]=255

        # Closing
        aux = cv2.morphologyEx(aux, cv2.MORPH_CLOSE, np.ones((5,5)))

        # Opening -> rm shadows
        aux = cv2.morphologyEx(aux, cv2.MORPH_OPEN, np.ones((5,5)))

        x,y,w,h = generate_bbox(aux)

        # Filter by overlap
        if np.sum(aux>0)/(w*h) >= .55:
            
            if fill:
                # Fill holes
                aux = binary_fill_holes(aux).astype(np.uint8)*255
                
            bg = bg+aux
            # Add new bbox
            bboxes.append([x,y,w,h])
    
    return bg, bboxes

def get_single_objs(bg, noise_filter='base', fill=True, mask=None):
    """

    """
    if noise_filter:
        # Filter noise
        bg, labels = filter_noise(bg, noise_filter)
    else:
        _, labels = cv2.connectedComponents(bg)

    # Generate bboxes and (optional) fill holes
    bg, bboxes = fill_and_get_bbox(labels, fill, mask)

    return bg, bboxes



class AICity:
    """

    """

    def __init__(self, data_path, test_mode = False, resize_factor=0.5, denoise=False, split_factor=0.15, grayscale=True, extension="png",
                 laplacian=False, pre_denoise=False, task=1.1, alpha=3, rho = 0.5, noise_filter=None, fill=False, adaptative_model=False, mask=None):
        """

        """

        # PARAMETERS
        self.data_path = data_path
        self.frames_paths = glob.glob(join(self.data_path, "*." + extension))
        self.frames_paths.sort()
        self.resize_factor = resize_factor
        self.denoise = denoise
        self.split_factor = split_factor
        self.grayscale = grayscale
        self.laplacian = laplacian
        self.pre_denoise = pre_denoise
        self.task = task
        self.alpha = alpha
        self.rho = rho
        self.noise_filter = noise_filter
        self.fill = fill
        self.test_mode = test_mode
        self.adaptative_model = adaptative_model
        self.mask = mask

        # FUNCTIONS
        self.split_data()

    def split_data(self):
        """

        """
        if self.test_mode:
            self.frames_paths = self.frames_paths[0:int(len(self.frames_paths)/10)]
            
        self.bg_modeling_frames_paths = self.frames_paths[
                                        :int(len(self.frames_paths) * self.split_factor)]  # 535 frames
        self.bg_frames_paths = self.frames_paths[int(len(self.frames_paths) * self.split_factor):]
        # 1606 frames

    def create_background_model(self):
        """

        """
        bg_modeling_frames = self.read_frames()
        n_frames, height, width = bg_modeling_frames.shape
        gaussian = np.zeros((height, width, 2))
        gaussian[:, :, 0] = np.mean(bg_modeling_frames, axis=0)
        gaussian[:, :, 1] = np.std(bg_modeling_frames, axis=0)

        del bg_modeling_frames

        self.background_model = gaussian

    def get_frame_background(self, frame, return_bboxes=False):
        """
        :param frame:
        :param model:
        :param grayscale:
        :return:
        """

        if self.grayscale:
            bg = np.zeros_like(frame)

            diff = frame - self.background_model[:, :, 0]
            foreground_idx = np.where(abs(diff) > self.alpha * (2 + self.background_model[:, :, 1]))

            bg[foreground_idx[0], foreground_idx[1]] = 255
            
        else:
            pass
        
        if return_bboxes:
            bg, bboxes = get_single_objs(bg, self.noise_filter, self.fill, self.mask)
            return bg, bboxes

        elif not return_bboxes and self.noise_filter:
            # Filter noise
            bg, _ = filter_noise(bg, self.noise_filter)
        
        return bg, None

        

    def get_frames_background(self, return_bboxes=False):
        for frame_id, frame_path in tqdm(enumerate(self.bg_frames_paths[450:]), 'Predicting background'):
            img, frame = self.read_frame(frame_path, laplacian=self.laplacian, pre_denoise=self.pre_denoise)
            bg, bboxes  = self.get_frame_background(frame, return_bboxes)
            bg = np.repeat(np.expand_dims(bg,axis=2),3,axis=2)

            if self.adaptative_model:
                self.update_gaussian(frame, bg)
            
            if return_bboxes:
                for x,y,w,h in bboxes:
                    bg = cv2.rectangle(bg,(x,y),(x+w,y+h),(255,0,0),2)
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            img = cv2.hconcat((bg, img))
            cv2.imwrite('laplacian/{}.jpg'.format(frame_id),img)
            #cv2.imshow("Background", img)
            #cv2.waitKey(100)
            
            # Free memory
            del img, frame, bg

    def read_frames(self):
        """
        :param paths:
        :param grayscale:
        :return:
        """

        images = []
        for file_name in tqdm(self.bg_modeling_frames_paths, 'Reading frames'):
            images.append(self.read_frame(file_name, laplacian=self.laplacian, pre_denoise=self.pre_denoise)[1])
        
        return np.asarray(images)

    def read_frame(self, path, laplacian=False, pre_denoise=False):
        """

        """

        if self.grayscale:
            img0 = cv2.imread(path)
            if self.resize_factor < 1.0:
                img0 = cv2.resize(img0,
                                   (int(img0.shape[1] * self.resize_factor), int(img0.shape[0] * self.resize_factor)),
                                   cv2.INTER_CUBIC)
            
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) 
            if pre_denoise:   
                img = cv2.fastNlMeansDenoising(img, templateWindowSize = 7)
            if laplacian:
                img = cv2.Laplacian(img, cv2.CV_8U)
            return img0, img
        else:
            # TODO
            pass    
    
    def update_gaussian(self, frame, bg):
        """

        """
        [x,y] = np.where(bg==0)
        #update mean
        self.background_model[x, y, 0] = self.rho*frame[x,y] + (1-self.rho)*self.background_model[x, y, 0]
        #update std
        self.background_model[x, y, 1] = self.rho*np.square(frame[x,y] - self.background_model[x, y, 0]) + (1-self.rho)*self.background_model[x, y, 1]
    
        
        
