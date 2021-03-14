import numpy as np
import cv2
from os.path import join
import glob
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


def filter_noise(bg, post_processing=True, min_area=0.15):
    """

    """
    if post_processing:
        # Opening
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN,  np.ones((1,1)))  
        # Closing
        bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((3,3)))

    # Connected components
    num_lab, labels = cv2.connectedComponents(bg)

    # Filter by area
    rm_labels = [u for u in np.unique(labels) if np.sum(labels == u) < min_area*len(bg)]
    for label in rm_labels:
        bg[np.where(labels == label)] = 0
        labels[np.where(labels == label)] = 0    
    
    return bg, labels

def generate_bbox(mask):
    """
    
    """
    pos = np.where(mask==255)

    y = np.min(pos[0])
    x = np.min(pos[1])
    h = np.max(pos[0]) - np.min(pos[0])
    w = np.max(pos[1]) - np.min(pos[1])

    return x,y,w,h

def fill_gaps(labels):
    """

    """
    bg_idx = np.argmax([np.sum(labels == u) for u in np.unique(labels)])
    fg_idx = np.delete(np.unique(labels),bg_idx)
    
    bg = np.zeros(labels.shape, dtype=np.uint8)

    for label in fg_idx:
        aux = np.zeros(labels.shape, dtype=np.uint8)
        aux[labels==label]=255

        contours,_ = cv2.findContours(aux, 1, 2)

        # create hull array for convex hull points
        hull = []
        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            #hull = cv2.convexHull(contours[i], False)
            epsilon = 0.1*cv2.arcLength(contours[i],True)
            approx = cv2.approxPolyDP(contours[i],epsilon,True)
            aux = cv2.fillPoly(aux, pts=[approx],color=(255))

        x,y,w,h = generate_bbox(aux)
        if np.sum(aux>0)/(w*h) > .35:
            aux = cv2.rectangle(aux,(x,y),(x+w,y+h),(50),2)
            bg = bg+aux

        '''drawing = np.zeros((aux.shape[0], aux.shape[1], 3), np.uint8)
        # draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0) # green - color for contours
            color = (255, 0, 0) # blue - color for convex hull
            # draw ith contour
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            cv2.drawContours(drawing, hull, i, color, 1, 8)'''
    
    return bg

def get_single_objs(bg):
    """

    """
    # Filter noise
    bg, labels = filter_noise(bg)

    # Generate bboxes    
    bg = fill_gaps(labels)
    
    # Filter by aspect ratio
    return bg



class AICity:
    """

    """

    def __init__(self, data_path, test_mode = False, resize_factor=0.5, denoise=False, split_factor=0.15, grayscale=True, extension="png",
                 laplacian=False, pre_denoise=False, task=1.1, alpha=3, rho = 0.5, rm_noise=None, fill=False, adaptative_model=False):
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
        self.rm_noise = rm_noise
        self.fill = fill
        self.test_mode = test_mode
        self.adaptative_model = adaptative_model
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

    def get_frame_background(self, frame):
        """
        :param frame:
        :param model:
        :param grayscale:
        :param rm_noise:
        :param fill:
        :return:
        """

        if self.grayscale:
            bg = np.zeros_like(frame)

            diff = frame - self.background_model[:, :, 0]
            foreground_idx = np.where(abs(diff) > self.alpha * (2 + self.background_model[:, :, 1]))

            bg[foreground_idx[0], foreground_idx[1]] = 255
            
            bg = get_single_objs(bg)

            '''if self.rm_noise is not None:
                bg = filter_noise(bg, self.rm_noise)
            if self.fill:
                bg = fill_gaps(bg)'''
        else:
            pass

        return bg

    def get_frames_background(self):
        for frame_id, frame_path in tqdm(enumerate(self.bg_frames_paths), 'Predicting background'):
            frame = self.read_frame(frame_path, laplacian=self.laplacian, pre_denoise=self.pre_denoise)
                        
            bg = self.get_frame_background(frame)                              
            img = self.read_frame(frame_path)
            img = cv2.hconcat((bg, img))
            cv2.imwrite('laplacian/{}.jpg'.format(frame_id),img)
            #cv2.imshow("Background", img)
            #cv2.waitKey(100)

            if self.adaptative_model:
                self.update_gaussian(frame, bg)
            
            frame, img, bg = None, None, None

    def read_frames(self):
        """
        :param paths:
        :param grayscale:
        :return:
        """

        images = []
        for file_name in tqdm(self.bg_modeling_frames_paths, 'Reading frames'):
            images.append(self.read_frame(file_name, laplacian=self.laplacian, pre_denoise=self.pre_denoise))
        
        return np.asarray(images)

    def read_frame(self, path, laplacian=False, pre_denoise=False):
        """

        """

        if self.grayscale:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.resize_factor < 1.0:
                image = cv2.resize(image,
                                   (int(image.shape[1] * self.resize_factor), int(image.shape[0] * self.resize_factor)),
                                   cv2.INTER_CUBIC)
            if pre_denoise:   
                image = cv2.fastNlMeansDenoising(image, templateWindowSize = 7)
            if laplacian:
                image = cv2.Laplacian(image, cv2.CV_8U)
            return image
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
    
        
        
