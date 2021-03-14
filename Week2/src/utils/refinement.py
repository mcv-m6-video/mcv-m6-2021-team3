import numpy as np
import cv2
from scipy.ndimage.morphology import binary_fill_holes

def filter_noise(bg, noise_filter='base', min_area=0.0003):
    """

    """
    if noise_filter in 'morph_filter':
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
        if np.sum(aux>0)/(w*h) >= .5:
            
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

