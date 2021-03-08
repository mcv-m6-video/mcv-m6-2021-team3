import random

def gen_noisy_bbox(list_bbox, x_size = 1920, y_size = 1080, bbox_generate = False, bbox_delete = False, random_noise = False, bbox_displacement = False):
    """
    Generate the different type of noise added to the bounding boxes
    :param list_bbox: list containing the coordinates of the bounding boxes
    :param x_size, y_size: size of the image 
    :param bbox_generate, bbox_delete, random_noise, bbox_displacement: 
    boolean used to determine which kind of noise is applied
    :param max_random_px: number of maximum pixels that increases the size of the bbox
    :param max_displacement_px: number of the maximum pixels where the bbox is moved
    :param max_perc_create_bbox: max probability of creating new bouding boxes
    :param max_prob_delete_bbox: max probability of removing bouding boxes
    :retur: return the list created with the new coordinates for the bboxes
    """
    
    # assumes each bbox is a list ordered as [xmin, ymin, width, height]

    noisy_list_bbox = list_bbox.copy()

    max_random_px = 5
    max_displacement_px = 5
    
    max_prob_delete_bbox = 0.5
    prob_delete_bbox = random.random() * max_prob_delete_bbox
    
    max_perc_create_bbox = 0.5
    perc_create_bbox = random.random() * max_perc_create_bbox
        
    num_bbox = len(list_bbox)
    new_generate_box = int(num_bbox*perc_create_bbox)
    
    max_ratio_bbox = 0.2
    min_size_bbox_px = 10
    
    if bbox_delete:
        new_list_bbox = []
        for bbox in noisy_list_bbox:
            # deletes the perc_create_bbox % of the bboxes
            if random.random() > prob_delete_bbox:
                new_list_bbox.append(bbox)
        noisy_list_bbox = new_list_bbox
               
    for bbox in noisy_list_bbox:
        if random_noise:
            #width
            bbox[2] = bbox[2] + random.randint(-max_random_px, max_random_px)
            #height
            bbox[3] = bbox[3] + random.randint(-max_random_px, max_random_px)
        
        if bbox_displacement:
            #xmin
            bbox[0] = bbox[0] + random.randint(-max_displacement_px, max_displacement_px)
            #ymin
            bbox[1] = bbox[1] + random.randint(-max_displacement_px, max_displacement_px)
            
    if bbox_generate:
        for i in range(new_generate_box):
            width = max(int(x_size * max_ratio_bbox * random.random()), min_size_bbox_px)
            height = max(int(x_size * max_ratio_bbox * random.random()), min_size_bbox_px)
            xmin = random.randint(0, x_size - width)
            ymin = random.randint(0, x_size - height) 
            
            noisy_list_bbox.append([xmin, ymin, width, height])

    return noisy_list_bbox
            
            

    