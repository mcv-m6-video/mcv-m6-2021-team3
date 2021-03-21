import os
import numpy as np
from utils.utils import dict_to_list


def interpolate_bb(bb_first, bb_last, distance):
    bb_first = np.array(bb_first)
    bb_last = np.array(bb_last)
    #interpolate new bbox depending on de distance in frames between first and last bbox
    new_bb = bb_first + (bb_last-bb_first)/distance

    return list(np.round(new_bb,2))

def compute_iou(bb_gt, bb, resize_factor=1):
    """ 
    iou = compute_iou(bb_gt, bb)
    Compute IoU between bboxes from ground truth and a single bbox.
    bb_gt: Ground truth bboxes
        Array of (num, bbox), num:number of boxes, bbox:(xmin,ymin,xmax,ymax)
    bb: Detected bbox
        Array of (bbox,), bbox:(xmin,ymin,xmax,ymax)
    """

    # intersection
    bb = bb / resize_factor

    ixmin = np.maximum(bb_gt[:, 0], bb[0])
    iymin = np.maximum(bb_gt[:, 1], bb[1])
    ixmax = np.minimum(bb_gt[:, 2], bb[2])
    iymax = np.minimum(bb_gt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (bb_gt[:, 2] - bb_gt[:, 0] + 1.) *
           (bb_gt[:, 3] - bb_gt[:, 1] + 1.) - inters)

    return inters / uni


def compute_miou(gt_frame, dets_frame, resize_factor=1):
    """
    Computes the mean iou by averaging the individual iou results.
    :param gt_frame: Ground truth bboxes
    :param dets_frame: list of detected bbox for each frame
    :return: Mean Intersection Over Union value, Standard Deviation of the IoU
    """
    iou = []
    for det in dets_frame:
        iou.append(np.max(compute_iou(gt_frame, det, resize_factor)))

    return np.mean(iou), np.std(iou)

def compute_centroid(bb, resize_factor=1):
    """
    Computes centroid of bb
    :param bb: Detected bbox
    :return: Centroid [x,y] 
    """
    # intersection
    bb = bb / resize_factor
    # (xmax - xmin)  / 2  
    x = (bb[2] + bb[0]) / 2
    # (ymax - ymin)  / 2  
    y = (bb[3] + bb[1]) / 2
    
    return [x, y]

def compute_total_miou(gt, dets, frames):
    """
    Computes miou for every frame being evaluated.
    :param gt: Ground truth bboxes   
    :param dets: list of detected bbox 
    :param frames: Frames names 
    return: Return the total moiu for the given sequence by averaging the resutls
    """

    miou = np.empty(0, )

    for frame in frames:
        if os.name == 'nt':
            frame = frame.replace(os.sep, '/')
        frame_id = (frame.split('/')[-1]).split('.')[0]

        if frame_id in gt.keys() and frame_id in dets.keys() and int(frame_id) > 210:
            gt_frame = np.array(dict_to_list(gt[frame_id], False))
            dets_frame = np.array(dict_to_list(dets[frame_id], False))

            miou = np.hstack((miou, compute_miou(gt_frame, dets_frame)[0]))

    return (np.sum(miou) / len(miou))


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(recs,
             imagenames,
             dets,
             classname='car',
             ovthresh=0.5,
             resize_factor=1,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(recs,
                                imagenames,
                                dets,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    gt_dir: Path to ground truth
    det_dir: Path to detections
    img_dir: Path to images
    det_model : Detection model name
        Name of the txt file where detections are written
    classname: Category name (car)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        if os.name == 'nt':
            imagename = imagename.replace(os.sep, '/')

        imgname = (imagename.split('/')[-1]).split('.')[0]
        try:
            R = [obj for obj in recs[imgname] if obj['name'] == classname]
        except:
            continue
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs[imgname] = {'bbox': bbox,
                               'det': det}

    image_ids = [frame for frame, objs in dets.items() for _ in objs if frame in class_recs.keys()]
    confidence = np.array(
        [obj['confidence'] for frame, objs in dets.items() for obj in objs if frame in class_recs.keys()])
    BB = np.array([obj['bbox'] for frame, objs in dets.items() for obj in objs if frame in class_recs.keys()])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            overlaps = compute_iou(BBGT, bb, resize_factor)

            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
