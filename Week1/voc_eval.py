# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan adapted by Group 3 MCV
# --------------------------------------------------------

import os
import numpy as np
from metrics import compute_iou

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
    confidence = np.array([obj['confidence'] for frame, objs in dets.items() for obj in objs if frame in class_recs.keys()])
    BB = np.array([obj['bbox'] for frame, objs in dets.items() for obj in objs if frame in class_recs.keys()])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
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
            overlaps = compute_iou(BBGT,bb)

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