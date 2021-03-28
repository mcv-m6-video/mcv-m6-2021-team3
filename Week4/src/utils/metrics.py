import os
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment as linear_assignment
from utils.utils import dict_to_list, bbox_overlap

def ssd(vec1, vec2):
    """
    Compute Sum of Squared Distances between two vectors
    :param vec1, vec2: Two vectors
    :return: Computed SSD distance
    """
    assert len(vec1) == len(vec2)
    return sum((vec1 - vec2) ** 2)

def vec_error(gt, det, nchannel=2):
    """
    Computes vectorial distance  
    :param gt: Ground truth vectors
    :param det: Detection vectors
    :return: Computed error
    """
    dist = det[:, :, :nchannel] - gt[:, :, :nchannel]
    error = np.sqrt(np.sum(dist ** 2, axis=2))

    # discard vectors which from occluded areas (occluded = 0)
    non_occluded_idx = gt[:, :, 2] != 0

    return error[non_occluded_idx], error


def compute_MSEN_PEPN(gt=None, det=None, error=None, nchannel=2, th=3):
    """
    Computes the error using the vectorial distance between gt and det
    :param gt: Ground truth values
    :param det: Detection values
    :param nchannel: Number of channels per frame
    :param op: mse or pep error computation
    :param th: threshold value to consider a distance as an error. 
    """

    if error is None:
        assert gt is not None, 'gt is None'
        assert det is not None, 'det is None'
        error = vec_error(gt, det, nchannel)[0]

    msen = np.mean(error)
    pepn = np.sum(error > th) / len(error)

    return msen, pepn

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
    bb = np.array(bb) / resize_factor
    # (xmax - xmin)  / 2  
    x = (bb[2] + bb[0]) / 2
    # (ymax - ymin)  / 2  
    y = (bb[3] + bb[1]) / 2
    
    return (int(x), int(y))

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

def IDF1(gtDB, stDB, threshold = 0.5):
    """
    compute IDF1 metric
    :param gtDB: list with the information of the detections in gt
    :param stDB: list with the information of the detections predicted
    :param threshold: thr to determine if the prediction is FP or FN
    :return: IDF1 (in %)
    """
    st_ids = np.unique(stDB[:, 1])
    gt_ids = np.unique(gtDB[:, 1])
    n_st = len(st_ids)
    n_gt = len(gt_ids)
    groundtruth = [gtDB[np.where(gtDB[:, 1] == gt_ids[i])[0], :]
                   for i in range(n_gt)]
    prediction = [stDB[np.where(stDB[:, 1] == st_ids[i])[0], :]
                  for i in range(n_st)]
    cost = np.zeros((n_gt + n_st, n_st + n_gt), dtype=float)
    cost[n_gt:, :n_st] = sys.maxsize  # float('inf')
    cost[:n_gt, n_st:] = sys.maxsize  # float('inf')

    fp = np.zeros(cost.shape)
    fn = np.zeros(cost.shape)
    # cost matrix of all trajectory pairs
    cost_block, fp_block, fn_block = cost_between_gt_pred(
        groundtruth, prediction, threshold)

    cost[:n_gt, :n_st] = cost_block
    fp[:n_gt, :n_st] = fp_block
    fn[:n_gt, :n_st] = fn_block

    # computed trajectory match no groundtruth trajectory, FP
    for i in range(n_st):
        cost[i + n_gt, i] = prediction[i].shape[0]
        fp[i + n_gt, i] = prediction[i].shape[0]

    # groundtruth trajectory match no computed trajectory, FN
    for i in range(n_gt):
        cost[i, i + n_st] = groundtruth[i].shape[0]
        fn[i, i + n_st] = groundtruth[i].shape[0]
    try:
        matched_indices = linear_assignment(cost)
    except:
        import pdb
        pdb.set_trace()
    nbox_gt = sum([groundtruth[i].shape[0] for i in range(n_gt)])
    nbox_st = sum([prediction[i].shape[0] for i in range(n_st)])

    IDFP = 0
    IDFN = 0
    for matched in zip(*matched_indices):
        IDFP += fp[matched[0], matched[1]]
        IDFN += fn[matched[0], matched[1]]
    IDTP = nbox_gt - IDFN
    assert IDTP == nbox_st - IDFP
    # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100

    return IDF1

def corresponding_frame(traj1, len1, traj2, len2):
    """
    Find the matching position in traj2 regarding to traj1
    Assume both trajectories in ascending frame ID
    :param traj1, traj2: trajectories (gt and estimated, respectively)
    :return: the location of the bbox in the new frame 
    """
    p1, p2 = 0, 0
    loc = -1 * np.ones((len1, ), dtype=int)
    while p1 < len1 and p2 < len2:
        if traj1[p1] < traj2[p2]:
            loc[p1] = -1
            p1 += 1
        elif traj1[p1] == traj2[p2]:
            loc[p1] = p2
            p1 += 1
            p2 += 1
        else:
            p2 += 1
    return loc

def cost_between_trajectories(traj1, traj2, threshold):
    """
    Compute the FP and FN matchings
    :param traj1, traj2: trajectories (gt and estimated, respectively)
    :param threshold: threshold used to determine if it is FP or FN
    :return: number of FP and FN
    """
    [npoints1, dim1] = traj1.shape
    [npoints2, dim2] = traj2.shape
    # find start and end frame of each trajectories
    start1 = traj1[0, 0]
    end1 = traj1[-1, 0]
    start2 = traj2[0, 0]
    end2 = traj2[-1, 0]

    # check frame overlap
    has_overlap = max(start1, start2) < min(end1, end2)
    if not has_overlap:
        fn = npoints1
        fp = npoints2
        return fp, fn

    # gt trajectory mapping to st, check gt missed
    matched_pos1 = corresponding_frame(
        traj1[:, 0], npoints1, traj2[:, 0], npoints2)
    # st trajectory mapping to gt, check computed one false alarms
    matched_pos2 = corresponding_frame(
        traj2[:, 0], npoints2, traj1[:, 0], npoints1)
    dist1 = compute_distance(traj1, traj2, matched_pos1)
    dist2 = compute_distance(traj2, traj1, matched_pos2)
    # FN
    fn = sum([1 for i in range(npoints1) if dist1[i] < threshold])
    # FP
    fp = sum([1 for i in range(npoints2) if dist2[i] < threshold])
    return fp, fn

def compute_distance(traj1, traj2, matched_pos):
    """
    Compute the loss hit in traj2 regarding to traj1
    :param traj1, traj2: trajectories (gt and estimated, respectively)
    :param matched_pos: positions matched
    :return: the loss hit between both trajectories
    """
    distance = np.zeros((len(matched_pos), ), dtype=float)
    for i in range(len(matched_pos)):
        if matched_pos[i] == -1:
            continue
        else:
            iou = bbox_overlap(traj1[i, 2:6], traj2[matched_pos[i], 2:6])
            distance[i] = iou
    return distance

def cost_between_gt_pred(groundtruth, prediction, threshold):
    """
    Compute cost between detections in gt and in prediction
    :param groundtruth: ft information
    :param prediction: predicted information
    :param threshold: thr used to determine if FP or FN
    :return: the cost, FP and FN
    """
    n_gt = len(groundtruth)
    n_st = len(prediction)
    cost = np.zeros((n_gt, n_st), dtype=float)
    fp = np.zeros((n_gt, n_st), dtype=float)
    fn = np.zeros((n_gt, n_st), dtype=float)
    for i in range(n_gt):
        for j in range(n_st):
            fp[i, j], fn[i, j] = cost_between_trajectories(
                groundtruth[i], prediction[j], threshold)
            cost[i, j] = fp[i, j] + fn[i, j]
    return cost, fp, fn