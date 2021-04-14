#!/usr/bin/env python

# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------

from time import time
import argparse

from iou_tracker import track_iou
from util import load_mot, save_to_csv


def main(args):
    with open(args.seqmap) as fd:
        seqs = [line.rstrip('\n') for line in fd]

    for idx, seq in enumerate(seqs):
        if seq == "name" or seq == "":
            continue
        else:
            if "DPM" in seq:
                sigma_l = -0.5
                sigma_h = 0.5
                sigma_iou = 0.4
                t_min = 4
            elif "FRCNN" in seq:
                sigma_l = 0.0
                sigma_h = 0.9
                sigma_iou = 0.3
                t_min = 3
            elif "SDP" in seq:
                sigma_l = 0.4
                sigma_h = 0.5
                sigma_iou = 0.2
                t_min = 2
            else:
                print("No detector name found, this could happen with the wrong seqmap seqmap file. "
                      "Please use c10-train.txt, c10-test.txt or c10-all.txt")
                exit()

            det_path = args.benchmark_dir + "/" + seq + "/det/det.txt"
            out_path = args.res_dir + "/" + seq + ".txt"

            detections = load_mot(det_path)

            start = time()
            tracks = track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min)
            end = time()

            num_frames = len(detections)
            print("finished " + seq + " at " + str(int(num_frames / (end - start))) + " fps!")

            save_to_csv(out_path, tracks)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="IOU Tracker MOT17 demo script. The best parameters for each detector "
                                     "are hardcoded.")
    parser.add_argument('-m', '--seqmap', type=str, required=True,
                        help="full path to the seqmap file to evaluate")
    parser.add_argument('-o', '--res_dir', type=str, required=True,
                        help="path to the results directory")
    parser.add_argument('-b', '--benchmark_dir', type=str, required=True,
                        help="path to the sequence directory")

    args = parser.parse_args()
    main(args)
