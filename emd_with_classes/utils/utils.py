import cv2
import numpy as np
from pycocotools import mask

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    
    # return the intersection over union value
    return interArea / float(boxAArea + boxBArea - interArea)
   
    
def encode_segmentation(seg, h, w):
    if len(seg) == 0:
        arr_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        size = len(seg) - 1
        tuples = []
        for i in range(0, size, 2):
            hcoord = seg[i+1]
            wcoord = seg[i]
            tuples.append([ wcoord, hcoord])
        img = np.zeros((h, w), dtype=np.uint8)
        polys = np.array([tuples], np.int32)
        arr_mask = cv2.fillConvexPoly(img, polys[0], 1)
    arr_mask = np.asfortranarray(arr_mask)
    seg_encoded = mask.encode(arr_mask)

    return seg_encoded

def sg_intersection_over_union(gt, p, h, w):
    gt_s = encode_segmentation(gt, h, w)
    p_s = encode_segmentation(p, h, w)
    return mask.iou([gt_s], [p_s], [0])[0][0]


def compute_aspect_ratio_of_segmentation(seg):
    if len(seg) == 0:
        aspect_ratio = 0
    else:
        size = len(seg) - 1
        h_coords, w_coords = [], []
        for i in range(0, size, 2):
            h_coords.append(seg[i+1])
            w_coords.append(seg[i])

        # TODO: check +1 for non zero division
        height = max(h_coords) -  min(h_coords) + 1
        width = max(w_coords) - min(w_coords) + 1
        return width /height