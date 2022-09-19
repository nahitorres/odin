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
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    
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
            tuples.append([wcoord, hcoord])
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
        return 0
    else:
        size = len(seg) - 1
        h_coords, w_coords = [], []
        for i in range(0, size, 2):
            h_coords.append(seg[i+1])
            w_coords.append(seg[i])

        height = max(h_coords) - min(h_coords)
        width = max(w_coords) - min(w_coords)
        if height == 0:
            return 0
        return width / height


def mask_to_boundary(seg, h, w, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    mask = np.zeros((h, w), np.uint8)
    mask = decode_segmentation(mask, seg)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def decode_segmentation(mask, seg):
    pts = [
        np
            .array(seg)
            .reshape(-1, 2)
            .round()
            .astype(int)
    ]
    mask = cv2.fillPoly(mask, pts, 1)

    return mask


def intersection_over_union(gt, det):
    gt_intersection_det = np.bitwise_and(gt, det).sum()
    gt_union_det = gt_intersection_det + np.bitwise_and(gt, np.logical_not(det)).sum() + np.bitwise_and(np.logical_not(gt), det).sum()
    return gt_intersection_det / gt_union_det
