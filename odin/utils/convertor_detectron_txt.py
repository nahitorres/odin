import os
import json
import shutil
from tqdm import tqdm, tqdm_notebook
from odin.utils.env import is_notebook
import cv2
import pycocotools.mask as mask_util


class DetectronToTXT:

    def __init__(self, gt_path, predictions_path, output_path, is_segmentation=False):
        self.gt_path = gt_path
        self.predictions_path = predictions_path
        self.output_path = output_path
        self.is_segmentation = is_segmentation
        self.__tqdm = tqdm_notebook if is_notebook() else tqdm
       
    
    def __polygon_from_mask(self, masked_arr):

        contours, _ = cv2.findContours(masked_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
        if len(segmentation) > 0:
            return segmentation[0]
        else:
            return []
    
    
    def __write_masks(self, preds, f):
        for p in preds:
            # rle to mask
            mask = mask_util.decode(p['segmentation'])
            # mask to contours 
            poly = self.__polygon_from_mask(mask)
            if len(poly) == 0:
                continue
            # write imgg_is, score, x1, y1, x2, y2, x3, y3, ....
            out_str = '{} {} '.format(p['image_id'], p['score'])
            out_str += ' '.join([str(point) for point in poly])                                                            
            f.write(out_str + '\n')
    
    
    def __write_bboxes(self, preds, f):
        for p in preds:
            # get bbox
            bbox = [int(c) for c in p['bbox']]
            # write in coco format: img_id, score, x, y, w, h
            f.write('{} {} {} {} {} {}\n'.format(p['image_id'], p['score'], bbox[0], bbox[1], bbox[2], bbox[3]))
    
    
    def convert_and_save(self):

        with open(self.gt_path, 'r') as f:
            gt = json.load(f)
            gt_images = gt['images']
            cats = gt['categories']

        # replace spaces with _ for the txt evaluation
        id_to_cat = {c['id']:c['name'].replace(' ', '_') for c in cats}

        with open(self.predictions_path, 'r') as f:
            preds = json.load(f)

        preds_by_class = dict()

        # load predictions for each image
        for p in preds:
            cat_id = p['category_id']
            if cat_id not in preds_by_class.keys():
                preds_by_class[cat_id] = []
            preds_by_class[cat_id].append(p)

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

        # write output txt for each image
        for cat_id, v in self.__tqdm(preds_by_class.items(), total=len(preds_by_class)):
            cat_name = id_to_cat[cat_id]
            with open(os.path.join(self.output_path, cat_name + '.txt'), 'w') as f:
                if self.is_segmentation:
                    self.__write_masks(v, f)
                else:
                    self.__write_bboxes(v, f)
                
        #return preds_by_class