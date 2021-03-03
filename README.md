# ODIN: an Object Detection and Instance Segmentation Diagnosis Framework

We an error diagnosis framework for object detection and instance segmentation that helps model developers to add meta-annotations to their data sets, to compute  performance metrics split by meta-annotation values, and  to visualize diagnosis reports. The framework supports the popular PASCAL VOC and MS COCO input formats,   is agnostic to the training platform, and can be extended with application- and domain-specific meta-annotations and metrics  with almost no coding.

ANALYZER

Detections: one .txt for category (categoryName.txt) with one line for each detection in the following format

for bounging boxes:

        image_name, confidence_score, min_x, min_y, width, height

for segmentation masks

        image_name, confidence_score, x1, y1, x2, y2, ..., xn, yn

Dataset: a .json file in coco format (see convertor VOCtoCOCO for PASCAL VOC formats and a folder containing the images (no needed for the analysis but for visualization capabilities).

Main reports:
- Performance for property: given a property name it computes the performances for its different values on the different categories
(Can be invoked in different ways and also you can plot the distribution of the property in the different categories)
- False positive impact: per-class analysis  of wrong object detection is supported, including confusion with background, poor localization, confusion with similar classes or confusion with other classes.
(Also distribution of the error in the entire ds, in a single category and the impact it will have to remove the error in the overall performance)
-Property sensitivity and impact:  for each property,  the worst and best performing image subsets can be  computed, with the maximum and minimum AP achieved. The difference between the maximum and minimum AP highlights the sensitivity of AP w.r.t. the property, while the maximum  w.r.t. the overall AP provides insight on the impact of the property onto the AP.

The analyzer can be extended by overrading the function "_evaluation_metric" to use an metric different of AP all points interpolation norm.

VISUALIZER

Visualization capabilities for the dataset (all images,  images with annotation with a certain property and/or class)

ANNOTATOR

The addition of meta-annotation is supported by a Jupyter Notebook that given set of images and a set of valid meta-annotation values allows the developer to iterate on the images and select the appropriate value, which is saved in the format  used for evaluation.


This repository contains three notebooks that presents how to use the different functionalities.

## **Cite this work**
If you use PolimiDL or wish to refer it, please use the following BibTex entry.

```
@inproceedings{torres2020odin,
  title={ODIN: An Object Detection and Instance Segmentation Diagnosis Framework},
  author={Torres, Rocio Nahime and Fraternali, Piero and Romero, Jesus},
  booktitle={European Conference on Computer Vision},
  pages={19--31},
  year={2020},
  organization={Springer}
}
```

## Reference

* [ODIN: An Object Detection and Instance Segmentation Diagnosis Framework](https://link.springer.com/chapter/10.1007%2F978-3-030-65414-6_3). Rocio Nahime Torres, Piero Fraternali and Jesus Romero.
