# ODIN: pluggable meta-annotations and metrics for the diagnosis of classification and localization

ODIN  is an open source diagnosis framework for generic ML classification tasks and for CV object detection and instance segmentation tasks that lets developers add meta-annotations to their data sets,  compute  performance metrics split by meta-annotation values, and visualize diagnosis reports.  

ODIN  is agnostic to the training platform and  input formats and can be extended with application- and domain-specific meta-annotations and metrics with almost no coding. It integrates a rapid annotation tool for classification and object detection data sets.

Developers can run all metrics, or a subset thereof, for a single class or a set of classes. The values of all metrics are reported using diagrams of multiple types, which can be visualized and saved. The following table summarizes the implemented metrics for each supported task.

|Metric / Analysis| |Binary|SL Class.|ML Class.|Obj. Det.|Ins. Seg.|
| ------ | ------ |------ |------ |------ |------ |------ |
|Base Metrics|Accuracy|X|X|X|-|-|
|Base Metrics|Precision|X|X|X|X|X
|Base Metrics|Recall|X|X|X|X|X
|Base Metrics|Average Precision|X|X|X|X|X
|Base Metrics|ROC AUC|X|X|X|-|-
|Base Metrics|Precision Recall AUC|X|X|X|X|X
|Base Metrics|F1 Score AUC|X|X|X|X|X
|Base Metrics|F1 Score|X|X|X|X|X
|Base Metrics|Custom|X|X|X|X|X
|Curves|Precision Recall|X|X|X|X|X
|Curves|F1 Score|X|X|X|X|X
|Curves|ROC|X|X|X|-|-
|Confusion Matrix||X|X|X|-|-
|Metric  per property value||X|X|X|X|X
|Distribution of Properties and Classes||X|X|X|X|X
|Impact Analysis||X|X|X|X|X
|False Positives Analysis||-|X|X|X|X
|FP, TP, FN, TN Distribution||-|X|X|X|X
|Calibration Diagrams||X|X|X|X|X
|Base Report|Total value|X|X|X|X|X
|Base Report|Per-category  value|X|X|X|X|X
|Base Report|Per-property value|X|X|X|X|X

#### Meta-Properties annotator
Meta-annotations can be automatically extracted (e.g., image color space) or manually provided. 
Meta-annotation editing is supported by a Jupyter Notebook that given the ground truth and the meta-annotation values,  allows the developer to iterate on the samples and select the appropriate value, which is saved in a standard format and can be analysed with the illustrated diagnosis functions.

#### Data set generation
In addition to editing meta-annotations, ODIN also supports the creation of a classification or object detection data set. The annotator can be configured to associate training labels to data samples and to draw bounding boxes over images and label them. The resulting data set is saved i a standard format and can be analysed with the illustrated diagnosis functions.

#### Data set visualizer
A GUI realized as a Jupyter Notebook enables the inspection of the data set. The visualization can be executed on all the samples, limited to  the samples of a  class, limited to the samples with a certain meta-annotation value, and limited to the samples of a class with a given meta-annotation value.

#### Input
For the analysis proposed, we need mainly two inputs: (1) the ground truth containing the observations of the dataset, with their corresponding classes, meta-properties and bbox or segmentation mask when applies (can be generated with the previously described annotators) (2) predictions of the model to evaluate.
Examples are found in test-data.

**Ground Truth**
For detection and localization, we use COCO format (for Pascal VOC an example of parser is provided).
For classification an adapation of COCO is employed:
```
{
        "categories" : [{"id":1, "name":"cat1"}, {"id":2, "name":"cat2"}],
        "observations": [ --> # in COCO this is images we renamed it to accept non-images datasets
                {
                        "id": 1,
                        "file_name": "example.png",
                        "category": 1, #--> for single label,
                        "categories": [1, 2] #--> for multi-label
                        "...": "..", # any other annotation
                        "...": ".." 
                }
        ]
}
```


**Predictions:**
One .txt for category (categoryName.txt) with one line for each prediction in the following format

For bounging boxes:

```
image_id, confidence_score, min_x, min_y, width, height
```

For segmentation masks:

```
image_id, confidence_score, x1, y1, x2, y2, ..., xn, yn
```

For classification:

```
observation_id, confidence_score
```

Additionally, record["file_name"] can be used instead of ID for matching the predictions to the GT, providing a parameter indicator "match_on_filename=True", when creating the dataset. 

#### Customize visualization
Sometimes the classes names or properties values are too long and make the graphs less readable.
When an Analyzer class is instanciated (using the classes of the examples), a json file (by default called properties.json but can be customized) is created. Here we can modify the "display names" of the different attributes.
More improvements on graphs customizations will be included in future commits (e.g. labels size), as well as performance improvements to accelerate the analysis execution time.

#### This repo
The folder "examples" contains notebooks that go through the main funcionality (analysis, annotators, visualizations), using the dataset ARTDL (http://www.artdl.org/). Some examples are provided using RacePlanes(https://www.cosmiqworks.org/RarePlanes/) or SpamSMS (https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/). Other notebooks are for you to play with your own dataset.
We tried to make the functions easy to use and intuitive, as can be seen in the esamples, but we're working on a better documentation for future commits.

 ```
 pip install -r requirements.txt
 ```

## Cite this work
If you use ODIN or wish to refer it, please use the following BibTex entry.

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




