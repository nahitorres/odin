# ODIN: pluggable meta-annotations and metrics for the diagnosis of classification and localization

ODIN  is an open source diagnosis framework for generic ML classification tasks and for CV object detection and instance segmentation tasks that lets developers add meta-annotations to their data sets,  compute  performance metrics split by meta-annotation values, and visualize diagnosis reports.  

ODIN  is agnostic to the training platform and  input formats and can be extended with application- and domain-specific meta-annotations and metrics with almost no coding. It integrates a rapid annotation tool for classification, object detection and instance segmentation data sets.

## NEW !
We have added support to the Time Series analysis. The documentation will be soon available!

## Documentation
For the complete documentation, please visit the [Odin website](https://rnt-pmi.github.io/odin-docs/)


### Analyzer

Developers can run all metrics, or a subset thereof, for a single class or a set of classes. The values of all metrics are reported using diagrams of multiple types, which can be visualized and saved. It is also possible to compare multiple models on the same analysis.

#### Evaluation Metrics
<table>
  <thead>
    <tr class="header">
      <th colspan=2>Metrics</th>
      <th>Binary Classification</th>
      <th>Single-label Classification</th>
      <th>Multi-label Classification</th>
      <th>Object Detection</th>
      <th>Instance Segmentation</th>
    </tr>
  </thead>
  <tbody  style="text-align:center;">
    <tr>
      <td rowspan=10><b>Base Metrics</b></td>
      <td>Accuracy</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>Error Rate</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>Precision</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Recall</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>F1 Score</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Average Precision</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Precision-Recall AUC</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>ROC AUC</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>F1 AUC</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Custom Metric</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=3><b>Curves</b></td>
      <td>Precision-Recall</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>F1</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>ROC</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td rowspan=4><b>CAMs Metrics</b></td>
      <td>Global IoU</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>Component IoU</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>Irrelevant Attention</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>Bbox Coverage</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
  </tbody>
</table>

#### Dataset Exploration
<table>
  <thead>
    <tr class="header">
      <th colspan=2>Analysis</th>
      <th>Binary Classification</th>
      <th>Single-label Classification</th>
      <th>Multi-label Classification</th>
      <th>Object Detection</th>
      <th>Instance Segmentation</th>
    </tr>
  </thead>
  <tbody  style="text-align:center;">
    <tr>
      <td rowspan=2><b>Distribution of Classes</b></td>
      <td>Total</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Per-property</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=2><b>Distribution of Properties</b></td>
      <td>Total</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Per-category</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=1><b>Co-occurrence Matrix</b></td>
      <td>Total</td><td >n/a</td><td >n/a</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
  </tbody>
</table>

#### Model Analyses
<table>
  <thead>
    <tr class="header">
      <th colspan=2>Analysis</th>
      <th>Models Comparison</th>
      <th>Binary Classification</th>
      <th>Single-label Classification</th>
      <th>Multi-label Classification</th>
      <th>Object Detection</th>
      <th>Instance Segmentation</th>
    </tr>
  </thead>
  <tbody  style="text-align:center;">
    <tr>
      <td rowspan=1><b>Performance Analysis</b></td>
      <td>Per-property</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=1><b>Sensitivity and Impact Analysis</b></td>
      <td>Per-property</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=2><b>Distribution of TP</b></td>
      <td>Total</td><td><strong>yes</strong></td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Per-property</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=2><b>Distribution of FP</b></td>
      <td>Total</td><td><strong>yes</strong></td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Per-property</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td rowspan=2><b>Distribution of FN</b></td>
      <td>Total</td><td><strong>yes</strong></td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Per-property</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=2><b>Distribution of TN</b></td>
      <td>Total</td><td><strong>yes</strong></td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>Per-property</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td rowspan=3><b>Confusion Matrix</b></td>
      <td>Total</td><td>n/a</td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>Per-category</td><td>n/a</td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>Per-property</td><td>n/a</td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td rowspan=1><b>FP Categorization and Impact*</b></td>
      <td>Per-category</td><td><strong>yes</strong></td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=1><b>FP Trend</b></td>
      <td>Per-category</td><td>no</td><td>no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=1><b>FN Categorization</b></td>
      <td>Per-category</td><td><strong>yes</strong></td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=2><b>Curve Analysis</b></td>
      <td>Total</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Per-category</td><td><strong>yes</strong></td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=2><b>Reliability Analysis</b></td>
      <td>Total</td><td>n/a</td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Per-category</td><td>n/a</td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=2><b>Top-1 Top-5 Analysis</b></td>
      <td>Total</td><td>no</td><td >n/a</td><td><strong>yes</strong></td><td >n/a</td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>Per-category</td><td>no</td><td >n/a</td><td><strong>yes</strong></td><td >n/a</td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td rowspan=1><b>IoU Analysis</b></td>
      <td>Per-category</td><td>no</td><td >n/a</td><td>n/a</td><td >n/a</td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td rowspan=2><b>CAMs Analysis</b></td>
      <td>Total</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td>Per-category</td><td>no</td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td >n/a</td><td >n/a</td>
    </tr>
    <tr>
      <td rowspan=3><b>Performance Summary</b></td>
      <td>Total</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Per-category</td><td><strong>yes</strong></td><td >no</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
    <tr>
      <td>Per-property</td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td><td ><strong>yes</strong></td>
    </tr>
  </tbody>
</table>

<div style="font-size:10px;">*From the previous version, we have modified the counting of the background errors ​for localization problems. For more information, see the documentation <a href="https://rnt-pmi.github.io/odin-docs/correction/fp_background_error.html" target="_blank">here</a>
</div>


### Annotator

#### Properties annotator
Meta-annotation editing is supported by a Jupyter Notebook that given the ground truth and the meta-annotation values, allows the developer to iterate on the samples and select the appropriate value, which is saved in a standard format and can be analysed with the illustrated diagnosis functions.

#### Data set generation
In addition to editing meta-annotations, ODIN also supports the creation of a classification, object detection or instance segmentation data set. The annotator can be configured to associate training labels to data samples and to draw bounding boxes or segmentation masks over images and label them. Moreover, at the same step of the association of the training labels, it is also possible to add meta-annotations. The resulting data set is saved in a standard format and can be analysed with the illustrated diagnosis functions.


### Visualizer
A GUI realized as a Jupyter Notebook enables the inspection of the data set and model predictions. The visualization can be executed:
* on all the samples
* limited to the samples of a class
* limited to the samples with a certain meta-annotation value
* limited to the samples of a class with a given meta-annotation value
* limited to the true positive predicted samples (of all the classes or for the specific ones)
* limited to the false positive predicted samples (of all the classes or for the specific ones)
* limited to the false negative predicted samples (of all the classes or for the specific ones)
* limited to a specific error type of the false positive predicted samples (of all the classes or for the specific ones)

The Visualizer allows also the visualization of the Class Activation Maps.


### Input
For the analysis proposed, we need mainly two inputs: (1) the ground truth containing the observations of the dataset, with their corresponding classes, meta-properties and bbox or segmentation mask when applies (can be generated with the previously described annotators) (2) predictions of the model to evaluate.

For the CAMs analysis, in addition to the ground truth file we need also the Class Activation Maps.

Examples are found in test-data.

#### Ground Truth
For localization, we use COCO format (for Pascal VOC an example of parser is provided).
For classification an adaptation of COCO is employed:
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
For the CAMs analysis, the classification file must contain also the bounding box/segmentation mask annotations:
```
{
        "categories" : [{"id":1, "name":"cat1"}, {"id":2, "name":"cat2"}],
        "observations": [
                {
                        "id": 1,
                        "file_name": "example.png",
                        "category": 1, #--> for single label,
                        "categories": [1, 2] #--> for multi-label
                        "...": "..", # any other annotation
                        "...": ".."
                }
        ]
        "annotations": [
                {
                        "id": 1,
                        "image_id": 1,
                        "bbox": [
                            50,
                            75,
                            483,
                            1323
                        ]
                        "category_id": 2
                }
        ]
}
```

#### Predictions
One .txt for category (categoryName.txt) with one line for each prediction in the following format

For bounding boxes:

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


#### CAMs
One .npy for image (file_name.npy) of size _h_ x _w_ x _c_ where _h_ and _w_ are the height and width of the image and _c_ is the number of categories.


### Customize visualization
Sometimes the classes names or properties values are too long and make the graphs less readable.
When an Analyzer class is instantiated (using the classes of the examples), a json file (by default called properties.json but can be customized) is created. Here we can modify the "display names" of the different attributes.


### This repo

The folder "examples" contains notebooks that go through the main functionalities (Analyzer, Annotator, Visualizer), using the dataset ARTDL (http://www.artdl.org/). Some examples are provided using RacePlanes(https://www.cosmiqworks.org/RarePlanes/) or SpamSMS (https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/). Other notebooks are for you to play with your own dataset.
We tried to make the functions easy to use and intuitive, as can be seen in the examples.

### Pip lib

The framework and its requirements can be install with

```
 pip install -e . --user
 ```
 
 After this command, the framework appears as a pip package and can be used everywhere as 
 
 ```
 from odin import ....
 ```
 
 Every change is reflected in the pip package without requiring any additional installation step.

 N.B. To be able to use the meta-annotation extractor, two more packages need to be installed (tensorflow and mtcnn). Simply execute the following command:

 ```
 pip install tensorflow mtcnn
 ```

### Docker

We provide the _Dockerfile_ to create a docker image.

```
  docker build -t odin .
```
Once the 'odin' image has been created, just run:

```
  docker run -it -p 8888:8888 odin
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

@inproceedings{torres2021odin,
  title={ODIN: Pluggable Meta-annotations and Metrics for the Diagnosis of Classification and Localization},
  author={Torres, Rocio Nahime and Milani, Federico and Fraternali, Piero},
  booktitle={International Conference on Machine Learning, Optimization, and Data Science},
  pages={383--398},
  year={2021},
  organization={Springer}
}

@article{zangrando2022odin,
  title={ODIN TS: A Tool for the Black-Box Evaluation of Time Series Analytics},
  author={Zangrando, Niccol{\`o} and Torres, Rocio Nahime and Milani, Federico and Fraternali, Piero},
  journal={Engineering Proceedings},
  volume={18},
  number={1},
  pages={4},
  year={2022},
  publisher={MDPI}
}
```
