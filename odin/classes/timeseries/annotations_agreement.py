import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from odin.classes.strings import err_type
from odin.classes.timeseries import DatasetTSAnomalyDetection

class AnnotationAgreement:
    """
    The AnnotatorAgreement class can be used to compare the annotations of different users.
    
    Parameters
    ----------
    in_dataset: DatasetTSAnomalyDetection
        Dataset annotated by different annotators.
    index_col: str
        Name of the column with the index containing the timestamps
    feature_name: str
        Name of the column containing the values of interest (i.e., the ones on which the annotation is done)
    annotations_paths: list of tuples
        List of tuples containing the path of the annotations of an annotator and the annotator's name/identifier  
    """
    
    __mandatory_params_no_dataset = {'in_dataset', 'index_col', 'feature_name', 'annotations_paths'}
    
    
    def __init__(self, in_dataset, index_col, feature_name, annotations_paths):
        
        if in_dataset is None or index_col is None or feature_name is None or annotations_paths is None:
            raise Exception(f"Invalid parameters. Please be sure to specify the following parameters: {self.__mandatory_params_no_dataset}")
            
        if not isinstance(in_dataset, DatasetTSAnomalyDetection):
            raise TypeError(err_type.format("in_dataset"))
            
        if not isinstance(index_col, str):
            raise TypeError(err_type.format("index_col"))            
            
        if not isinstance(feature_name, str):
            raise TypeError(err_type.format("feature_name"))
            
        if not isinstance(annotations_paths, list):
            raise TypeError(err_type.format("annotations_paths"))
            
        for annotation_path in annotations_paths:
            if not os.path.exists(annotation_path[0]):
                raise TypeError(err_type.format('annotations_paths'))
        
        self.dataset = in_dataset.get_observations() # get the dataframe from the dataset
        self.dataset.index = pd.to_datetime(self.dataset.index)
        self.dataset = self.dataset.resample("1min").mean().fillna(0).copy()

        self.annotators = []
        self.annotators_names = []
        for annotator in annotations_paths:
            ann = self.json_to_dataframe(annotator[0]) # the same output as the old .csv format
            ann["start_date"] = pd.to_datetime(ann["start_date"])
            ann["end_date"] = pd.to_datetime(ann["end_date"])
            self.annotators.append(ann.copy())
            self.annotators_names.append(annotator[1])
        
        self.n_annotators = len(self.annotators)

        self.feature_name = feature_name

        self.interannotator_agreement = None
    

    def json_to_dataframe(self, anomaly_path):
        """Converts a JSON file to a Pandas DataFrame

        Parameters
        ----------
        anomaly_path : string
            Path of the anomalies .json file.

        Returns
        -------
        anomalies: DataFrame
            Pandas DataFrame containing the anomalies contained in the file at 'anomaly_path'.
        """
        
        with open(anomaly_path, "r") as f:
            data = json.load(f)
                
        anomalies_dict = dict()
        anomalies_dict["start_date"] = []
        anomalies_dict["end_date"] = []
        anomalies_dict["anomaly_type"] = []

        anomalies_list = data["anomalies"]

        for el in anomalies_list:
            if isinstance(el, str):
                anomalies_dict["start_date"].append(pd.to_datetime(el))
                anomalies_dict["end_date"].append(pd.to_datetime(el))
            elif isinstance(el, list) and len(el) == 2:
                anomalies_dict["start_date"].append(pd.to_datetime(el[0]))
                anomalies_dict["end_date"].append(pd.to_datetime(el[1]))

            anomalies_dict["anomaly_type"] = "unknown"

        anomalies = pd.DataFrame.from_dict(anomalies_dict) 
            
        return anomalies
    
    def show_iou_agreement(self, matrix):
        """Plots the IoU agreement

        Parameters
        ----------
        matrix : 2D array-like

        Returns
        -------
        None
        """
        _, ax = plt.subplots(1, dpi=100)
        sns.heatmap(matrix, ax=ax, annot=True, xticklabels=self.annotators_names, yticklabels=self.annotators_names, cmap="Blues", cbar=False)
        ax.xaxis.set_ticklabels(self.annotators_names, rotation=-45, ha="left")
        ax.yaxis.set_ticklabels(self.annotators_names, rotation=45, va="top")

        plt.title("Annotators agreement")
        plt.tight_layout()
        plt.show()
    
    def show_anomalies_annotated(self):
        '''
        Shows the number of annotated anomalies divided by annotator.
        '''
        plt.close("all")

        plt.figure(figsize=(10, 5))

        for name, annotator in zip(self.annotators_names, self.annotators):
            plt.bar([name],[len(annotator)], label="Annotator: {}".format(name))
        plt.legend()
        plt.xticks(fontsize=14)
        plt.xlabel('Annotators')
        plt.ylabel('# annotated anomalies')

        plt.title('Annotated anomalies')

        plt.show()
    

    def calculate_iou_agreement(self):
        """Computes the IoU agreement between annotators as a matrix.
        
        Returns
        -------
        iou: 2D array-like
            2D matrix containing the IoU agreement between annotators.
        """
        iou = np.zeros((self.n_annotators, self.n_annotators))
        iou_data = self.dataset.copy()

        for i, m in enumerate(self.annotators):
            iou_data["m{}".format(i)] = 0
            for j, r in m.iterrows():
                iou_data.loc[(r["start_date"] <= iou_data.index) & (iou_data.index <= r["end_date"]), "m{}".format(i)] = 1

        self.iou_data = iou_data

        # iou = (A inter B) / (A+B - (A inter B))
        for i in range(0, self.n_annotators):
            iou[i][i] = 1
            for j in range(i+1, self.n_annotators):
                A = np.sum(self.iou_data["m{}".format(i)].values)
                B = np.sum(self.iou_data["m{}".format(j)].values)
                A_inter_B = len(self.iou_data.loc[(self.iou_data["m{}".format(i)]==1) & (self.iou_data["m{}".format(j)]==1)])

                metric = A_inter_B / (A+B-A_inter_B)
                iou[i][j] = metric
                iou[j][i] = metric

        self.show_iou_agreement(iou)
        return iou
    

    def _calculate_interannotator_agreement(self):
        """Computes the interannotator agreement"""
        agreement = np.zeros(len(self.dataset))
        for annotator in self.annotators:
            for i, row in annotator.iterrows():
                agreement[np.where((self.dataset.index.values >= row["start_date"]) & (self.dataset.index.values <= row["end_date"]))] += 1
        agreement = np.around(agreement/self.n_annotators * 100, 2)
        
        interannotator_agreement = self.dataset.copy()
        interannotator_agreement["InterannotatorAgreement"] = agreement
        self.interannotator_agreement = interannotator_agreement.copy()

        print("Done!")

    def get_congruent_anomalies(self, agreement_threshold=50):
        """Gets the congruent anomalies.

        Parameters
        ----------
        agreement_threshold : float, default=50
            The agreement threshold, used to determine whether an anomaly annotated by multiple users should be considered a GT anomaly or not.

        Returns
        -------
        result: DataFrame
            Pandas DataFrame containing the computation results.
        """
        if self.interannotator_agreement is None:
                self._calculate_interannotator_agreement()
        anomalies_analysis = self.interannotator_agreement.copy()
        
        anomalies_analysis["anomaly"] = np.where(anomalies_analysis["InterannotatorAgreement"] >= agreement_threshold, 1, 0)
        anomalies_analysis["start_end"] = anomalies_analysis["anomaly"].values - anomalies_analysis["anomaly"].shift(1).values

        if anomalies_analysis["anomaly"].values[0] == 1:
            v = anomalies_analysis["start_end"].values
            v[0] = 1
            anomalies_analysis["start_end"] = v.copy()
        
        anomalies_analysis["week_day"] = anomalies_analysis.index.dayofweek.values
        anomalies_analysis["hour"] = anomalies_analysis.index.hour.values
        result = []

        max_value = self.dataset.loc[self.dataset[self.feature_name] > 30][self.feature_name].mean()        
        
        for start, end in zip(np.where(anomalies_analysis["start_end"]==1)[0], np.where(anomalies_analysis["start_end"]==-1)[0]):
            r = anomalies_analysis.iloc[start:end]
            info = {
                "day_of_the_week": r.iloc[0]["week_day"],
                "hour_of_the_day": r.iloc[0]["hour"],
                "start_date": r.iloc[0].name,
                "end_date": r.iloc[-1].name,
                "duration (min)": (r.iloc[-1].name - r.iloc[0].name)/np.timedelta64(1,'m'),
                "mean_consumption [W]": r[self.feature_name].mean()
                }
            # Anomaly Type
            if info["mean_consumption [W]"] > max_value*1.1:
                if info["duration (min)"] < 30: #min
                    info["anomaly_type"] = "Spike"
                else:
                    info["anomaly_type"] = "Spike + Continuous"
            elif info["mean_consumption [W]"] > max_value*0.7: 
                if info["duration (min)"] < 30:
                    info["anomaly_type"] = "other"
                else:
                    info["anomaly_type"] = "Continuous ON state"
            elif info["mean_consumption [W]"] < max_value*0.25 and info["duration (min)"] >= 30:
                info["anomaly_type"] = "Continuous OFF state"
            else:
                info["anomaly_type"] = "other"

            result.append(info)
            
        if len(result) == 0:
            print("There are no anomalies on which everybody agrees on.")
            return
            
        result = pd.DataFrame(result)
        
        result["distance_from_previous_anomaly (min)"] = (result["start_date"].values - result["end_date"].shift(1).values)/np.timedelta64(1,'m')
        return result

