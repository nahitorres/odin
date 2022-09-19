import os
import unittest
import math

from odin.classes.timeseries import DatasetTSAnomalyDetection, TimeSeriesType, TSProposalsType


class PeriodicityTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        periodic_ds_path = os.path.join(dir_path, 'data/periodicity/sinusoid.csv')
        non_periodic_ds_path = os.path.join(dir_path, 'data/periodicity/non_periodic.csv')
        
        anomalies_path = os.path.join(dir_path, 'data/periodicity/anomalies.json')
        props_path = [("TEST", os.path.join(dir_path, 'data/periodicity/proposals.csv'), TSProposalsType.LABEL)]

        index_gt = "timestamp"
        index_props = "timestamp"
        cls.periodic_dataset = DatasetTSAnomalyDetection(periodic_ds_path,
                                                        TimeSeriesType.UNIVARIATE,
                                                        anomalies_path=anomalies_path,
                                                        proposals_paths=props_path,
                                                        index_gt=index_gt,
                                                        index_proposals=index_props)
        
        cls.non_periodic_dataset = DatasetTSAnomalyDetection(non_periodic_ds_path,
                                                        TimeSeriesType.UNIVARIATE,
                                                        anomalies_path=anomalies_path,
                                                        proposals_paths=props_path,
                                                        index_gt=index_gt,
                                                        index_proposals=index_props)
        
        
    def test_correctness_periodic_signal(self):
        period, _, is_periodic = self.periodic_dataset.analyze_periodicity()
        true_period = 2*math.pi
        error_period = abs((period-true_period)/true_period)
                
        assert error_period < 0.01, "The error on the period is greater than 1%"
        assert is_periodic == "PERIODIC", "Periodicity has not been detected"
        
    
    def test_correctness_nonperiodic_signal(self):
        _, _, is_periodic = self.non_periodic_dataset.analyze_periodicity()
                
        assert is_periodic != "PERIODIC", "Non-existing periodicity has been detected"
        
        
            
if __name__ == '__main__':
    unittest.main()
