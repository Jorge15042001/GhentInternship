import abc
import pickle
import numpy as np

class BaseFaultDetectionAlgorithm(abc.ABC):
    """
    Abstract base class for all fault detection algorithms.
    Defines the key methods and attributes expected in any
    fault detection implementation.
    """

    @abc.abstractmethod
    def train(self, X_train, y_train):
        """
        Train the fault detection model.
        
        Parameters
        ----------
        X_train : array-like or DataFrame
            Feature data for training.
        y_train : array-like
            Labels or targets for training (if applicable).
        """
        pass

    @abc.abstractmethod
    def compute_indicators(self, X):
        """
        Compute the monitoring statistics or indicators (e.g., SPE, T², anomaly scores).
        
        Parameters
        ----------
        X : array-like
            Input feature data.
        
        Returns
        -------
        dict
            Dictionary containing computed indicators.
        """
        pass

    @abc.abstractmethod
    def detect_faults(self, indicators, parmas= None):
        """
        Apply decision rules or thresholds to indicators to detect faults.
        
        Parameters
        ----------
        indicators : dict
            Dictionary of computed indicators from compute_indicators().
        
        Returns
        -------
        array-like
            Binary fault decisions (1 = fault, 0 = no fault).
        """
        pass
    def predict(self, X):
        """
        Make predictions for fault detection.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature data for prediction.

        Returns
        -------
        array-like
            Predictions indicating detected faults (1) or normal status (0).
        """
        return self.detect_faults(self.compute_indicators(X))
    @abc.abstractmethod
    def roc_parametrers_range(self):
        pass
    @abc.abstractmethod
    def use_default_predictor(self):
        pass
    
    def roc_curve_data(self, X_test, y_test, fault_numbers, by_fault_type = True, precomputed_indicators = None):
        """
        compute the data for generating an roc curve

        Parameters
        ----------
        X_test : array-like or DataFrame
            Feature data for testing.
        y_test : array-like
            Ground truth labels or targets.
        fault_numbers: array-like
            id of the fault: 0-> no fault
        by_fault_type: bool
            perform the computation for each fault separately

        Returns
        -------
        dict
            Dictionary of evaluation metrics. This base method can be overridden
            for custom evaluation logic.
        """
        fault_ids = np.unique(fault_numbers)
        roc_parameters_list = self.roc_parametrers_range()

        if precomputed_indicators is not None and self.use_default_predictor():
            indicators = precomputed_indicators
        else:
            indicators = self.compute_indicators(X_test)
            
        
        global_roc_data_dict= {
            "Fault Detection Rate" : [],
            "False Alarm Rate" : []
        }
        by_fault_roc_data_dict = {fault_id: {
            "Fault Detection Rate" : [],
            "False Alarm Rate" : []
        } for fault_id in fault_ids}
        
        print("iterating over roc parameters")
        
        for idx,desicion_params in enumerate(roc_parameters_list):
            print(f"iterating over roc parameters: {(idx+1)/len(roc_parameters_list)}")
            faults = self.detect_faults(indicators, desicion_params)
            global_error_metrics = BaseFaultDetectionAlgorithm.compute_error_metrics(faults, y_test)
            global_roc_data_dict["Fault Detection Rate"].append(global_error_metrics["Fault Detection Rate"])
            global_roc_data_dict["False Alarm Rate"].append(global_error_metrics["False Alarm Rate"])

        if by_fault_type:
            print("computing threshold by fault")
            for idx, fault_num in enumerate(fault_ids):
                print(f"iterating over faults: {(idx+1)/len(fault_ids)}")
                fault_selector = fault_numbers==fault_num
                expected= y_test[fault_selector]
                fault_indicators = [indicator[fault_selector] for indicator in indicators]
                for desicion_params in roc_parameters_list:
                    predicted = self.detect_faults(fault_indicators, desicion_params) 
                    fault_error_metrics =  BaseFaultDetectionAlgorithm.compute_error_metrics(predicted, expected)
                    by_fault_roc_data_dict[fault_num]["Fault Detection Rate"].append(fault_error_metrics["Fault Detection Rate"])
                    by_fault_roc_data_dict[fault_num]["False Alarm Rate"].append(fault_error_metrics["False Alarm Rate"])
        
        return {
            "global": global_roc_data_dict,
            "by_fault": by_fault_roc_data_dict,
            "thresholds": self.thresholds
        }
        
    @classmethod
    def compute_error_metrics(cls, y_pred, y_test):
        """
        compute the data for generating an roc curve

        Parameters
        ----------
        y_pred: array-like 
            predicted labels.
        y_test : array-like
            Ground truth labels or targets.

        Returns
        -------
        dict
            Dictionary of evaluation metrics. This base method can be overridden
            for custom evaluation logic.
        """
        TP = np.sum(y_pred &  y_test)
        FP = np.sum(y_pred & ~y_test)
        FN = np.sum(~y_pred &  y_test)
        TN = np.sum(~y_pred & ~y_test)
    
        fault_detection_rate = TP/(TP+FN)
        false_alarm_rate = FP/(TN+FP)
        false_detection_rate= FP/(TP+FP)
        return {
            "Fault Detection Rate": fault_detection_rate,
            "False Detection Rate": false_detection_rate,
            "False Alarm Rate": false_alarm_rate
        }
    def evaluate(self, X_test, y_test, fault_numbers, roc_curve=False, by_fault_type = True):
        """
        Evaluate the model performance on test data.

        Parameters
        ----------
        X_test : array-like or DataFrame
            Feature data for testing.
        y_test : array-like
            Ground truth labels or targets.

        Returns
        -------
        dict
            Dictionary of evaluation metrics. This base method can be overridden
            for custom evaluation logic.
        """
        precomputed_indicators = None
        if self.use_default_predictor():
            precomputed_indicators = self.compute_indicators(X_test)
            predicted_fault =self.detect_faults(precomputed_indicators)
        else:
            predicted_fault = self.predict(X_test)

        result_dict = {"global": dict(), "by_fault":dict()}

        if by_fault_type:
            for fault_num in np.unique(fault_numbers):
                fault_selector = fault_numbers==fault_num
                predicted= predicted_fault[fault_selector]
                expected= y_test[fault_selector]
                result_dict["by_fault"][fault_num]=BaseFaultDetectionAlgorithm.compute_error_metrics(predicted, expected)
                
        result_dict["global"] = BaseFaultDetectionAlgorithm.compute_error_metrics(predicted_fault, y_test) 
        if roc_curve:
            result_dict["roc_data"] = self.roc_curve_data(X_test, y_test, fault_numbers, by_fault_type, precomputed_indicators)

        return result_dict
            
        
    @classmethod
    def load(cls, filename):
        """
        Class method to load a fault detection model from a pickle file.
        """
        with open(filename, 'rb') as f:
            instance = pickle.load(f)
        # Ensure the loaded object is actually a subclass instance
        # if not isinstance(instance, cls):
        #     raise TypeError("Loaded object is not an instance of this class.")
        return instance
    
    def save(self, filename):
        """
        Instance method to save a fault detection model to a pickle file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)