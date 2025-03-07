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
        pass
    @abc.abstractmethod
    def roc_curve_data(self, X_test, y_test, fault_numbers, by_fault_type = True):
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
        pass
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
            result_dict["roc_data"] = self.roc_curve_data(X_test, y_test, fault_numbers, by_fault_type)

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