import abc
import pickle
import numpy as np

class BaseFaultDetectionAlgorithm(abc.ABC):
    """
    Abstract base class for fault detection algorithms.

    Provides a standardized interface for implementing different fault detection methods.
    All subclasses must implement the core methods defined here.
    """

    @abc.abstractmethod
    def train(self, X_train, y_train):
        """
        Train the fault detection model.

        Parameters:
        - X_train (array-like): Training data.
        - y_train (array-like): Optional labels (can be None if unsupervised).
        """
        pass

    @abc.abstractmethod
    def compute_indicators(self, X):
        """
        Compute monitoring indicators (e.g., SPE, T²) from input data.

        Parameters:
        - X (array-like): Input feature data.

        Returns:
        - dict or tuple: Computed indicators.
        """
        pass

    @abc.abstractmethod
    def detect_faults(self, indicators, params=None):
        """
        Apply decision logic to indicators to detect faults.

        Parameters:
        - indicators (dict or tuple): Computed monitoring indicators.
        - params (optional): Thresholds or parameters for decision making.

        Returns:
        - array-like: Binary fault decisions (1 = fault, 0 = no fault).
        """
        pass

    def predict(self, X):
        """
        Full prediction pipeline from data to fault decisions.

        Parameters:
        - X (array-like): Input feature data.

        Returns:
        - array-like: Binary fault decisions.
        """
        return self.detect_faults(self.compute_indicators(X))

    @abc.abstractmethod
    def roc_parametrers_range(self):
        """
        Define the range of parameters (e.g., thresholds) to use for ROC curve computation.

        Returns:
        - list: List of parameter sets.
        """
        pass

    @abc.abstractmethod
    def use_default_predictor(self):
        """
        Indicates if the model uses its own internal prediction method.

        Returns:
        - bool: True if using default predictor, False otherwise.
        """
        pass

    def roc_curve_data(self, X_test, y_test, fault_numbers, by_fault_type=True, precomputed_indicators=None):
        """
        Compute ROC curve data for model evaluation.

        Parameters:
        - X_test (array-like): Test data.
        - y_test (array-like): Ground truth labels.
        - fault_numbers (array-like): Fault identifiers.
        - by_fault_type (bool): Compute ROC curves for each fault type separately.
        - precomputed_indicators (optional): Precomputed indicators to reuse.

        Returns:
        - dict: ROC curve data.
        """
        fault_ids = np.unique(fault_numbers)
        roc_parameters_list = self.roc_parametrers_range()

        if precomputed_indicators is not None and self.use_default_predictor():
            indicators = precomputed_indicators
        else:
            indicators = self.compute_indicators(X_test)

        global_roc_data = {"Fault Detection Rate": [], "False Alarm Rate": []}
        by_fault_roc_data = {fid: {"Fault Detection Rate": [], "False Alarm Rate": []} for fid in fault_ids}

        for params in roc_parameters_list:
            faults = self.detect_faults(indicators, params)
            metrics = self.compute_error_metrics(faults, y_test)
            global_roc_data["Fault Detection Rate"].append(metrics["Fault Detection Rate"])
            global_roc_data["False Alarm Rate"].append(metrics["False Alarm Rate"])

        if by_fault_type:
            for fid in fault_ids:
                selector = fault_numbers == fid
                for params in roc_parameters_list:
                    faults = self.detect_faults([ind[selector] for ind in indicators], params)
                    metrics = self.compute_error_metrics(faults, y_test[selector])
                    by_fault_roc_data[fid]["Fault Detection Rate"].append(metrics["Fault Detection Rate"])
                    by_fault_roc_data[fid]["False Alarm Rate"].append(metrics["False Alarm Rate"])

        return {"global": global_roc_data, "by_fault": by_fault_roc_data}

    @classmethod
    def compute_error_metrics(cls, y_pred, y_test):
        """
        Compute standard fault detection metrics.

        Parameters:
        - y_pred (array-like): Predicted labels.
        - y_test (array-like): Ground truth labels.

        Returns:
        - dict: Fault Detection Rate, False Detection Rate, False Alarm Rate.
        """
        TP = np.sum(y_pred & y_test)
        FP = np.sum(y_pred & ~y_test)
        FN = np.sum(~y_pred & y_test)
        TN = np.sum(~y_pred & ~y_test)

        return {
            "Fault Detection Rate": TP / (TP + FN),
            "False Detection Rate": FP / (TP + FP),
            "False Alarm Rate": FP / (TN + FP)
        }

    def evaluate(self, X_test, y_test, fault_numbers, roc_curve=False, by_fault_type=True):
        """
        Evaluate model performance on test data.

        Parameters:
        - X_test (array-like): Test data.
        - y_test (array-like): Ground truth labels.
        - fault_numbers (array-like): Fault identifiers.
        - roc_curve (bool): Whether to compute ROC data.
        - by_fault_type (bool): Compute metrics for each fault type separately.

        Returns:
        - dict: Evaluation results.
        """
        if self.use_default_predictor():
            indicators = self.compute_indicators(X_test)
            predictions = self.detect_faults(indicators)
        else:
            predictions = self.predict(X_test)

        results = {"global": self.compute_error_metrics(predictions, y_test), "by_fault": {}}

        if by_fault_type:
            for fid in np.unique(fault_numbers):
                selector = fault_numbers == fid
                results["by_fault"][fid] = self.compute_error_metrics(predictions[selector], y_test[selector])

        if roc_curve:
            results["roc_data"] = self.roc_curve_data(X_test, y_test, fault_numbers, by_fault_type, indicators)

        return results

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
