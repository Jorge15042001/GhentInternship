import numpy as np


import scipy.stats as stats
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler


from .fault_detector import BaseFaultDetectionAlgorithm

class PCAFaultDetector(BaseFaultDetectionAlgorithm):
    def __init__(self, retained_variance = 0.9, confidence_level = 0.99):
        self.confidence_level = confidence_level 
        self.retained_variance = retained_variance
        self.x_standard_scaler = StandardScaler()
        self.n_samples = 0
        self.n_features = 0
        self.cov_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None 
        self.idx = None
        self.thresholds = None
        
    def train(self, X_train, y_train=None):
        """
        Train the fault detection model.
        
        Parameters
        ----------
        X_train : array-like
        y_train : ignored 
        """
        self.n_samples, self.n_features = X_train.shape
        X_normalized =  self.x_standard_scaler.fit_transform(X_train)

        self.cov_matrix = np.cov(X_normalized, rowvar=False)
        self.eigenvalues, self.eigenvectors = eigh(self.cov_matrix)  # Compute eigen decomposition
        self.idx = np.argsort(self.eigenvalues)[::-1]  # Sort eigenvalues in descending order
        self.eigenvalues, self.eigenvectors = self.eigenvalues[self.idx], self.eigenvectors[:, self.idx]
        
        self.explained_variance_ratio = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues)
        self.n_components = np.argmax(self.explained_variance_ratio >= self.retained_variance) + 1  # Number of retained PCs

        self.P_pc = self.eigenvectors[:, :self.n_components]  # Retained PCs
        self.P_res = self.eigenvectors[:, self.n_components:]  # Residual PCs
        
        theta_1 = np.sum(self.eigenvalues[self.n_components:])
        theta_2 = np.sum(self.eigenvalues[self.n_components:] ** 2)
        theta_3 = np.sum(self.eigenvalues[self.n_components:] ** 3)
        h0 = 1 - (2 * theta_1 * theta_3) / (3 * theta_2 ** 2)
        def get_thresholds(conf_lvl):
            
            c_alpha = stats.norm.ppf(conf_lvl) 
            J_th_SPE = theta_1 * (c_alpha * np.sqrt(2 * theta_2 * h0 ** 2 / theta_1) + 1 + (theta_2 * h0 * (h0 - 1)) / (theta_1 ** 2)) ** (1 / h0)
    
            F_alpha = stats.f.ppf(conf_lvl, self.n_components, self.n_samples - self.n_components)
            J_th_T2 = (self.n_components * (self.n_samples**2 - 1)) / (self.n_samples * (self.n_samples - 1)) * F_alpha
            return J_th_SPE, J_th_T2
        self.J_th_SPE, self.J_th_T2 = get_thresholds(self.confidence_level)

        self.thresholds = [get_thresholds(conf_lvl) for conf_lvl in np.linspace(0,1,100)]
            

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
        X_norm = self.x_standard_scaler.transform(X)
        eigs = self.eigenvalues[:self.n_components]
        residual = X_norm @ self.P_res               # shape (m, n-l)
        SPE_vals = np.sum(residual**2, axis=1)  # shape (m,)

        pc_scores = X_norm @ self.P_pc
        T2_vals = np.sum((pc_scores**2) / eigs, axis=1)  # shape (m,)
        faults = (SPE_vals > self.J_th_SPE) | (T2_vals > self.J_th_T2)
        return faults

    def roc_curve_data(self, X_test, y_test, fault_numbers=[], by_fault_type = True):
        print("computing roc curve data")
        print("computing roc initial steps")
        X_norm = self.x_standard_scaler.transform(X_test)
        eigs = self.eigenvalues[:self.n_components]
        residual = X_norm @ self.P_res               # shape (m, n-l)
        SPE_vals = np.sum(residual**2, axis=1)  # shape (m,)

        pc_scores = X_norm @ self.P_pc
        T2_vals = np.sum((pc_scores**2) / eigs, axis=1)  # shape (m,)

        fault_ids = np.unique(fault_numbers)

        global_roc_data_dict= {
            "Fault Detection Rate" : [],
            "False Alarm Rate" : []
        }
        by_fault_roc_data_dict = {fault_id: {
            "Fault Detection Rate" : [],
            "False Alarm Rate" : []
        } for fault_id in fault_ids}
        
        print("iterating over thresholds")
        for idx,(J_th_SPE, J_th_T2) in enumerate(self.thresholds):
            print(f"iterating over thresholds: {(idx+1)/len(self.thresholds)}")
            faults = (SPE_vals > J_th_SPE) | (T2_vals > J_th_T2)
            global_error_metrics = BaseFaultDetectionAlgorithm.compute_error_metrics(faults, y_test)
            global_roc_data_dict["Fault Detection Rate"].append(global_error_metrics["Fault Detection Rate"])
            global_roc_data_dict["False Alarm Rate"].append(global_error_metrics["False Alarm Rate"])

        if by_fault_type:
            print("computing threshold by fault")
            for idx, fault_num in enumerate(fault_ids):
                print(f"iterating over faults: {(idx+1)/len(fault_ids)}")
                fault_selector = fault_numbers==fault_num
                expected= y_test[fault_selector]
                SPE_vals_fault = SPE_vals[fault_selector]
                T2_vals_fault = T2_vals[fault_selector]
                for (J_th_SPE, J_th_T2) in self.thresholds:
                    predicted = (SPE_vals_fault > J_th_SPE) | (T2_vals_fault > J_th_T2)
                    fault_error_metrics =  BaseFaultDetectionAlgorithm.compute_error_metrics(predicted, expected)
                    by_fault_roc_data_dict[fault_num]["Fault Detection Rate"].append(fault_error_metrics["Fault Detection Rate"])
                    by_fault_roc_data_dict[fault_num]["False Alarm Rate"].append(fault_error_metrics["False Alarm Rate"])
        return {
            "global": global_roc_data_dict,
            "by_fault": by_fault_roc_data_dict,
            "thresholds": self.thresholds
        }
            
            
        
        
    # def evaluate(self, X_test, y_test, roc_curve=False):
    #     """
    #     Evaluate the model performance on test data.

    #     Parameters
    #     ----------
    #     X_test : array-like or DataFrame
    #         Feature data for testing.
    #     y_test : array-like
    #         Ground truth labels or targets.

    #     Returns
    #     -------
    #     dict
    #         Dictionary of evaluation metrics. This base method can be overridden
    #         for custom evaluation logic.
    #     """
    #     # A default implementation could be to compute and return simple metrics
    #     # but we leave it empty for now
    
        