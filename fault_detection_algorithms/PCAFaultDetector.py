import numpy as np
import scipy.stats as stats
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

from .fault_detector import BaseFaultDetectionAlgorithm

class PCAFaultDetector(BaseFaultDetectionAlgorithm):
    """
    PCA-based Fault Detection Algorithm.
    
    This algorithm performs fault detection using Principal Component Analysis (PCA),
    monitoring two key indicators:
        - Squared Prediction Error (SPE)
        - Hotelling's T² statistic

    The model is trained on normal operation data and detects anomalies by comparing
    computed indicators against statistically derived thresholds.
    """

    def __init__(self, retained_variance=0.9, confidence_level=0.99):
        """
        Initialize the PCA Fault Detector.

        Parameters:
        - retained_variance (float): Fraction of total variance to retain in the principal components.
        - confidence_level (float): Confidence level for threshold calculation.
        """
        self.confidence_level = confidence_level
        self.retained_variance = retained_variance
        self.x_standard_scaler = StandardScaler()
        self.n_samples = 0
        self.n_features = 0
        self.cov_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.retained_eigenvalues = None
        self.residual_eigenvalues = None
        self.thresholds = None

    def use_default_predictor(self):
        return True

    def train(self, X_train, y_train=None):
        """
        Train the PCA model on normal data.

        Parameters:
        - X_train (array-like): Normal operation data.
        """
        self.n_samples, self.n_features = X_train.shape
        X_normalized = self.x_standard_scaler.fit_transform(X_train)

        # Compute covariance matrix and perform eigen decomposition
        self.cov_matrix = np.cov(X_normalized, rowvar=False)
        self.eigenvalues, self.eigenvectors = eigh(self.cov_matrix)
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues, self.eigenvectors = self.eigenvalues[idx], self.eigenvectors[:, idx]

        # Determine number of principal components to retain
        self.explained_variance_ratio = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues)
        self.n_components = np.argmax(self.explained_variance_ratio >= self.retained_variance) + 1

        # Split retained and residual components
        self.P_pc = self.eigenvectors[:, :self.n_components]
        self.P_res = self.eigenvectors[:, self.n_components:]
        self.retained_eigenvalues = self.eigenvalues[:self.n_components]
        self.residual_eigenvalues = self.eigenvalues[self.n_components:]

        # Calculate SPE and T² thresholds
        theta_1 = np.sum(self.residual_eigenvalues)
        theta_2 = np.sum(self.residual_eigenvalues ** 2)
        theta_3 = np.sum(self.residual_eigenvalues ** 3)
        h0 = 1 - (2 * theta_1 * theta_3) / (3 * theta_2 ** 2)

        def get_thresholds(conf_lvl):
            c_alpha = stats.norm.ppf(conf_lvl)
            J_th_SPE = theta_1 * (c_alpha * np.sqrt(2 * theta_2 * h0 ** 2 / theta_1) + 1 + (theta_2 * h0 * (h0 - 1)) / (theta_1 ** 2)) ** (1 / h0)
            F_alpha = stats.f.ppf(conf_lvl, self.n_components, self.n_samples - self.n_components)
            J_th_T2 = (self.n_components * (self.n_samples**2 - 1)) / (self.n_samples * (self.n_samples - 1)) * F_alpha
            return J_th_SPE, J_th_T2

        self.J_th_SPE, self.J_th_T2 = get_thresholds(self.confidence_level)
        self.thresholds = [get_thresholds(conf_lvl) for conf_lvl in np.linspace(0.0, 1, 1000)]

    def roc_parametrers_range(self):
        """
        Returns threshold pairs for ROC curve computation.
        """
        return self.thresholds

    def compute_indicators(self, X):
        """
        Compute SPE and T² indicators for input data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - tuple: SPE values and T² values.
        """
        X_norm = self.x_standard_scaler.transform(X)
        residual_scores = X_norm @ self.P_res
        SPE_vals = np.sum((residual_scores**2) / self.residual_eigenvalues, axis=1)

        pc_scores = X_norm @ self.P_pc
        T2_vals = np.sum((pc_scores**2) / self.retained_eigenvalues, axis=1)

        return SPE_vals, T2_vals

    def detect_faults(self, indicators, params=None):
        """
        Apply fault detection logic based on SPE and T² thresholds.

        Parameters:
        - indicators (tuple): SPE and T² values.
        - params (tuple, optional): Custom thresholds (SPE_threshold, T2_threshold).

        Returns:
        - array: Binary array where 1 indicates a fault.
        """
        SPE_vals, T2_vals = indicators
        SPE_threshold, T2_threshold = self.J_th_SPE, self.J_th_T2

        if params is not None:
            SPE_threshold, T2_threshold = params

        return (SPE_vals > SPE_threshold) | (T2_vals > T2_threshold)
