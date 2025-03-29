import numpy as np
import scipy.stats as stats
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


from .fault_detector import BaseFaultDetectionAlgorithm


@dataclass
class PCAFaultDetectorParameters():
    retained_variance: float = 0.9
    confidence_level: float = 0.99
    scale_residuals: bool = False


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

    def __init__(self, pca_parameters: PCAFaultDetectorParameters = PCAFaultDetectorParameters()):
        """
        Initialize the PCA Fault Detector.

        Parameters:
        - retained_variance (float): Fraction of total variance to retain in the principal components.
        - confidence_level (float): Confidence level for threshold calculation.
        """
        self.confidence_level = pca_parameters.confidence_level
        self.retained_variance = pca_parameters.retained_variance
        self.scale_residuals = pca_parameters.scale_residuals
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

    def indicators_names(self):
        return ["SPE", "T2"]

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
        self.explained_variance_ratio = np.cumsum(
            self.eigenvalues) / np.sum(self.eigenvalues)
        self.n_components = np.argmax(
            self.explained_variance_ratio >= self.retained_variance) + 1

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

        if not self.scale_residuals:
            def get_thresholds(conf_lvl):
                c_alpha = stats.norm.ppf(conf_lvl)
                J_th_SPE = theta_1 * (c_alpha * np.sqrt(2 * theta_2 * h0 ** 2 / theta_1) + 1 + (
                    theta_2 * h0 * (h0 - 1)) / (theta_1 ** 2)) ** (1 / h0)
                F_alpha = stats.f.ppf(
                    conf_lvl, self.n_components, self.n_samples - self.n_components)
                J_th_T2 = (self.n_components * (self.n_samples**2 - 1)) / \
                    (self.n_samples * (self.n_samples - 1)) * F_alpha
                return J_th_SPE, J_th_T2
        else:
            train_spe_val, _ = self.compute_indicators(X_train)
            train_spe_val = np.sort(train_spe_val)

            def get_thresholds(conf_lvl):
                th_percentile_idx = min(
                    int(len(train_spe_val)*conf_lvl), len(train_spe_val)-1)
                J_th_SE = train_spe_val[th_percentile_idx]
                F_alpha = stats.f.ppf(
                    conf_lvl, self.n_components, self.n_samples - self.n_components)
                J_th_T2 = (self.n_components * (self.n_samples**2 - 1)) / \
                    (self.n_samples * (self.n_samples - 1)) * F_alpha
                return J_th_SPE, J_th_T2
        indicators = None
        expected = None
        if self.scale_residuals:
            indicators = self.compute_indicators(X_train)
            expected = np.zeros_like(indicators[0], np.bool_)

        # self.roc_parametrers_range(indicators=indicators,expected = expected, conf_lvls=[self.confidence_level])[0]
        self.J_th_SPE, self.J_th_T2 = get_thresholds(self.confidence_level)

        # conf_levs = np.linspace(0,1,100)
        # self.confidence_levels = conf_levs
        # conf_levs = np.linspace(0.5,1,10)

        # self.thresholds = [get_thresholds(conf_lvl) for conf_lvl in conf_levs]

    # def get_thresholds(self,conf_lvl, indicators = None, expected):
    #     # Calculate SPE and T² thresholds
    #     F_alpha = stats.f.ppf(conf_lvl, self.n_components, self.n_samples - self.n_components)
    #     J_th_T2 = (self.n_components * (self.n_samples**2 - 1)) / (self.n_samples * (self.n_samples - 1)) * F_alpha

    #     if self.scale_residuals:
    #         train_spe_val,_ = self.compute_indicators(indicators)
    #         train_spe_val = np.sort(train_spe_val)
    #         th_percentile_idx = min(int(len(train_spe_val)*conf_lvl), len(train_spe_val)-1)
    #         J_th_SPE = train_spe_val[th_percentile_idx]
    #         return J_th_SPE, J_th_T2

    #     else:
    #         theta_1 = np.sum(self.residual_eigenvalues)
    #         theta_2 = np.sum(self.residual_eigenvalues ** 2)
    #         theta_3 = np.sum(self.residual_eigenvalues ** 3)
    #         h0 = 1 - (2 * theta_1 * theta_3) / (3 * theta_2 ** 2)

    #         c_alpha = stats.norm.ppf(conf_lvl)
    #         J_th_SPE = theta_1 * (c_alpha * np.sqrt(2 * theta_2 * h0 ** 2 / theta_1) + 1 + (theta_2 * h0 * (h0 - 1)) / (theta_1 ** 2)) ** (1 / h0)
    #         return J_th_SPE, J_th_T2

    def roc_parametrers_range(self, indicators=None, expected=None, npoints=100, conf_lvls=None):
        """
        Returns threshold pairs for ROC curve computation.
        """
        if conf_lvls is None:
            conf_lvls = np.linspace(0, 1, npoints)
        else:
            conf_lvls = np.array(conf_lvls)
        F_alphas = stats.f.ppf(conf_lvls, self.n_components,
                               self.n_samples - self.n_components)
        T2_thresholds = (self.n_components * (self.n_samples**2 - 1)) / \
            (self.n_samples * (self.n_samples - 1)) * F_alphas

        # if self.scale_residuals:
        #     spe_values = indicators[0]

        #     min_faulty_spe = 0
        #     faulty_spe = spe_values[expected]
        #     if faulty_spe.shape[0] > 0:
        #         min_faulty_spe = np.min(faulty_spe)

        #     max_non_faulty_spe = 1000
        #     nonfaulty_spe= spe_values[np.bitwise_not(expected)]
        #     if nonfaulty_spe.shape[0] > 0:
        #         max_non_faulty_spe = np.max(nonfaulty_spe)
        #     elif faulty_spe.shape[0]>0:
        #         max_non_faulty_spe = np.max(faulty_spe)

        #     SPE_thresholds = conf_lvls*(max_non_faulty_spe-min_faulty_spe)+min_faulty_spe
        #     return np.array([SPE_thresholds, T2_thresholds]).T
        # else:
        #     theta_1 = np.sum(self.residual_eigenvalues)
        #     theta_2 = np.sum(self.residual_eigenvalues ** 2)
        #     theta_3 = np.sum(self.residual_eigenvalues ** 3)
        #     h0 = 1 - (2 * theta_1 * theta_3) / (3 * theta_2 ** 2)
        #     c_alphas = stats.norm.ppf(conf_lvls)
        #     SPE_thresholds = theta_1 * (c_alphas * np.sqrt(2 * theta_2 * h0 ** 2 / theta_1) + 1 + (theta_2 * h0 * (h0 - 1)) / (theta_1 ** 2)) ** (1 / h0)
        #     return np.array([SPE_thresholds, T2_thresholds]).T

        spe_values = indicators[0]

        min_faulty_spe = 0
        faulty_spe = spe_values[expected]
        if faulty_spe.shape[0] > 0:
            min_faulty_spe = np.min(faulty_spe)

        max_non_faulty_spe = 1000
        nonfaulty_spe = spe_values[np.bitwise_not(expected)]
        if nonfaulty_spe.shape[0] > 0:
            max_non_faulty_spe = np.max(nonfaulty_spe)
        elif faulty_spe.shape[0] > 0:
            max_non_faulty_spe = np.max(faulty_spe)
        if max_non_faulty_spe <= min_faulty_spe:
            min_faulty_spe = 0
        selector = (spe_values >= min_faulty_spe) & (
            spe_values <= max_non_faulty_spe)
        spe_values_in_range = spe_values[selector]
        # t2_values_in_range = indicators[1][selector]
        spe_idx = np.argsort(spe_values_in_range)
        spe_values_sorted = spe_values_in_range[spe_idx]
        # t2_values_sorted = t2_values_in_range[spe_idx]

        return np.array([np.percentile(spe_values_sorted, conf_lvls*100), T2_thresholds]).T

        # return self.thresholds

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
        if self.scale_residuals:
            SPE_vals = np.sum((residual_scores**2) /
                              self.residual_eigenvalues, axis=1)
        else:
            SPE_vals = np.sum((residual_scores**2), axis=1)

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
