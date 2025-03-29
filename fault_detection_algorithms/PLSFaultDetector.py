import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from .fault_detector import BaseFaultDetectionAlgorithm


@dataclass
class PLSFaultDetectorParameters:
    n_components: float = 6
    confidence_level: float = 0.99


class PLSFaultDetector(BaseFaultDetectionAlgorithm):
    """
    PLS-based Fault Detection Algorithm.

    Implements Partial Least Squares (PLS) fault detection as described in Yin et al. (2012).
    Monitors SPE and T² statistics for detecting anomalies based on latent variable modeling.
    """

    def __init__(self, pls_params=PLSFaultDetectorParameters()):
        """
        Initialize the PLS Fault Detector.

        Parameters:
        - n_components (int): Number of latent variables to retain.
        - confidence_level (float): Confidence level for threshold calculation.
        """
        self.n_components = pls_params.n_components
        self.confidence_level = pls_params.confidence_level
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def use_default_predictor(self):
        return True

    def train(self, X_train, Y_train):
        """
        Train the PLS model on normal operation data.

        Parameters:
        - X_train (array-like): Input data.
        - Y_train (array-like): Output (quality) variables.
        """
        self.n_samples, self.n_features = X_train.shape

        X_norm = self.x_scaler.fit_transform(X_train)
        Y_norm = self.y_scaler.fit_transform(Y_train)

        T, P, R, Q, U = [], [], [], [], []
        X_residual, Y_residual = X_norm.copy(), Y_norm.copy()

        for _ in range(self.n_components):
            w = X_residual.T @ Y_residual / np.linalg.norm(X_residual.T @ Y_residual)
            t = X_residual @ w
            p = (X_residual.T @ t) / (t.T @ t)
            q = (Y_residual.T @ t) / (t.T @ t)
            u = Y_residual @ q

            T.append(t)
            P.append(p)
            R.append(w)
            Q.append(q)
            U.append(u)

            X_residual -= np.outer(t, p)
            Y_residual -= np.outer(t, q)

        self.T = np.column_stack(T)
        self.P = np.column_stack(P)
        self.R = np.column_stack(R)
        self.Q = np.column_stack(Q)

        # SPE threshold
        SPE_residual = X_residual
        theta_1 = np.sum(np.var(SPE_residual, axis=0))
        theta_2 = np.sum(np.var(SPE_residual, axis=0)**2)
        theta_3 = np.sum(np.var(SPE_residual, axis=0)**3)
        h0 = 1 - (2 * theta_1 * theta_3) / (3 * theta_2**2)

        def get_thresholds(conf_lvl):
            c_alpha = stats.norm.ppf(conf_lvl)
            J_th_SPE = theta_1 * (c_alpha * np.sqrt(2 * theta_2 * h0**2 / theta_1) + 1 + (theta_2 * h0 * (h0 - 1)) / (theta_1**2))**(1 / h0)
    
            # T² threshold
            F_alpha = stats.f.ppf(conf_lvl, self.n_components, self.n_samples - self.n_components)
            J_th_T2 = (self.n_components * (self.n_samples**2 - 1)) / (self.n_samples * (self.n_samples - 1)) * F_alpha
            return J_th_SPE, J_th_T2
        
        self.J_th_SPE, self.J_th_T2 = get_thresholds(self.confidence_level)
        
        conf_levs = np.linspace(0,1,100)
        self.confidence_levels = conf_levs**10
        #conf_levs = np.linspace(0.5,1,10)
        
        self.thresholds = [get_thresholds(conf_lvl) for conf_lvl in self.confidence_levels]

    def roc_parametrers_range(self):
        return self.thresholds

    def compute_indicators(self, X):
        """
        Compute SPE and T² indicators for input data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - tuple: SPE values and T² values.
        """
        X_norm = self.x_scaler.transform(X)
        t_scores = X_norm @ self.R
        X_hat = t_scores @ self.P.T
        residual = X_norm - X_hat

        SPE_vals = np.sum(residual**2, axis=1)
        covariance_T = np.cov(self.T, rowvar=False)
        T2_vals = np.sum(t_scores @ np.linalg.inv(covariance_T) * t_scores, axis=1)

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
