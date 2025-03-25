import numpy as np
from scipy.stats import f, chi2
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from .fault_detector import BaseFaultDetectionAlgorithm

class PLSFaultDetectorImproved(BaseFaultDetectionAlgorithm):
    """
    Partial Least Squares (PLS) model + fault detection,
    implemented closely to the algorithm in the referenced paper.
    """

    def __init__(self, n_latent=6, alpha=0.99, use_percentiles = False):
        """
        Parameters
        ----------
        n_latent : int
            Number of latent variables (LVs) to extract (γ in the paper).
        alpha : float
            Significance level for threshold calculations (e.g. 0.95 or 0.99).
        """
        self.n_latent = n_latent
        self.alpha = alpha
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.use_percentiles = use_percentiles
        
        
        # Model matrices
        self.P_ = None   # loadings on X   (m x n_latent)
        self.Q_ = None   # loadings on Y   (a x n_latent)
        self.T_ = None   # scores on X     (N x n_latent)
        self.W_ = None   # raw weight vectors w_k^* (m x n_latent)
        self.R_ = None   # orthogonal weight vectors r_k   (m x n_latent)
        
        # Thresholds
        self.T2_threshold_  = None
        self.SPE_threshold_ = None

    def use_default_predictor(self):
        return True
        
    def train(self, X_train, Y_train):
        """
        Off-line PLS model building (training):
          1) Normalize X, Y
          2) Extract latent variables via NIPALS-like method
          3) Store P, Q, T, R
          4) Compute T^2 and SPE thresholds
        """
        # 1. Normalize data
        X = self.x_scaler.fit_transform(X_train)
        Y = self.y_scaler.fit_transform(Y_train)
        
        N, m = X.shape
        self.N = N
        _, a = Y.shape
        n_lv = self.n_latent
        
        # Initialize containers
        T_scores = np.zeros((N, n_lv))   # T
        P_loads  = np.zeros((m, n_lv))   # P
        Q_loads  = np.zeros((a, n_lv))   # Q
        W_weights= np.zeros((m, n_lv))   # w_k^*
        R_orth   = np.zeros((m, n_lv))   # r_k
        
        # We'll deflate X and Y in each iteration
        Xk = X.copy()
        Yk = Y.copy()
        
        # 2. Iteratively compute latent variables
        for k in range(n_lv):
            # Step (w_k, q_k) = arg max  w^T X_k^T Y_k q
            # A convenient approach: largest singular vectors of X_k^T Y_k
            M = Xk.T @ Yk            # shape (m x a)
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            # Largest singular vector => column 0 of U, row 0 of Vt
            w_star = U[:, 0]  # shape (m,)
            q_star = Vt.T[:, 0]  # shape (a,)
            
            #  t_k = X_k w_k^*
            t_k = Xk @ w_star   # shape (N,)
            
            #  p_k = (X_k^T t_k) / (t_k^T t_k)
            denom = np.dot(t_k, t_k)
            p_k = (Xk.T @ t_k) / denom  # shape (m,)
            
            #  c_k = (Y_k^T t_k) / (t_k^T t_k)
            #  (Sometimes called q_k in standard PLS; used to deflate Y)
            c_k = (Yk.T @ t_k) / denom  # shape (a,)
            
            # Deflate X_k and Y_k
            #  X_{k+1} = X_k - t_k p_k^T
            Xk = Xk - np.outer(t_k, p_k)
            #  Y_{k+1} = Y_k - t_k c_k^T
            Yk = Yk - np.outer(t_k, c_k)
            
            # Now store them
            T_scores[:, k] = t_k
            P_loads[:, k]  = p_k
            Q_loads[:, k]  = c_k
            W_weights[:, k]= w_star
            
            # The "r_k" vector is the orthogonalized version of w_star
            # r_1 = w_1^*
            # r_k = ∏_{j=1}^{k-1} (I - w_j^* p_j^T) w_k^*
            r_k = w_star.copy()
            for j in range(k):
                # subtract out the direction w_j^* p_j^T from r_k
                # because (I - a b^T) x = x - a (b^T x)
                w_j = W_weights[:, j]
                p_j = P_loads[:, j]
                r_k = r_k - w_j * np.dot(p_j, r_k)
            R_orth[:, k] = r_k
        
        # Store model parameters
        self.P_ = P_loads
        self.Q_ = Q_loads
        self.T_ = T_scores
        self.W_ = W_weights
        self.R_ = R_orth

        # 3. Compute thresholds for T^2 and SPE
        #    T^2 threshold from F-dist approximation (eq. (8) in paper):
        #    J_th,T^2 = (γ (N^2 - 1)) / (N (N - γ))  *  F_α(γ, N-γ)

        alpha = self.alpha
        gamma = self.n_latent 
        N = self.N
        numer = gamma * (N**2 - 1.0)
        denom = N * (N - gamma)
        f_val = f.ppf(alpha, gamma, N - gamma)  # F_{alpha}(γ, N-γ)
        T2_threshold_ = (numer / denom) * f_val

        print(self.use_percentiles)

        if self.use_percentiles:
            print("using percentiles")
            Xhat = T_scores @ P_loads.T  # shape (N,m)
            E = X - Xhat
            SPE_vals = np.sum(E*E, axis=1)  # per-sample squared norms
            print(SPE_vals.shape)
            SPE_threshold_ = np.percentile(SPE_vals, self.alpha*100)
            print(f"spe: {SPE_threshold_}")
            
        else:
            print("not using percentiles")
            #    SPE threshold from χ^2 approximation
            #    SPE = ||(I - P R^T) x||^2  (use training residuals to estimate g, h)
            # We'll estimate g, h by the sample mean & variance of the training SPE.
            #  1) Reconstruct Xhat = T P^T  => E = X - Xhat
            #  2) compute SPE_i = ||E_i||^2
            Xhat = T_scores @ P_loads.T  # shape (N,m)
            E = X - Xhat
            SPE_vals = np.sum(E*E, axis=1)  # per-sample squared norms
            self.mu  = mu = np.mean(SPE_vals)   # sample mean
            self.var = S = np.var(SPE_vals)    # sample variance
            
            g = S / (2.0 * mu) #if mu>1e-12 else 1.0
            h = (2.0 * (mu**2)) / S  # if S>1e-12 else 2.0
            # threshold = g * χ^2_{α}(h)
            chi2_val = chi2.ppf(alpha, h)
            SPE_threshold_ = g * chi2_val
            print(f"spe: {SPE_threshold_}")
            
        self.SPE_threshold_, self.T2_threshold_= SPE_threshold_, T2_threshold_ #self.roc_parametrers_range(conf_lvls= [self.alpha])[0]
        
        # conf_levs = np.linspace(0,1,100)
        
        # self.thresholds = [get_thresholds(conf_lvl) for conf_lvl in conf_levs]
            
    def roc_parametrers_range(self, indicators = None, expected = None, npoints = 100, conf_lvls= None):
        # 3. Compute thresholds for T^2 and SPE
        #    T^2 threshold from F-dist approximation (eq. (8) in paper):
        #    J_th,T^2 = (γ (N^2 - 1)) / (N (N - γ))  *  F_α(γ, N-γ)
        if conf_lvls is None:
            alphas = np.linspace(0,1, npoints) #**(1/(self.n_latent))
            if self.use_percentiles:
                alphas = np.linspace(0,1, npoints)
        else:
            alphas = conf_lvls
        #print(alphas)
        gamma = self.n_latent 
        N = self.N
        numer = gamma * (N**2 - 1.0)
        denom = N * (N - gamma)
        f_val = f.ppf(alphas, gamma, N - gamma)  # F_{alpha}(γ, N-γ)
        T2_threshold_ = (numer / denom) * f_val
        
        # # from paper eq. (7) or references:  h = 2 μ^2 / var,   g = var / 2 μ
        # # (some references invert these, but we'll keep consistent with eqn)
        # # Actually, the paper says: g = S/(2μ), h = 2μ^2/S
        # # with S=var and μ=mean
        # S = self.var # could also be calculated as the variance of indicators[0] where expected = True
        # mu = self.mu
        # g = S / (2.0 * mu) #if mu>1e-12 else 1.0
        # h = (2.0 * (mu**2)) / S  # if S>1e-12 else 2.0
        # # threshold = g * χ^2_{α}(h)
        # chi2_val = chi2.ppf(alphas, h)
        # SPE_threshold_ = g * chi2_val
        # return np.array([SPE_threshold_, T2_threshold_ ]).T
        
        spe_values = indicators[0]
        
        min_faulty_spe = 0
        faulty_spe = spe_values[expected]
        if faulty_spe.shape[0] > 0:
            min_faulty_spe = np.min(faulty_spe)
            
        max_non_faulty_spe = 1000
        nonfaulty_spe= spe_values[np.bitwise_not(expected)]
        if nonfaulty_spe.shape[0] > 0:
            max_non_faulty_spe = np.max(nonfaulty_spe)
        elif faulty_spe.shape[0]>0:
            max_non_faulty_spe = np.max(faulty_spe)
        if max_non_faulty_spe<= min_faulty_spe:
            min_faulty_spe = 0
        print("spe_thresholds",min_faulty_spe, max_non_faulty_spe)
            
        selector = (spe_values>=min_faulty_spe) & (spe_values<=max_non_faulty_spe)
        spe_values_in_range = spe_values[selector]
        # spe_idx= np.argsort(spe_values_in_range)
        # spe_values_sorted = spe_values_in_range[spe_idx]
        spe_values_sorted = np.sort(spe_values_in_range)
        return np.array([np.percentile(nonfaulty_spe, alphas*100), T2_threshold_]).T

        # SPE_thresholds = conf_lvls*(max_non_faulty_spe-min_faulty_spe)+min_faulty_spe
        # return np.array([SPE_thresholds, T2_thresholds]).T


    def compute_indicators(self, X_new):
        """
        Online (fault detection) for multiple new samples at once.
    
        Parameters
        ----------
        X_new : ndarray of shape (N_new, m)
            Multiple new samples (rows).
    
        Returns
        -------
        fault_codes : ndarray of shape (N_new,)
            Each element is one of: 'fault_x_related_y', 'fault_x_unrelated_y', or 'no_fault'.
        T2_values : ndarray of shape (N_new,)
        SPE_values : ndarray of shape (N_new,)
        """
        # 1) Normalize X_new
        X_norm = self.x_scaler.transform(X_new)
    
        # 2) Compute T^2 for each new sample
        #    T^2 = t^T (Cov(T))^-1 t,  where t = X_norm R.
        N = self.T_.shape[0]
        covT = (1.0 / (N - 1)) * (self.T_.T @ self.T_)  # (n_latent x n_latent)
        invCovT = np.linalg.inv(covT)
    
        t_mat = X_norm @ self.R_  # shape (N_new, n_latent)
    
        # T2_values[i] = t_mat[i] * invCovT * t_mat[i].T
        # We can do this in a vectorized way:
        temp = t_mat @ invCovT             # (N_new, n_latent)
        T2_values = np.einsum('ij,ij->i', temp, t_mat)  # shape (N_new,)
    
        # 3) Compute SPE = || (I - P R^T) x ||^2 = || x - x_hat ||^2
        #    with x_hat = X_norm (R P^T).
        x_hat = t_mat @ self.P_.T                  # shape (N_new, m)
        residual = X_norm - x_hat                  # shape (N_new, m)
        SPE_values = np.sum(residual**2, axis=1)   # shape (N_new,)

        return SPE_values, T2_values


    
    def detect_faults(self, indicators, params=None):
    
        # 4) Compare to thresholds -> fault codes
        T2_thr = self.T2_threshold_
        SPE_thr = self.SPE_threshold_
        
        SPE_values, T2_values = indicators

        if params is not None:
            SPE_thr, T2_thr= params
    
        # fault_codes = np.full(X_new.shape[0], 'no_fault', dtype=object)
    
        # # 'fault_x_related_y' if T^2 exceeds threshold
        # mask_related    = (T2_values > T2_thr)
        # # 'fault_x_unrelated_y' if T^2 <= threshold but SPE exceeds threshold
        # mask_unrelated  = (~mask_related) & (SPE_values > SPE_thr)
    
        # fault_codes[mask_related]   = 'fault_x_related_y'
        # fault_codes[mask_unrelated] = 'fault_x_unrelated_y'
    
        # return fault_codes, T2_values, SPE_values
        return (SPE_values > SPE_thr) | (T2_values > T2_thr)
    

    def detect_(self, x_new):
        """
        Online (fault detection) for a single new sample x_new.

        Steps:
          1) Normalize x_new
          2) Compute T^2 and SPE
          3) Compare with thresholds => 'faulty' or 'fault-free'

        Returns
        -------
        fault_code : str
            'fault_x_related_y'   if T^2 > T2_threshold
            'fault_x_unrelated_y' if SPE > SPE_threshold (but T^2 <= threshold)
            'no_fault'            otherwise
        T2_value : float
            Computed T^2
        SPE_value : float
            Computed SPE
        """
        # 1. Normalize x_new
        x_norm = (x_new - self.x_mean_) / self.x_std_
        
        # 2. T^2 statistic = x^T R (T^T T / (N-1))^-1 R^T x
        # But we do not strictly have to re-invert each time if we reuse T. 
        # We'll do a direct approach:  t = x R, then T^2 = t^T (Cov of T)^-1 t.
        # Approx. use the sample covariance of T: (1/(N-1)) T^T T.
        # Let's do an inverse once:
        N = self.T_.shape[0]
        covT = (1.0/(N-1)) * (self.T_.T @ self.T_)
        invCovT = np.linalg.inv(covT)  # shape (n_latent x n_latent)
        
        t_vec = x_norm @ self.R_  # shape (n_latent,)
        T2_value = t_vec @ invCovT @ t_vec
        
        # 3. SPE = ||(I - P R^T) x||^2
        #    first reconstruct x_hat = (P R^T) x
        x_hat = (self.P_ @ (self.R_.T @ x_norm))
        residual = x_norm - x_hat
        SPE_value = residual @ residual
        
        # Compare to thresholds
        T2_thr  = self.T2_threshold_
        SPE_thr = self.SPE_threshold_
        
        if T2_value > T2_thr:
            return 'fault_x_related_y', T2_value, SPE_value
        elif SPE_value > SPE_thr:
            return 'fault_x_unrelated_y', T2_value, SPE_value
        else:
            return 'no_fault', T2_value, SPE_value


    # --------------- Utility methods ---------------

    def _mean_center_and_scale(self, M):
        """
        Zero-mean, unit-variance normalization for a 2D array M.
        Returns (M_scaled, mean, std).
        """
        mean_ = M.mean(axis=0)
        std_  = M.std(axis=0, ddof=1)
        std_[std_ < 1e-12] = 1.0  # avoid tiny stdev
        M_scaled = (M - mean_) / std_
        return M_scaled, mean_, std_
