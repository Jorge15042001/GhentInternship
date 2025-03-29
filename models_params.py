from fault_detection_algorithms.PCAFaultDetector import PCAFaultDetector, PCAFaultDetectorParameters
from fault_detection_algorithms.PLSFaultDetector import PLSFaultDetector, PLSFaultDetectorParameters
from fault_detection_algorithms.PLSFaultDetectorImproved import PLSFaultDetectorImproved, PLSFaultDetectorImprovedParameters
from fault_detection_algorithms.fault_detector import BaseFaultDetectionAlgorithm
from datasets import tep_fault_free_training
from DatasetManager import DatasetLoader


from dataclasses import dataclass


@dataclass
class TrainingParams:
    dataset: DatasetLoader
    model: BaseFaultDetectionAlgorithm
    model_params: any
    retrain: bool = False


models_params = dict()


# PCA model

pca_params = PCAFaultDetectorParameters(
    retained_variance=0.9,
    confidence_level=0.99,
    scale_residuals=False
)
models_params["pca"] = TrainingParams(
    dataset=tep_fault_free_training,
    model=PCAFaultDetector,
    model_params=pca_params,
    retrain=False
    )

# PCA scale residuals
pca_params = PCAFaultDetectorParameters(
    retained_variance=0.9,
    confidence_level=0.99,
    scale_residuals=True
)
models_params["pca_r_scl"] = TrainingParams(
    dataset=tep_fault_free_training,
    model=PCAFaultDetector,
    model_params=pca_params,
    retrain=False
    )

# PLS 6 latent variables
pls_params = PLSFaultDetectorImprovedParameters(
    alpha=0.99,
    n_latent=6,
    use_percentiles=False
    )
models_params["pls LVs(6)"] = TrainingParams(
    dataset=tep_fault_free_training,
    model=PLSFaultDetectorImproved,
    model_params=pls_params,
    retrain=False
    )
