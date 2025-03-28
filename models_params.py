from fault_detection_algorithms.PCAFaultDetector import PCAFaultDetector
from fault_detection_algorithms.PLSFaultDetector import PLSFaultDetector 
from fault_detection_algorithms.PLSFaultDetectorImproved import PLSFaultDetectorImproved

models_params = {
    "pca": {
        'model_class': PCAFaultDetector,
        "dataset_id":"tep_fault_free_training",
        'retained_variance': 0.9,
        'confidence_level': 0.99,
        'scale_residuals': False,
        'ignore_cache': True 
     },
    "pca_r_scl": {
        'model_class': PCAFaultDetector,
        "dataset_id":"tep_fault_free_training",
        'retained_variance': 0.9,
        'confidence_level': 0.99,
        'scale_residuals': True,
        'ignore_cache': False, 
    },
    # "pls LVs(6)":{
    #     'model_class': PLSFaultDetector,
    #     "dataset_id":"tep_fault_free_training",
    #     'confidence_level': 0.99,
    #     'n_components': 6,
    #     'ignore_cache': True 
    # },
    # "pls LVs(29)":{
    #     'model_class': PLSFaultDetector,
    #     "dataset_id":"tep_fault_free_training",
    #     'confidence_level': 0.99,
    #     'n_components': 29,
    #     'ignore_cache': False 
    # },
    # "pls LVs(17)":{
    #     'model_class': PLSFaultDetector,
    #     "dataset_id":"tep_fault_free_training",
    #     'confidence_level': 0.99,
    #     'n_components': 17,
    #     'ignore_cache': True 
    # },
    
    "plsi LVs(6)":{
        'model_class': PLSFaultDetectorImproved,
        "dataset_id":"tep_fault_free_training",
        'alpha': 0.99,
        'n_latent': 6,
        'ignore_cache': True,
        'use_percentiles': False
    },
    "plsi LVs(29)":{
        'model_class': PLSFaultDetectorImproved,
        "dataset_id":"tep_fault_free_training",
        'alpha': 0.99,
        'n_latent': 29,
        'ignore_cache': True,
        'use_percentiles': False
    },
    "plsi LVs(6) percentiles":{
        'model_class': PLSFaultDetectorImproved,
        "dataset_id":"tep_fault_free_training",
        'alpha': 0.99,
        'n_latent': 6,
        'ignore_cache': True,
        'use_percentiles': True 
    },
    "plsi LVs(29) percentiles":{
        'model_class': PLSFaultDetectorImproved,
        "dataset_id":"tep_fault_free_training",
        'alpha': 0.99,
        'n_latent': 29,
        'ignore_cache': True ,
        'use_percentiles': True 
    },
    
}