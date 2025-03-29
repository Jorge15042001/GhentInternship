
from utils_ import load_models_with_cache
from models_params import models_params
from dataclasses import dataclass
from DatasetManager import DatasetLoader
from datasets import tep_testing_mini, tep_testing
from fault_detection_algorithms.fault_detector import BaseFaultDetectionAlgorithm

models = load_models_with_cache(models_params)


@dataclass
class EvaluationParams:
    dataset: DatasetLoader
    model: BaseFaultDetectionAlgorithm
    roc_curve: bool = True
    by_fault: bool = True
    reevaluate: bool = False


evaluations = dict()


for model_alias, model in models.items():
    evaluations[model_alias] = EvaluationParams(
        dataset=tep_testing_mini,
        model=model,
        roc_curve=True,
        by_fault=True,
        reevaluate=False
    )

#  evaluations["pca_big_dataset"] = EvaluationParams(
#          dataset=tep_testing,
#          model=models["pca_r_scl"],
#          roc_curve=True,
#          by_fault=True,
#          reevaluate=False
#          )
