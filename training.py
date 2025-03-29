

from fast_dataset_open import open_with_cache
from utils_ import train_models_with_cache
from models_params import models_params
from datasets import tep_fault_free_training


#  dataset_dir = "./datasets/TEP/"
#  X_columns = ['xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15',
#               'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11']
#  # TODO: Dataset manager
#  training_fault_free_df = open_with_cache(
#      f"{dataset_dir}/TEP_FaultFree_Training.RData")
#
#
#  X = training_fault_free_df[X_columns].values
#  Y = training_fault_free_df["xmeas_35"].values  # just for pls


models = train_models_with_cache(models_params)
