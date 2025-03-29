from DatasetManager import DatasetLoader
import numpy as np
import pandas as pd


tep_dataset_dir = "./datasets/TEP/"

physical_variables = ['xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmeas_10', 'xmeas_11',
                      'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20', 'xmeas_21', 'xmeas_22', ]
manipulated_variables = ['xmv_1', 'xmv_2', 'xmv_3', 'xmv_4',
                         'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11']

chemical_variables = ['xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31',
                      'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40', 'xmeas_41']

tep_fault_free_training = DatasetLoader(
    "tep_fault_free_training",
    (f"{tep_dataset_dir}/TEP_FaultFree_Training.RData",),
    column_selector_x=physical_variables + manipulated_variables,
    column_selector_y=["xmeas_35"]
)


def compute_extra_columns(df):
    expected_fault = df["sample"] > 160
    expected_fault[df["faultNumber"] == 0] = False  # non faulty test

    return pd.DataFrame.from_dict(
        {
            "y": expected_fault,
            "faultNumber": df["faultNumber"],
        }
    )


def row_selector_100_simulations(df):
    return df["simulationRun"] < 100


tep_testing_mini = DatasetLoader(
    "tep_testing_small",
    (f"{tep_dataset_dir}/TEP_Faulty_Testing.RData",
     f"{tep_dataset_dir}/TEP_FaultFree_Testing.RData"),
    column_selector_x=physical_variables + manipulated_variables,
    extra_columns_fn=compute_extra_columns,
    row_selector=row_selector_100_simulations
)

tep_testing = DatasetLoader(
    "tep_testing",
    (f"{tep_dataset_dir}/TEP_Faulty_Testing.RData",
     f"{tep_dataset_dir}/TEP_FaultFree_Testing.RData"),
    column_selector_x=physical_variables + manipulated_variables,
    extra_columns_fn=compute_extra_columns,
)
