import matplotlib.pyplot as plt
import numpy as np
from DatasetManager import DatasetLoader
from fault_detection_algorithms.EvaluationMetrics import BasicErrorMetricsCalculator, AdvanceErrorMetricsCalculator

from cache_models import (
    train_with_cache, load_model_with_cache, evaluate_with_cache
)

import pandas as pd
from IPython.display import display


def show_error_metrics_comparison_new(
    results,
    fault_ids=None,
    model_list=None,
    show_metrics=True,
    return_metrics=False
):
    # error metrics accessors
    fault_error_metrics = {
        "Fault Detection Rate": lambda x: x.fault_detection_rate,
        "False Detection Rate": lambda x: x.false_detection_rate
    }
    non_fault_error_metrics = {
        "False Alarm Rate": lambda x: x.false_alarm_rate
    }
    #  if fault_error_metrics is None:
    #      fault_error_metrics = ["Fault Detection Rate"]
    #  if non_fault_error_metrics is None:
    #      non_fault_error_metrics = ["False Alarm Rate"]

    fault_metrics_dict = {metric: dict()
                          for metric in fault_error_metrics.keys()}
    non_fault_metrics_dict = {metric: dict()
                              for metric in non_fault_error_metrics.keys()}

    if model_list is None:
        model_list = results.keys()
    for model_alias in model_list:
        if model_alias not in results:
            print(f"Not results available for model with alias:{model_alias}")
            continue

        model_results: BasicErrorMetricsCalculator = results[model_alias]["errors"]
        if fault_ids is None:
            fault_ids = list(model_results.by_fault_ids.keys())+[-1]
        for fault_id in fault_ids:
            if fault_id <= 0:
                continue
            if fault_id not in model_results.by_fault_ids:
                print(f"Not results available for fault with id: {
                      fault_id} on model with alias:{model_alias}")
                continue
            error_metrics: BasicErrorMetricsCalculator = model_results.by_fault_ids[fault_id]
            fault_name = f"IDV({fault_id})"

            for metric_name, fault_metric_dict in fault_metrics_dict.items():
                if fault_name not in fault_metric_dict:
                    fault_metric_dict[fault_name] = dict()
                fault_metric_dict[fault_name][model_alias] = fault_error_metrics[metric_name](
                    error_metrics)
        if -1 in fault_ids:
            for metric_name, fault_metric_dict in fault_metrics_dict.items():
                if "GLOBAL" not in fault_metric_dict:
                    fault_metric_dict["GLOBAL"] = dict()
                fault_metric_dict["GLOBAL"][model_alias] = fault_error_metrics[metric_name](
                    model_results)
        # flase alarm rate

        if 0 in fault_ids:
            fault_name = "IDV(0)"
            for metric_name, nonfault_metric_dict in non_fault_metrics_dict.items():
                if fault_name not in nonfault_metric_dict:
                    nonfault_metric_dict[fault_name] = dict()
                nonfault_metric_dict[fault_name][model_alias] = non_fault_error_metrics[metric_name](
                    model_results.by_fault_ids[0])

    fault_metrics_df = {metric_name: pd.DataFrame.from_dict(fault_metric_dict, orient="index")
                        for metric_name, fault_metric_dict in fault_metrics_dict.items()}
    non_fault_metrics_df = {metric_name: pd.DataFrame.from_dict(nonfault_metric_dict, orient="index")
                            for metric_name, nonfault_metric_dict in non_fault_metrics_dict.items()}

    if show_metrics:
        for metric_name, df in fault_metrics_df.items():
            print()
            print(metric_name)
            display(df)
        for metric_name, df in non_fault_metrics_df.items():
            print()
            print(metric_name)
            display(df)

    if return_metrics:
        return fault_metrics_df, non_fault_metrics_df


def show_error_metrics_comparison(
    results,
    fault_ids=None,
    model_list=None,
    fault_error_metrics=None,
    non_fault_error_metrics=None,
    show_metrics=True,
    return_metrics=False
):
    if fault_error_metrics is None:
        fault_error_metrics = ["Fault Detection Rate"]
    if non_fault_error_metrics is None:
        non_fault_error_metrics = ["False Alarm Rate"]
    fault_metrics_dict = {metric: dict() for metric in fault_error_metrics}
    non_fault_metrics_dict = {metric: dict()
                              for metric in non_fault_error_metrics}

    if model_list is None:
        model_list = results.keys()
    for model_alias in model_list:
        if model_alias not in results:
            print(f"Not results available for model with alias:{model_alias}")
            continue
        model_result_dict = results[model_alias]
        error_metrics_dict = model_result_dict["by_fault"]
        if fault_ids is None:
            fault_ids = list(error_metrics_dict.keys())+[-1]
        for fault_id in fault_ids:
            if fault_id <= 0:
                continue
            if fault_id not in error_metrics_dict:
                print(f"Not results available for fault with id: {
                      fault_id} on model with alias:{model_alias}")
                continue
            error_metrics = error_metrics_dict[fault_id]
            fault_name = f"IDV({fault_id})"

            for metric_name, fault_metric_dict in fault_metrics_dict.items():
                if fault_name not in fault_metric_dict:
                    fault_metric_dict[fault_name] = dict()
                fault_metric_dict[fault_name][model_alias] = error_metrics[metric_name]
        if -1 in fault_ids:
            for metric_name, fault_metric_dict in fault_metrics_dict.items():
                global_error_metrics = model_result_dict["global"]
                if "GLOBAL" not in fault_metric_dict:
                    fault_metric_dict["GLOBAL"] = dict()
                fault_metric_dict["GLOBAL"][model_alias] = global_error_metrics[metric_name]
        # flase alarm rate

        if 0 in fault_ids:
            # false_alarm_rate = error_metrics_dict[0]["False Alarm Rate"]
            fault_name = "IDV(0)"
            for metric_name, nonfault_metric_dict in non_fault_metrics_dict.items():
                if fault_name not in nonfault_metric_dict:
                    nonfault_metric_dict[fault_name] = dict()
                nonfault_metric_dict[fault_name][model_alias] = error_metrics_dict[0][metric_name]

    fault_metrics_df = {metric_name: pd.DataFrame.from_dict(fault_metric_dict, orient="index")
                        for metric_name, fault_metric_dict in fault_metrics_dict.items()}
    non_fault_metrics_df = {metric_name: pd.DataFrame.from_dict(nonfault_metric_dict, orient="index")
                            for metric_name, nonfault_metric_dict in non_fault_metrics_dict.items()}

    if show_metrics:
        for metric_name, df in fault_metrics_df.items():
            print()
            print(metric_name)
            display(df)
        for metric_name, df in non_fault_metrics_df.items():
            print()
            print(metric_name)
            display(df)

    if return_metrics:
        return fault_metrics_df, non_fault_metrics_df


def plot_all_roc_curves(results, fault_ids=list(range(21)), plot_models=None, scatter=False, show_identity=True, scatter_size=1, error_metrics=["SPE"], show_plot=True, save_plot=None):
    # For exactly 21 faults, this fits nicely in a 7 x 3 grid.
    # Adjust rows and cols if you have more or fewer.
    ncols = 3
    nrows, ncols = 7, 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 20))
    axes = axes.flatten()  # Easier to index in a single loop
    n_faults = len(fault_ids)

    if plot_models is None:
        plot_models = list(results.keys())

    if show_identity:
        for i, fault_id in enumerate(fault_ids):
            axes[i].plot([0, 1], [0, 1], linestyle='--', label='indentity')

    for model_alias in plot_models:
        if model_alias not in results:
            continue
        result_dict = results[model_alias]
        for ind_name, roc_dict in result_dict["roc_data"].items():
            if ind_name not in error_metrics:
                continue
        #  roc_dict: AdvanceErrorMetricsCalculator = result_dict["roc_data"]
            fault_dict = roc_dict.by_fault_ids
            result_label = f"{model_alias} ({ind_name})"

            for i, fault_id in enumerate(fault_ids):
                if fault_id not in fault_dict:
                    continue
                tpr = fault_dict[fault_id].fault_detection_rate
                far = fault_dict[fault_id].false_alarm_rate
                if i == 0:
                    tpr = roc_dict.fault_detection_rate
                    far = roc_dict.false_alarm_rate

                # Plot the ROC curve on the i-th axis
                if scatter:
                    axes[i].scatter(
                        far, tpr, label=result_label, s=scatter_size)
                else:
                    axes[i].plot(far, tpr, label=result_label)
                # axes[i].plot([0,1], [0, 1], linestyle='--')

                # Labeling
                axes[i].set_title(f"IDV({fault_id})")
                if i == 0:
                    axes[i].set_title("Global Error")

                axes[i].set_xlabel("False Alarm Rate")
                axes[i].set_ylabel("True Positive Rate")
                axes[i].set_xlim([0, 1.1])
                axes[i].set_ylim([0, 1.1])
                axes[i].legend()

    # If there are leftover subplots (in case n_faults < nrows*ncols), turn them off
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if show_plot:
        plt.show()
    if save_plot is not None:
        plt.savefig(save_plot)


def load_models_with_cache(models_params):
    models = dict()
    for model_alias, model_params in models_params.items():
        model_class = model_params.model
        dataset = model_params.dataset

        model = load_model_with_cache(
            model_class=model_class,
            model_params=model_params.model_params,
            dataset_id=dataset.alias
        )
        models[model_alias] = model

    return models


def train_models_with_cache(models_params):
    #  models_params_ = {mod_alias: mod_params.copy()
    #                    for mod_alias, mod_params in models_params.items()}
    models = dict()
    for model_alias, model_params in models_params.items():
        #  model_class = model_params.pop("model_class")
        #  dataset = model_params.pop("dataset_id")
        #  ignore_cache = model_params.pop("ignore_cache", False)
        model_class = model_params.model
        ignore_cache = model_params.retrain
        dataset = model_params.dataset

        model = train_with_cache(
            model_class=model_class,
            model_params=model_params.model_params,
            dataset_id=dataset.alias,
            dataset_loader=dataset,
            force_retrain=ignore_cache
        )
        models[model_alias] = model

    return models


#  def evaluate_models_with_cache(models, eval_params,dataset:DatasetLoader, recompute=[]):
#      results = dict()
#      for model_alias, model in models.items():
#          result = evaluate_with_cache(
#              model_obj=model,
#              dataset_id=dataset.alias,
#              eval_params=eval_params,
#              dataset=dataset,
#              force_recompute=model_alias in recompute
#          )
#          results[model_alias] = result
#      return results
def evaluate_models_with_cache(evaluations):
    results = dict()
    for model_alias, eval_params in evaluations.items():
        result = evaluate_with_cache(
            model_obj=eval_params.model,
            dataset_id=eval_params.dataset.alias,
            eval_params={"roc_curve": eval_params.roc_curve,
                         "by_fault": eval_params.by_fault},
            dataset=eval_params.dataset,
            force_recompute=eval_params.reevaluate
        )
        results[model_alias] = result
    return results
