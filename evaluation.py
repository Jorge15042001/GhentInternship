#  from datasets import tep_testing_mini


from utils_ import load_models_with_cache, evaluate_models_with_cache, show_error_metrics_comparison_new, plot_all_roc_curves
from evaluation_params import evaluations

#  from models_params import models_params
#  models = load_models_with_cache(models_params)

#  eval_params = {'roc_curve': True, 'by_fault_type': True}


#  recompute_evaluation = []
# recompute_evaluation = models.keys()

results = evaluate_models_with_cache(evaluations)


ordering = [0, 1, 2, 4, 5, 6, 7, 8, 12, 13, 14,
            17, 18, 10, 11, 16, 19, 20, 3, 9, 15, -1]

#  show_error_metrics_comparison_new(results, ordering, fault_error_metrics=[
#                                "Fault Detection Rate", "False Detection Rate"])
fault_metrics_df, non_fault_metrics_df = show_error_metrics_comparison_new(results, ordering, return_metrics=True)



plot_all_roc_curves(results, show_identity=False, show_plot=False, save_plot="evaluation_results/roc.svg")
plot_all_roc_curves(results, show_identity=False, show_plot=False, save_plot="evaluation_results/roc.png")
