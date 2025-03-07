import matplotlib.pyplot as plt
import numpy as np


def plot_all_roc_curves(roc_dict):
    fault_dict = roc_dict["by_fault"]
    fault_ids = list(fault_dict.keys())
    n_faults = len(fault_ids)

    # For exactly 21 faults, this fits nicely in a 7 x 3 grid.
    # Adjust rows and cols if you have more or fewer.
    nrows, ncols = 7, 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 20))
    axes = axes.flatten()  # Easier to index in a single loop

    for i, fault_id in enumerate(fault_ids):
        tpr = fault_dict[fault_id]["Fault Detection Rate"]
        far = fault_dict[fault_id]["False Alarm Rate"]
        if i == 0: 
            tpr = roc_dict["global"]["Fault Detection Rate"]
            far = roc_dict["global"]["False Alarm Rate"]
        
        # Plot the ROC curve on the i-th axis
        axes[i].plot(far, tpr, label="ROC")
        axes[i].plot([0, 1], [0, 1], linestyle='--', label="Identity")
        
        # Labeling
        axes[i].set_title(f"IDV({fault_id})")
        if i == 0:
            axes[i].set_title("Global Error")
            
        axes[i].set_xlabel("False Alarm Rate")
        axes[i].set_ylabel("True Positive Rate")
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])
        axes[i].legend()


    # If there are leftover subplots (in case n_faults < nrows*ncols), turn them off
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()

# Usage: