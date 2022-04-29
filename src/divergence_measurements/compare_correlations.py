import argparse
from datetime import datetime, timezone
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from display_names import model_display_name, metric_display_name, title_display_name, dataset_display_name

from correlation_utils import visualize_attention_map
from utils.util import load_pickle


def create_save_plot(correlations, labels, measure, title, out_filename):
    """Save a heatmap containing the correlation values of each layer and head"""
    plt.rcParams.update({'font.size': 32})

    lab_corr = [(lab, corr) for lab, corr in zip(labels, correlations)]
    lab_corr = sorted(lab_corr)
    labels = [t[0] for t in lab_corr]
    correlations = [t[1] for t in lab_corr]

    fig, ax = plt.subplots(figsize=(15, 12))
    for lab, corr in zip(labels, correlations):
        ax.plot(np.arange(len(corr)), corr, label=lab, linewidth=4)

    ax.set_title(title)
    plt.rcParams.update({'font.size': 30})
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.42), ncol=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel(measure)

    fig.subplots_adjust(bottom=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tick_params(axis='both',
                    left=True,
                    top=False,
                    right=False,
                    bottom=False,
                    labelleft=True,
                    labeltop=False,
                    labelright=False,
                    labelbottom=True)
    plt.grid(axis="y")
    plt.savefig(out_filename, format='jpg', dpi=300)


def main(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    run_dir = os.path.join(output_path, timestamp)
    os.mkdir(run_dir)

    input_data = [
        load_pickle(os.path.join(input_path, file)) for file in os.listdir(input_path)
        if file.endswith(".pkl")
    ]

    metadata = [{
            'experiment_name': input_metadata['experiment_name'],
            'experiment_id': input_metadata['experiment_id'],
            'dataset': input_metadata['dataset']
        } for input_metadata in input_data
    ]

    dataset = input_data[0]['dataset']
    experiment_names = set([m['experiment_name'] for m in metadata if model_display_name(m['experiment_name']) != m['experiment_name']])

    divergences = {}

    for name in experiment_names:

        data = [data for data in input_data
                if data['experiment_name'] == name]

        divergences[name] = {}

        divergences[name]["jsdivergence_attention"] = np.array([np.mean(d["jsdivergence_attention"],
                                                                        axis=1) for d in data])
        divergences[name]["svcca_correlations"] = np.array([d["svcca_correlations"] for d in data])

        for corr_type in ["svcca_correlations", "jsdivergence_attention"]:
            divergences[name][f"{corr_type}_error"] = np.std(divergences[name][corr_type], axis=0)
            divergences[name][corr_type] = np.mean(divergences[name][corr_type], axis=0)

    all_div = np.array([d["jsdivergence_attention"] for d in input_data
                        if d["experiment_name"] in experiment_names])

    visualize_attention_map(np.std(all_div, axis=0), "St. dev. of JS divergence", "All experiments",
                            dataset_display_name(dataset), os.path.join(run_dir, "std_js_div.jpg"),
                            vmax=0.25, cmap="RdYlGn_r")
    visualize_attention_map(np.mean(all_div, axis=0), "Average JS divergence", "All experiments",
                            dataset_display_name(dataset), os.path.join(run_dir, "mean_js_div.jpg"))

    last_layer_div = [{
                        "experiment_name": d["experiment_name"],
                        "experiment_id": d["experiment_id"],
                        "svcca_last_layer": d["svcca_correlations"][-1],
                        "jsdivergence_attention": np.mean(d["jsdivergence_attention"][-1])
        }
        for d in input_data]

    pd.DataFrame(last_layer_div).to_csv(os.path.join(run_dir, "last_layer_div.csv"), index=False)

    for corr_type in ["svcca_correlations", "jsdivergence_attention"]:
        out_file = os.path.join(run_dir, f"{corr_type}.jpg")
        correlations = [divergences[name][corr_type] for name in experiment_names if name]
        labels = [f'{model_display_name(name)}' for name in experiment_names]

        create_save_plot(correlations,
                         labels,
                         metric_display_name(corr_type),
                         f"{title_display_name(corr_type)}\n({dataset_display_name(dataset)})",
                         out_file)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Script to compare correlation values across experiments')
    args.add_argument('-i', '--input_path', type=str, required=True,
                      help='Path of model for inference')
    args.add_argument('-o', '--output_path', type=str, required=True,
                      help='Output directory to save plot figures to')

    args = vars(args.parse_args())

    main(**args)
