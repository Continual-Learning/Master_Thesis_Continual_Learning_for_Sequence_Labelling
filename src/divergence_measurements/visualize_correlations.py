import argparse
from datetime import datetime, timezone
import os

import numpy as np

from correlation_utils import visualize_attention_map
from utils.util import load_pickle, write_yaml, dump_pickle


def main(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    run_dir = os.path.join(output_path, timestamp)
    os.mkdir(run_dir)

    for file in os.listdir(input_path):
        if not file.endswith(".pkl"):
            continue

        input_data = load_pickle(os.path.join(input_path, file))

        metadata = {
            'experiment_name': input_data['experiment_name'],
            'experiment_id': input_data['experiment_id'],
            'dataset': input_data['dataset']
        }

        write_yaml(metadata, os.path.join(run_dir, "metadata.yaml"))

        attention_jsdiv_avg = np.zeros((12, 12))
        for result in input_data["data"]:
            attention_jsdiv_avg += np.array(result["attention_jsdivergence"])

        attention_jsdiv_avg /= len(input_data["data"])

        out_filename = os.path.join(run_dir, f"attention_jsdivergence_{file}.png")
        visualize_attention_map(attention_jsdiv_avg, "JS Divergence", metadata['experiment_name'],
                                metadata["dataset"], out_filename)

        metadata["svcca_correlations"] = 1 - np.array(input_data["svcca_correlations"])
        metadata["jsdivergence_attention"] = attention_jsdiv_avg

        dump_pickle(metadata, os.path.join(run_dir, f"average_divergences_{file}"))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-i', '--input_path', type=str, required=True,
                      help='Path of model for inference')
    args.add_argument('-o', '--output_path', type=str, required=True,
                      help='Output directory to save plot figures to')

    args = vars(args.parse_args())

    main(**args)
