import argparse
import logging
from multiprocessing.pool import ThreadPool
import os
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm import tqdm

from correlation_utils import sequence_js_divergence, svcca
import data_loader.data_loaders as module_data
import model.model as module_arch
from utils.util import dump_pickle

logger = logging.getLogger('multi_cls_model:extract_attention')
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(name)s - %(levelname)-8s %(message)s")


def extract_correlations(config, pool):
    logger.info("Starting evaluation of model %s...", config['model_path'])
    logger.info("Setting up devices...")
    device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')

    data_dir = config["data_dir"]

    logger.info("Loading model...")
    checkpoint = torch.load(config["model_path"], map_location=device)
    model_config = checkpoint["config"]["model"]

    model = getattr(module_arch, model_config["type"])(**model_config["args"])
    base_model = getattr(module_arch, model_config["type"])(**model_config["args"])

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model = model.to(device)
    base_model = base_model.to(device)
    model.eval()
    base_model.eval()
    experiment_name = checkpoint['config']['experiment_config']['experiment_name']
    experiment_id = checkpoint['config']['experiment_config']['experiment_id']

    label_codes = [file.split("_data.csv")[0] for file in os.listdir(data_dir) if "_data.csv" in file]

    batch_size = 16
    data_loader = module_data.MultiTaskDataloader(
        data_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        label_codes=label_codes,
        tokenizer_path=model_config["args"]["model_path"]
    )

    # Preprare results structure, one element for each evaluation report
    results = []
    hidden_rep_by_layer = [[] for _ in range(13)]
    hidden_rep_by_layer_base = [[] for _ in range(13)]

    with torch.no_grad():
        for data, target in tqdm(data_loader):
            tasks = target["tasks"]
            labels = target["labels"].tolist()
            sequence_lengths = torch.sum(data['attention_mask'], dim=1)
            data = {k: t.to(device) for k, t in data.items()}

            _, hidden, attn = model(**data, cls_head_id=tasks, output_attn_and_hidden=True)
            _, base_hidden, base_attn = base_model(**data, cls_head_id=tasks, output_attn_and_hidden=True)

            attn = torch.stack(attn).permute((1, 0, 2, 3, 4)).to("cpu").numpy()
            hidden = torch.stack(hidden).permute((1, 0, 2, 3)).to("cpu").numpy()

            base_attn = torch.stack(base_attn).permute((1, 0, 2, 3, 4)).to("cpu").numpy()
            base_hidden = torch.stack(base_hidden).permute((1, 0, 2, 3)).to("cpu").numpy()

            async_results = [[[
                pool.apply_async(sequence_js_divergence,
                                 (base_attn[i, lay, h, :s, :s].astype(np.float64),
                                  attn[i, lay, h, :s, :s].astype(np.float64)))
                for h in range(12)]
                for lay in range(12)]
                for i, s in enumerate(sequence_lengths)
            ]
            attn_correlation_jsdiv = [[[res.get() for res in x] for x in y] for y in async_results]

            results.extend([{
                "task": tasks[i],
                "label": labels[i],
                "attention_jsdivergence": attn_corr,
            } for i, attn_corr in enumerate(attn_correlation_jsdiv)])

            for lay in range(13):
                hidden_rep_by_layer[lay].extend([hidden[i, lay, :s] for i, s in enumerate(sequence_lengths)])
                hidden_rep_by_layer_base[lay].extend([base_hidden[i, lay, :s] for i, s in enumerate(sequence_lengths)])

    hidden_rep_by_layer = [np.concatenate(layer) for layer in hidden_rep_by_layer]
    hidden_rep_by_layer_base = [np.concatenate(layer) for layer in hidden_rep_by_layer_base]

    num_tokens = hidden_rep_by_layer[0].shape[0]
    # Retain a consistent % of all tokens
    token_sampling = np.random.choice([False, True], num_tokens, p=[0.98, 0.02])
    logger.info("Retaining %i tokens for SVCCA", sum(token_sampling))

    hidden_rep_by_layer = [layer_rep[token_sampling] for layer_rep in hidden_rep_by_layer]
    hidden_rep_by_layer_base = [layer_rep[token_sampling] for layer_rep in hidden_rep_by_layer_base]

    async_results = [pool.apply_async(svcca, (hidden_rep_by_layer[lay].astype(np.float32),
                                              hidden_rep_by_layer_base[lay].astype(np.float32)))
                     for lay in range(len(hidden_rep_by_layer))]

    corr_by_layer_svcca = [res.get() for res in async_results]

    output = {
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "dataset": data_dir,
        "data": results,
        "svcca_correlations": corr_by_layer_svcca
    }

    return output, experiment_name, experiment_id


def main(config):
    logger.info("Creating output directory and run config...")
    os.makedirs(config["output_path"], exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    run_dir = os.path.join(config["output_path"], timestamp)
    os.mkdir(run_dir)

    model_dir = config["model_path"]

    pool = ThreadPool(processes=64)

    for model in os.listdir(model_dir):
        if model.endswith(".pth"):
            config["model_path"] = os.path.join(model_dir, model)
            outputs, model_name, model_id = extract_correlations(config, pool)
            logger.info("Dumping computed outputs to file")
            dump_pickle(outputs, os.path.join(run_dir, f"model_outputs_{model_name}_{model_id}.pkl"))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Extract attention outputs and hidden representations from model.')
    args.add_argument('-d', '--data_dir', type=str, required=True,
                      help='Inference batch size')
    args.add_argument('-i', '--model_path', type=str, required=True,
                      help='Path of models to compare')
    args.add_argument('-g', '--gpu_id', default=0, type=str, required=False,
                      help='Id of GPU to use, if available, else uses CPU')
    args.add_argument('-o', '--output_path', default="output/evaluation", type=str, required=False,
                      help='Output path')
    args = vars(args.parse_args())

    main(args)
