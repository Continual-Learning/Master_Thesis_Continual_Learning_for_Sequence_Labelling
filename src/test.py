import argparse
import csv
import logging
import os
from datetime import datetime, timezone

import pandas as pd
import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
from model.metric import ClassificationMetrics
from utils.util import write_yaml, read_yaml, label_code_to_name, label_name_to_code

logger = logging.getLogger('multi_cls_model:test')
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(name)s - %(levelname)-8s %(message)s")
logger.info("Starting evaluation...")


def flatten(doc_text, line_sep='\n'):
    lines = doc_text.split(sep=line_sep)
    result = ' '.join([line.strip() for line in lines if len(line.strip()) > 0])
    return result


def full_text_preprocessing(raw_text):
    without_footer = raw_text.split("##### DOCUMENT #####")[0]
    result = flatten(without_footer)
    return result


def import_data(config):
    input_dir = config["data_dir"]
    output_dir = config["run_dir"]
    findings_to_evaluate = [find for find in os.listdir(input_dir) if find in label_name_to_code]
    for find in findings_to_evaluate:
        find_dir = os.path.join(input_dir, find)
        logger.info('Importing from %s to %s', find_dir, output_dir)
        manifest = [elem for elem in os.listdir(find_dir) if elem.endswith(".csv")]
        # ensure that only one manifest is present
        assert len(manifest) == 1
        manifest = manifest[0]

        metadata = [row.split(",") for row in
                    open(os.path.join(find_dir, manifest), 'r', encoding='utf8').read().split("\n")
                    if row != ""][1:]

        data_to_export = [["text", "label"]]
        for text_path, label in metadata:
            text_file_name = text_path.split("/")[-1]
            if "1024971244_WAN2341619660_currentstudy.txt" in text_file_name:
                continue

            record_full_path = os.path.join(find_dir, text_file_name)
            raw_text = open(record_full_path, 'r', encoding='utf8').read()
            full_record = full_text_preprocessing(raw_text)
            data_to_export += [[full_record, label]]

        out_path = os.path.join(output_dir, f"{label_name_to_code[find]}_data.csv")
        with open(out_path, "w", encoding='utf8') as file:
            csv.register_dialect("custom", delimiter=",", skipinitialspace=True)
            writer = csv.writer(file, dialect="custom")
            for tup in data_to_export:
                writer.writerow(tup)

    return [label_name_to_code[find] for find in findings_to_evaluate]


def generate_run_config(config, run_dir):
    return {
        "run_dir": run_dir,
        **config
    }


def main(config):
    device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
    tokenizer_path = os.path.join(config["model_path"], "tokenizer")

    logger.info("Creating output directory and run config...")
    os.makedirs(config["output_path"], exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    run_dir = os.path.join(config["output_path"], timestamp)
    os.mkdir(run_dir)
    run_config = generate_run_config(config, run_dir)
    write_yaml(run_config, os.path.join(run_dir, "run_config.yaml"))

    logger.info("Loading data...")
    if not config["input_is_merged"]:
        logger.info("Preprocessing data...")
        label_code_opts = import_data(run_config)
    else:
        raise NotImplementedError("Testing on merged input data not supported yet")

    eval_metrics = []

    for file in os.listdir(config["model_path"]):
        if not file.endswith(".pth"):
            continue
        logger.info("Loading model checkopoint and metadata in %s...", file)

        model_path = os.path.join(config["model_path"], file)
        checkpoint = torch.load(model_path, map_location=device)
        exp_config = checkpoint["config"]["experiment_config"]
        model_config = checkpoint["config"]["model"]

        logger.info('Evaluating model: %s/%s',
                    exp_config["experiment_name"], exp_config["experiment_id"])

        model_config["args"]["model_path"] = tokenizer_path
        model = getattr(module_arch, model_config["type"])(**model_config["args"])

        loss_fn = module_loss.BCELoss(device=device)  # Fixed loss FN

        # prepare model for testing
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()

        for label in label_code_opts:
            logger.info("Evaluating classifier for task: %s/%s",
                        label, label_code_to_name[label])
            # setup data_loader instances
            data_loader = module_data.TextDataloader(
                run_config['run_dir'],
                batch_size=run_config["eval_batch_size"],
                shuffle=False,
                num_workers=2,
                label_code=label,
                tokenizer_path=tokenizer_path
            )

            test_metrics = ClassificationMetrics(f"/{label}", writer=None)
            with torch.no_grad():
                iterator = tqdm(data_loader)
                for data, target in iterator:
                    target = {
                        "labels": target["labels"].to(device),
                        "weights": torch.tensor([1.0] * len(target["labels"])).to(device)
                    }
                    for key, tensor in data.items():
                        data[key] = tensor.to(device)

                    output, _ = model(**data, cls_head_id=label)
                    _, loss = loss_fn(output, target)

                    test_metrics.update(loss, output, target["labels"], log=False)
                    iterator.set_description(f"F1 score: {test_metrics['f1_score']:1.3f}", refresh=True)

            logger.info(test_metrics.metrics())

            final_metrics = {
                "label_code": label,
                "label_name": label_code_to_name[label],
                "experiment_name": exp_config["experiment_name"],
                "experiment_id": exp_config["experiment_id"],
                **{k.split(f"/{label}")[0]: v for k, v in test_metrics.metrics().items()}
            }
            eval_metrics.append(final_metrics)

    result_df = pd.DataFrame(eval_metrics)
    result_df.to_csv(os.path.join(run_config["run_dir"], f"results_{timestamp}.csv"), index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Testing script')
    args.add_argument('-c', '--config', type=str, required=True,
                      help='Path to config file')

    args = vars(args.parse_args())

    config_from_file = read_yaml(args["config"])
    main(config_from_file)
