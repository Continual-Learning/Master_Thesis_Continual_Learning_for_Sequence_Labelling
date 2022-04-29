import argparse
import logging
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast

import model.loss as module_loss
import model.model as module_arch
from model.metric import ClassificationMetrics
from logger.visualization import TensorboardWriter

logger = logging.getLogger('multi_cls_model:evaluate')
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(name)s - %(levelname)-8s %(message)s")
logger.info("Starting evaluation...")


def run_dirs(output_path,
             experiment_id,
             kd=False,
             dataset_id=None,
             embeddings=False):
    eval_root_dir = os.path.join(output_path, "evaluation", experiment_id)
    paths = {
        "experiment_eval": eval_root_dir,
        "KD": os.path.join(eval_root_dir, dataset_id) if kd else None,
        "embeddings": os.path.join(eval_root_dir, "embeddings") if embeddings else None,
        "model": os.path.join(output_path, "models", experiment_id)
    }
    os.makedirs(eval_root_dir, exist_ok=True)
    if kd:
        os.mkdir(paths["KD"])
    if embeddings:
        os.makedirs(paths["embeddings"], exist_ok=True)
    return paths


def main(config):
    kd = config["knowledge_distillation"]
    embeddings = config["store_embeddings"]

    dataset_id = str.strip(config["input_path"].replace("/", "_"), "_")

    logger.info("Creating run directories...")
    paths = run_dirs(config["output_path"],
                     config["experiment_id"],
                     kd, dataset_id, embeddings)

    model_paths = [path for path in os.listdir(paths["model"])
                   if path.endswith(".pth") and "model_best" in path]
    model_paths = [os.path.join(paths["model"], path) for path in model_paths]

    logger.info("Building model architecture...")
    # build model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_paths[0], map_location=device)

    checkpoint_exp = f'{checkpoint["config"]["experiment_config"]["experiment_name"]}'\
                     f'/{checkpoint["config"]["experiment_config"]["experiment_id"]}'
    assert(checkpoint_exp == config["experiment_id"])

    model_config = checkpoint["config"]["model"]
    model_config["args"]["model_path"] = config["tokenizer_path"]
    model = getattr(module_arch, model_config["type"])(**model_config["args"])

    # Currently does not work with L2Loss on embeddings due to different DataLoaders used for train and eval
    loss_fn = module_loss.BCELoss(device=device)

    eval_metrics = []

    tokenizer = BertTokenizerFast.from_pretrained(config["tokenizer_path"])
    batch_size = 64 if embeddings else 512

    logger.info("Loading evaluation data...")
    eval_data = {}

    for file in os.listdir(config["input_path"]):
        if "_data.csv" in file:
            finding = file.split("_data.csv")[0]
            eval_data[finding] = pd.read_csv(os.path.join(config["input_path"], file))

    for model_path in model_paths:
        logger.info(f'Loading checkpoint: {model_path} ...')
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()

        if embeddings:
            writer = TensorboardWriter(paths["embeddings"], logger, True)
            embeddings_list = []
            embedding_labels = []

        for label, data_df in eval_data.items():
            n_reports = len(data_df)
            if kd:
                kd_outputs = {"text": [], "label": [], "pred": []}

            logger.info(f"Evaluating classifier for task: {label}")

            test_metrics = ClassificationMetrics(f"/{label}", writer=None)

            with torch.no_grad():
                softmax = torch.nn.Softmax(dim=1)
                iter = tqdm(range(0, n_reports, batch_size))
                for batch_start in iter:
                    batch_end = min(batch_start + batch_size, n_reports)

                    input_texts = list(data_df["text"][batch_start:batch_end])
                    label_list = list(data_df["label"][batch_start:batch_end])
                    target = {
                        "labels": torch.tensor(label_list).to(device),
                        "weights": torch.tensor([1.0] * (batch_end-batch_start)).to(device)
                    }

                    data = tokenizer(input_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    for key, tensor in data.items():
                        data[key] = tensor.to(device)

                    if embeddings:
                        embedding_labels.extend([
                            [f"{label}", f"{bin_label}", f"{label}_{bin_label}"] for bin_label in target['labels']
                        ])
                        output, hidden = model(**data, cls_head_id=label, output_attn_and_hidden=False)
                        # Only keep output of last layer, embedding of CLS token, all dimensions
                        hidden = hidden.to("cpu")
                        embeddings_list.append(hidden)
                    else:
                        output, _ = model(**data, cls_head_id=label, output_attn_and_hidden=False)

                    if kd:
                        kd_outputs["text"].extend(input_texts)
                        pred = softmax(output)[:, 1].tolist()
                        kd_outputs["pred"].extend(pred)
                        kd_outputs["label"].extend(target["labels"].tolist())

                    # Currently does not work with L2Loss on embeddings due to DataLoader used for eval
                    _, loss = loss_fn(output, target)

                    test_metrics.update(loss, output, target["labels"])
                    iter.set_description(f"F1 score: {test_metrics['f1_score']:1.3f}", refresh=True)

            logger.info(test_metrics.metrics())

            final_metrics = {
                "label": label,
                "model": model_path,
                "dataset": config["input_path"],
                **{k.split(f"/{label}")[0]: v for k, v in test_metrics.metrics().items()}
            }
            eval_metrics.append(final_metrics)

            if kd:
                kd_df = pd.DataFrame(kd_outputs)
                kd_df.to_csv(os.path.join(paths["KD"], f"{label}_data.csv"), index=False)

    if embeddings:  # Store embeddings to tensorboard logdir
        embeddings = torch.cat(embeddings_list)
        writer.set_step(0)
        # Tag is dataset name
        writer.add_embedding(embeddings,
                             embedding_labels,
                             metadata_header=["Task", "Label", "Task_Label"],
                             label_img=None, tag=dataset_id)

    result_df = pd.DataFrame(eval_metrics)
    output_filename = os.path.join(paths["experiment_eval"], dataset_id + ".csv")
    result_df.to_csv(output_filename, index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Evaluate a PyTorch model')
    args.add_argument('-o', '--output_path', default=None, type=str,
                      help='Path to the experiment_config:output_path directory of the training config')
    args.add_argument('-i', '--input_path', default=None, type=str,
                      help='Path to the evaluation data dir')
    args.add_argument('-x', '--experiment_id', default=None, type=str,
                      help='<experiment_name>/<experiment_id> to resume')
    args.add_argument('-t', '--tokenizer_path', default=None, type=str,
                      help='Path to the pretrained tokenizer to use')
    args.add_argument('-k', '--knowledge_distillation', action='store_true',
                      help='Whether to output predictions for KD')
    args.add_argument('-e', '--store_embeddings', action='store_true',
                      help='Whether to output embeddings of last layer to tensorboard')

    args = vars(args.parse_args())
    main(args)
