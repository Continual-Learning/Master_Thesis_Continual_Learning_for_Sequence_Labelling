import os
import torch

import numpy as np
import pandas as pd


def remove_optimizer_from_checkpoints(input_dir, output_dir):
    for model in os.listdir(input_dir):
        if model.endswith(".pth"):
            print(f"Processing model {model}")
            model_checkpoint = torch.load(os.path.join(input_dir, model), map_location='cpu')
            model_checkpoint.pop("optimizer")
            out_file = os.path.join(output_dir, model)
            print(f"Saving model to {out_file}")
            torch.save(model_checkpoint, out_file)


def transfer_embedding_to_parallel_corpus(original_dir, mt_par_dir, embeddings_dir, id_col="filename"):
    embeddings = np.loadtxt(os.path.join(embeddings_dir, 'tensors.tsv'))
    metadata = pd.read_csv(os.path.join(embeddings_dir, 'metadata.tsv'), sep='\t')
    embedding_idx = 0

    os.makedirs(os.path.join(original_dir, "parallel"), exist_ok=True)

    for file in os.listdir(mt_par_dir):
        if "_data" in file:
            embeddings_mt = []
            df_mt = pd.read_csv(os.path.join(mt_par_dir, file))
            df_original = pd.read_csv(os.path.join(original_dir, file))

            report_to_data = {}
            for id in df_original[id_col]:
                assert str(metadata["Task"][embedding_idx]) == file.split("_data")[0],\
                    'Embeddings and original data out of order.'
                report_to_data[id] = embeddings[embedding_idx]
                embedding_idx += 1

            for id in df_mt[id_col]:
                embeddings_mt.append(report_to_data[id])

            df_mt.to_csv(os.path.join(original_dir, "parallel", file), index=False)
            np.save(os.path.join(original_dir, "parallel", file.replace("_data.csv", "_embeddings.npy")), np.stack(embeddings_mt))
