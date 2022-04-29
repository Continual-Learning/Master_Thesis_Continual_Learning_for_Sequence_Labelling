import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class TextDataset(Dataset):
    def __init__(self, input_dir: str, task: str, tokenizer_path: str,
                 text_col="text", label_col="label", kd_dataset=False) -> None:
        self._df = pd.read_csv(os.path.join(input_dir, f"{task}_data.csv"))
        self._text_col = text_col
        self._label_col = label_col
        self.task = task
        self._n_samples = len(self._df)
        self._kd_dataset = kd_dataset

        self.n_pos = sum(self._df[self._label_col])
        self.n_neg = self._n_samples - self.n_pos

        self._weights = {0: 1.0, 1: 1.0}  # No weights by default

        self.data = []
        self.labels = []
        self.tasks = []

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        for text, label in zip(self._df[self._text_col], self._df[self._label_col]):
            tokenized_text = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
            # Tokenizer returns batches of 1
            for key, tensor in tokenized_text.items():
                tokenized_text[key] = tensor.squeeze()
            self.data.append(tokenized_text)
            self.labels.append(label)

    def set_weigths(self, w_pos: float, w_neg: float) -> None:
        self._weights = {0: w_neg, 1: w_pos}

    def __len__(self):
        return self._n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.labels[idx]
        target = {"labels": label,
                  "tasks": self.task,
                  "weights": 1.0 if self._kd_dataset else self._weights[label],
                  "kd_sample": self._kd_dataset}
        return self.data[idx], target


class TextDatasetWithEmbeddings(TextDataset):
    def __init__(self, input_dir: str, task: str, tokenizer_path: str,
                 text_col="text", label_col="label") -> None:
        super().__init__(input_dir, task, tokenizer_path, text_col, label_col, False)
        self.embeddings = np.load(os.path.join(input_dir, f"{task}_embeddings.npy")).astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.labels[idx]
        target = {"labels": label,
                  "tasks": self.task,
                  "weights": 1.0 if self._kd_dataset else self._weights[label],
                  "embeddings": self.embeddings[idx]}
        return self.data[idx], target
