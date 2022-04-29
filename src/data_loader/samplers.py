from abc import abstractmethod
import math
import typing as t

import torch
from torch.utils.data import Dataset, Sampler


class MultiDatasetSampler(Sampler):
    """
    Base class for samplers sampling from multiple datasets.
    """
    def __init__(self, datasets: t.List[Dataset], batch_size: int, shuffle: bool = True) -> None:
        super(MultiDatasetSampler).__init__()
        self._datasets = datasets
        self._n_datasets = len(datasets)
        self._dataset_lengths = [len(dataset) for dataset in datasets]
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._offsets = [0] + [sum(self._dataset_lengths[:i]) for i in range(1, self._n_datasets)]
        self._n_batches = 0

    def __len__(self):
        return self._n_batches

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError("__iter__ not implemented")


class EqualSamplerWithTruncation(MultiDatasetSampler):
    """
    Sampler that retrieves an equal number of examples per batch from datasets with different lengths.
    Longer datasets are truncated. The last batch may have fewer samples than the specified batch size.

    This sampler is meant to be used with datasets of very different sizes, to ensure balance between the number of
    samples from the different datasets. Some samples from the larger dataset may never be used.
    """
    def __init__(self, datasets: t.List[Dataset], batch_size: int, shuffle: bool = True) -> None:
        if batch_size % len(datasets) != 0:
            raise ValueError("Batch size must be multiple of number of datasets to ensure equal representation.")
        super(EqualSamplerWithTruncation, self).__init__(datasets, batch_size, shuffle)
        self._n_samples = min(self._dataset_lengths)
        self._n_batches = math.ceil(self._n_samples * self._n_datasets / self._batch_size)

    def __iter__(self):
        if self._shuffle:
            indices = [torch.randperm(dset_len) for dset_len in self._dataset_lengths]
        else:
            indices = [torch.arange(0, dset_len) for dset_len in self._dataset_lengths]

        for i in range(0, self._n_samples, self._batch_size // self._n_datasets):
            batch_end = min(self._n_samples, i + self._batch_size // self._n_datasets)
            batch = [offset + index[i:batch_end] for index, offset in zip(indices, self._offsets)]
            batch = torch.cat(batch, dim=0)
            yield batch


class RoundRobinSampler(MultiDatasetSampler):
    """
    Sampler that retrieves an approximately equal number of examples per batch from datasets with different lengths.
    Sampling from the datasets happens in a round robin fashion, until there is no more data in the dataset.
    The last batch may have fewer samples than the specified batch size.

    This sampler is meant to be used with datasets of similar but not identical sizes, to ensure balance between the
    number of samples from the different datasets, without the downsampling enforced by the EqualSamplerWithTruncation.
    """
    def __init__(self, datasets: t.List[Dataset], batch_size: int, shuffle: bool = True) -> None:
        super(RoundRobinSampler, self).__init__(datasets, batch_size, shuffle)
        self._n_samples = sum(self._dataset_lengths)
        self._n_batches = math.ceil(self._n_samples / self._batch_size)

    def __iter__(self):
        if self._shuffle:
            dset_indices = [torch.randperm(dset_len) for dset_len in self._dataset_lengths]
        else:
            dset_indices = [torch.arange(0, dset_len) for dset_len in self._dataset_lengths]

        curr_dataset = 0
        indices = [0] * self._n_datasets
        completed = [False] * self._n_datasets

        while not all(completed):
            batch = []
            while len(batch) < self._batch_size and not all(completed):
                if not completed[curr_dataset]:
                    batch.append(self._offsets[curr_dataset] + dset_indices[curr_dataset][indices[curr_dataset]])
                    indices[curr_dataset] += 1
                    if indices[curr_dataset] >= len(dset_indices[curr_dataset]):
                        completed[curr_dataset] = True
                curr_dataset = (curr_dataset + 1) % self._n_datasets
            yield batch


class RoundRobinSamplerWithOversampling(MultiDatasetSampler):
    """
    Sampler that retrieves an approximately equal number of examples per batch from datasets with different lengths.
    Sampling from the datasets happens in a round robin fashion, until there is no more data in the dataset.
    The last batch may have fewer samples than the specified batch size.

    This sampler is meant to be used with datasets of similar but not identical sizes, to ensure balance between the
    number of samples from the different datasets, without the downsampling enforced by the EqualSamplerWithTruncation.
    """
    def __init__(self, datasets: t.List[Dataset], batch_size: int, shuffle: bool = True) -> None:
        super(RoundRobinSamplerWithOversampling, self).__init__(datasets, batch_size, shuffle)
        self._n_samples = sum(self._dataset_lengths)
        self._n_batches = math.ceil(max(self._dataset_lengths) * self._n_datasets / self._batch_size)

    def __iter__(self):
        if self._shuffle:
            dset_indices = [torch.randperm(dset_len) for dset_len in self._dataset_lengths]
        else:
            dset_indices = [torch.arange(0, dset_len) for dset_len in self._dataset_lengths]

        curr_dataset = 0
        indices = [0] * self._n_datasets
        completed = [False] * self._n_datasets

        while not all(completed):
            batch = []
            while len(batch) < self._batch_size and not all(completed):
                batch.append(self._offsets[curr_dataset] + dset_indices[curr_dataset][indices[curr_dataset]])
                indices[curr_dataset] += 1
                if indices[curr_dataset] >= len(dset_indices[curr_dataset]):
                    completed[curr_dataset] = True
                    indices[curr_dataset] = 0
                curr_dataset = (curr_dataset + 1) % self._n_datasets
            yield batch
