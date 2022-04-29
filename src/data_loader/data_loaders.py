from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset

from data_loader.datasets import TextDataset, TextDatasetWithEmbeddings
import data_loader.samplers as sampler_module


class TextDataloader(DataLoader):
    """
    Text dataset format loader, handling one dataset at a time.
    """
    def __init__(self,
                 data_dir=None,
                 batch_size=64,
                 shuffle=True,
                 num_workers=1,
                 label_code=None,
                 tokenizer_path=None,
                 sampler=None):
        self.label_code = label_code
        self.dataset = TextDataset(data_dir, label_code, tokenizer_path)
        w_pos = len(self.dataset) / (2 * self.dataset.n_pos)
        w_neg = len(self.dataset) / (2 * self.dataset.n_neg)
        self.dataset.set_weigths(w_pos, w_neg)
        if sampler is None:
            super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            sampler = getattr(sampler_module, sampler)([self.dataset], batch_size, shuffle)
            super().__init__(self.dataset, batch_sampler=sampler, num_workers=num_workers)


class MultiTaskDataloader(DataLoader):
    """
    Text dataset format loader handling multiple datasets at a time.
    """
    def __init__(self,
                 data_dir=None,
                 label_col="label",
                 batch_size=64,
                 shuffle=True,
                 num_workers=1,
                 label_codes=None,
                 tokenizer_path=None,
                 sampler=None):
        datasets = [TextDataset(data_dir, label_code, tokenizer_path, label_col=label_col) for label_code in label_codes]

        n_samples = sum([len(dataset) for dataset in datasets])

        n_samples_by_class = {label: {} for label in label_codes}
        for label, dataset in zip(label_codes, datasets):
            n_samples_by_class[label][0] = dataset.n_neg
            n_samples_by_class[label][1] = dataset.n_pos

        task_weight = {
                task: {
                    0: n_samples / (2 * len(label_codes) * n_by_cls[0]),
                    1: n_samples / (2 * len(label_codes) * n_by_cls[1])
                } for task, n_by_cls in n_samples_by_class.items()
            }
        for dataset in datasets:
            dataset.set_weigths(task_weight[dataset.task][1], task_weight[dataset.task][0])
        self.dataset = ConcatDataset(datasets)
        if sampler is None:
            super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            sampler = getattr(sampler_module, sampler)(datasets, batch_size, shuffle)
            super().__init__(self.dataset, batch_sampler=sampler, num_workers=num_workers)


class KDDataloader(DataLoader):
    """
    Text dataset format loader handling ground truth and knowledge distillation dataset.
    """
    def __init__(self,
                 data_dir=None,
                 kd_data_dir=None,
                 kd_label_col="pred",
                 batch_size=64,
                 shuffle=True,
                 num_workers=1,
                 label_codes=None,
                 tokenizer_path=None,
                 sampler="EqualSamplerWithTruncation"):
        gt_datasets = [TextDataset(data_dir, label_code, tokenizer_path, kd_dataset=False)
                       for label_code in label_codes]
        kd_datasets = [TextDataset(kd_data_dir, label_code, tokenizer_path, label_col=kd_label_col, kd_dataset=True)
                       for label_code in label_codes]
        self.dataset = ConcatDataset(gt_datasets + kd_datasets)

        sampler = getattr(sampler_module, sampler)(gt_datasets + kd_datasets, batch_size, shuffle)
        super(KDDataloader, self).__init__(self.dataset, batch_sampler=sampler, num_workers=num_workers)


class EmbeddingKDDataloader(DataLoader):
    """
    Text dataset format loader handling ground truth and previously calculated embeddings.
    """
    def __init__(self,
                 data_dir=None,
                 batch_size=64,
                 shuffle=True,
                 num_workers=1,
                 label_codes=None,
                 tokenizer_path=None,
                 sampler=None):
        datasets = [TextDatasetWithEmbeddings(data_dir, label_code, tokenizer_path)
                    for label_code in label_codes]
        self.dataset = ConcatDataset(datasets)

        if sampler is None:
            super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            sampler = getattr(sampler_module, sampler)(self.dataset, batch_size, shuffle)
            super().__init__(self.dataset, batch_sampler=sampler, num_workers=num_workers)
