from pathlib import Path

import pickle
import torch
import yaml

label_name_to_code = {
    'rib_fracture': '33737001',
    'pulmonary_embolism': '59282003',
    'lung_nodules': '427359005',
    'spine_fracture': '50448004',
    'liver_lesions': '300332007',
    'emphysema': '87433001',
    'pneumothorax': '36118008',
    'aortic_dilatation': '26660001',
}

label_code_to_name = {v: k for k, v in label_name_to_code.items()}


def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt', encoding='utf8') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt', encoding='utf8') as handle:
        yaml.dump(content, handle)


def dump_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, -1)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def prepare_device(gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    n_gpu_use = len(gpu_use)
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, "
               "but only {n_gpu} are available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device(f'cuda:{gpu_use[0]}' if n_gpu_use > 0 else 'cpu')
    return device, gpu_use
