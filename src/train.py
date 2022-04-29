import argparse
from copy import deepcopy

import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
import trainer.optimizer as module_optim
from parse_config import ConfigParser
from trainer import MultiTaskTrainer
from utils.util import prepare_device, read_yaml

DEFAULT_SEED = 123


def fix_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def main(config: ConfigParser) -> None:
    logger = config.get_logger('train')
    experiment_config = config.config["experiment_config"]

    # build model architecture, then print to console
    config["model"]["args"].update({"classifier_ids": experiment_config["label_code_opts"]})
    model = config.init_obj('model', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['gpu_list'])
    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    logger.info("Creating loss function objects")
    loss = config.init_obj('loss', module_loss, device=device)

    ewc_loss, ewc_lambda = None, None
    if config["ewc"] is not None:
        logger.info("Creating EWC loss object")
        if config.resume is None:
            logger.warning("Using EWC in a non-continual learning setting")
        ewc_data_loader = module_data.MultiTaskDataloader(config["ewc"]["data_dir"],
                                                          batch_size=32,
                                                          shuffle=False,
                                                          label_codes=experiment_config["label_code_opts"],
                                                          tokenizer_path=config["model"]["args"]["model_path"])
        ewc_loss = module_loss.EWCLoss(ewc_data_loader, task_loss=module_loss.BCELoss(device=device), device=device)
        ewc_lambda = config["ewc"]["lambda"]

    optimizer = config.init_obj('optimizer', module_optim, model)
    lr_scheduler = config.init_obj('lr_scheduler', module_optim, optimizer)

    if "eval_dir" in config["data_loader"] and "train_dir" in config["data_loader"]:
        data_loader = config.init_obj('data_loader', module_data,
                                        data_dir=config["data_loader"]["train_dir"],
                                        label_codes=experiment_config["label_code_opts"])
        tokenizer_path = config["data_loader"]["args"]["tokenizer_path"]
        valid_data_loaders = [
            module_data.TextDataloader(data_dir=config["data_loader"]["eval_dir"],
                                        batch_size=64,
                                        label_code=label,
                                        tokenizer_path=tokenizer_path)
            for label in experiment_config["label_code_opts"]
        ]
    else:
        raise KeyError("Need to explicitly pass validation dir to multi task dataloader")

    trainer = MultiTaskTrainer(model, loss, optimizer,
                                config=config,
                                device=device,
                                data_loader=data_loader,
                                valid_data_loaders=valid_data_loaders,
                                lr_scheduler=lr_scheduler,
                                save_epoch_checkpoint=experiment_config["save_epoch_checkpoints"],
                                ewc_loss=ewc_loss,
                                ewc_lambda=ewc_lambda)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = vars(parser.parse_args())

    config_yaml = read_yaml(args["config"])

    seeds = config_yaml.get("random_seeds", [DEFAULT_SEED])

    for random_seed in seeds:
        fix_random_seeds(random_seed)
        run_config = ConfigParser(deepcopy(config_yaml), args["resume"])
        main(run_config)
        # Need to manually update experiment ids if they were initally set
        experiment_id = config_yaml["experiment_config"]["experiment_id"]
        if experiment_id is not None:
            config_yaml["experiment_config"]["experiment_id"] = str(int(experiment_id) + 1)
