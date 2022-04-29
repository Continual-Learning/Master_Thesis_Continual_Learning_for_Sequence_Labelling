import torch
from numpy import inf
from tqdm import tqdm

from logger import TensorboardWriter
from model.loss import BCELoss
from model.metric import ClassificationMetrics


class MultiTaskTrainer():
    """Trainer class to train on a set of tasks simultaneously.

    Training is performed on all tasks of interest (single training data loader).
    Validation is performed individually for each task.
    """
    def __init__(self, model, criterion, optimizer, config, device, data_loader,
                 valid_data_loaders=None,
                 lr_scheduler=None,
                 save_epoch_checkpoint=True,
                 ewc_loss=None,
                 ewc_lambda=None):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        # Used for model saving to avoid dependency on myltiple GPUs
        self.model_is_data_parallel = isinstance(model, torch.nn.DataParallel)
        self.loss = criterion
        # Currently does not work with general loss due to different DataLoaders used for train and eval
        self.eval_loss = BCELoss(device=device)
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.device = device

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        self.ewc_loss = ewc_loss
        self.ewc_lambda = ewc_lambda
        if self.ewc_loss:
            self.logger.info("Computing EWC penalty coefficients...")
            self.ewc_loss.set_model(self.model)

        self.accumulate_gradients = cfg_trainer.get("accumulate_gradients", 1)
        self.data_loader = data_loader

        self.len_epoch = len(data_loader)
        self.valid_data_loaders = valid_data_loaders
        self.do_validation = self.valid_data_loaders is not None
        self.lr_scheduler = lr_scheduler

        self.save_epoch_checkpoint = save_epoch_checkpoint

        self.train_metrics = ClassificationMetrics("", writer=self.writer)
        self.valid_metrics = [ClassificationMetrics(f"/{data_loader.label_code}", writer=self.writer)
                              for data_loader in self.valid_data_loaders]

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch, embedding_tag="latest")

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info(f'    {str(key):15s}: {value}')

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # Average performance over all tasks
                    avg_mnt_metric = 0.0
                    for metric in self.valid_metrics:
                        task_mnt_metric = metric.metric_name(self.mnt_metric)
                        avg_mnt_metric += log[task_mnt_metric]

                    avg_mnt_metric /= len(self.valid_metrics)
                    self.logger.info(f"Macro {self.mnt_metric}: {avg_mnt_metric}")
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and avg_mnt_metric <= self.mnt_best) or \
                        (self.mnt_mode == 'max' and avg_mnt_metric >= self.mnt_best)
                except KeyError:
                    self.logger.warning(f"Warning: Metric '{task_mnt_metric}' is not found. "
                                        "Model performance monitoring is disabled.")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = avg_mnt_metric
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(f"Validation performance didn\'t improve for {self.early_stop} epochs. "
                                     "Training stops.")
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, task="", save_best=best, save_epoch=self.save_epoch_checkpoint)

        # Dump metrics for this task to csv
        self.train_metrics.dump_metrics(self.checkpoint_dir / "train_metrics.csv")
        for i, valid in enumerate(self.valid_data_loaders):
            self.valid_metrics[i].dump_metrics(self.checkpoint_dir / f"eval_metrics_{valid.label_code}.csv")

    def _train_epoch(self, epoch, embedding_tag=None):
        self.model.train()
        self.train_metrics.reset()
        batch_iter = tqdm(self.data_loader, total=self.len_epoch)
        batch_idx = 0
        self.optimizer.zero_grad()

        for data, target in batch_iter:
            target["labels"] = target["labels"].to(self.device)
            target["weights"] = target["weights"].to(self.device)
            data = {k: t.to(self.device) for k, t in data.items()}
            output = self.model(**data, cls_head_id=target["tasks"])
            loss, loss_components = self.loss(output, target)
            loss = loss / self.accumulate_gradients
            loss.backward()

            if ((batch_idx + 1) % self.accumulate_gradients == 0) or (batch_idx + 1 == len(self.data_loader)):
                if self.ewc_loss:
                    ewc_loss_tensor, ewc_loss_component = self.ewc_loss(self.model)
                    loss += self.ewc_lambda * ewc_loss_tensor
                    loss_components.update(ewc_loss_component)
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update(loss_components, output, target["labels"])

            batch_iter.set_description(f"Epoch: {epoch:02} - Loss: {loss.item():2.4f}", refresh=True)
            if batch_idx == self.len_epoch:
                break
            batch_idx += 1

        log = self.train_metrics.metrics()
        self.train_metrics.save_metrics(step=f"epoch_{epoch}")

        if self.do_validation:
            all_embeddings, all_labels = [], []
            for metric, valid_data_loader in zip(self.valid_metrics, self.valid_data_loaders):
                val_log, embeddings, labels = self._valid_epoch(epoch, valid_data_loader, metric)
                all_embeddings.append(embeddings)
                all_labels.extend(labels)
                log.update(val_log)

            if not self.ewc_loss:
                all_embeddings = torch.cat(all_embeddings)
                self.writer.set_step(epoch)  # So we only save one set of embeddings at a time
                self.writer.add_embedding(all_embeddings, all_labels, label_img=None, tag=embedding_tag)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # add histogram of model parameters to the tensorboard
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, bins="doane")

        return log

    def _valid_epoch(self, epoch: int, valid_data_loader, metrics):
        self.model.eval()
        metrics.reset()
        cls_head_id = valid_data_loader.label_code

        # Save end of epoch metrics
        global_step = epoch * len(self.data_loader)
        self.writer.set_step(global_step, 'val')

        embedding_labels = []
        embeddings = []
        with torch.no_grad():
            self.logger.info(f"Epoch evaluation on task {valid_data_loader.label_code}...")
            for _, (data, target) in enumerate(valid_data_loader):
                target["labels"] = target["labels"].to(self.device)
                target["weights"] = torch.tensor([1.0] * len(target["labels"])).to(self.device)

                for key, tensor in data.items():
                    data[key] = tensor.to(self.device)

                if not self.ewc_loss:
                    output, hidden, _ = self.model(**data, cls_head_id=cls_head_id, output_attn_and_hidden=True)
                    # Only keep output of last layer, embedding of CLS token, all dimensions
                    hidden = torch.squeeze(hidden[-1][:, 0, :]).to("cpu")
                    embedding_labels.extend([f"{cls_head_id}_{label}" for label in target['labels']])
                    embeddings.append(hidden)
                else:
                    output = self.model(**data, cls_head_id=cls_head_id, output_attn_and_hidden=False)

                _, loss_components = self.eval_loss(output, target)

                metrics.update(loss_components, output, target["labels"], log=False)

        if not self.ewc_loss:
            embeddings = torch.cat(embeddings)

        metrics.log_to_writer()
        metrics.save_metrics(step=f"epoch_{epoch}")
        return metrics.metrics(), embeddings, embedding_labels

    def _save_checkpoint(self, epoch, task=None, save_best=False, save_epoch=True):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'task': task,
            'state_dict': self.model.module.state_dict() if self.model_is_data_parallel else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_epoch:
            filename = str(self.checkpoint_dir / f'checkpoint-epoch{epoch}-{task}.pth')
            torch.save(state, filename)
            self.logger.info(f'Saving checkpoint: {filename} ...')
        if save_best:
            best_path = str(self.checkpoint_dir / f'model_best-{task}.pth')
            torch.save(state, best_path)
            self.logger.info(f'Saving current best: model_best-{task}.pth ...')

    def _resume_checkpoint(self, resume_path):
        train_opts = self.config["trainer"]
        cl_experiment = train_opts["cl_incremental_training"] if "cl_incremental_training" in train_opts else False

        resume_path = str(resume_path)
        self.logger.info(f'Loading checkpoint: {resume_path} ...')
        checkpoint = torch.load(resume_path, map_location=self.device)
        if not cl_experiment:
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']
            # load optimizer state from checkpoint only when optimizer type is not changed.
            if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger.warning("Warning: Optimizer type config file is different from that of checkpoint. "
                                    "Optimizer parameters not being resumed.")
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        if self.model_is_data_parallel:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
