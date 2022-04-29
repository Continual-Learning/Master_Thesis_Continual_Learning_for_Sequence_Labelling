import logging
import logging.config
from pathlib import Path

from utils import read_yaml


def setup_logging(save_dir, log_config='src/logger/logger_config.yaml', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_yaml(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print(f"Warning: logging configuration file is not found in {log_config}.")
        logging.basicConfig(level=default_level)


def get_logger(self, name, verbosity=2):
    msg_verbosity = f'verbosity option {verbosity} is invalid. Valid options are {self.log_levels.keys()}.'
    assert verbosity in self.log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(self.log_levels[verbosity])
    return logger
