import logging
import torch

class GpuRankFilter(logging.Filter):
    def filter(self, record):
        if torch.cuda.is_available():
            record.gpu_rank = torch.cuda.current_device()
        else:
            record.gpu_rank = "n/a"  # Handle case if no GPU is available
        return True

class ColouredFormatter(logging.Formatter):
    """Source: 
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    #format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = '%(levelname)-8s [gpu %(gpu_rank)s] [%(filename)s:%(lineno)d] %(message)s'

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(name):

    logger = logging.getLogger(name)

    #print("get_logger() invoked with name:", name, "has handlers:", logger.handlers)
  
    logger.setLevel(logging.INFO)
    # https://stackoverflow.com/questions/533048/how-to-log-source-file-name-and-line-number-in-python
    formatter = ColouredFormatter()
    ch = logging.StreamHandler()
    #ch.setLevel(logging.INFO)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    logger.addFilter(GpuRankFilter())
    # https://stackoverflow.com/questions/6729268/log-messages-appearing-twice-with-python-logging
    logger.propagate = False
    
    logging.basicConfig(datefmt='%Y-%m-%d:%H:%M:%S')
    
    return logger