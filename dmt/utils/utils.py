import json
import logging
import os, sys

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

    @property
    def model(self):
        try:
            return self.__dict__['model']
        except AttributeError:
            self.__dict__['model'] = None
            return None


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

if sys.platform.lower() == "win32":
    os.system('color')

# Group of Different functions for different styles
class style():
    BLACK = lambda x: '\033[30m' + str(x) + '\033[0m'
    RED = lambda x: '\033[31m' + str(x) + '\033[0m'
    GREEN = lambda x: '\033[32m' + str(x) + '\033[0m'
    YELLOW = lambda x: '\033[33m' + str(x) + '\033[0m'
    BLUE = lambda x: '\033[34m' + str(x) + '\033[0m'
    MAGENTA = lambda x: '\033[35m' + str(x) + '\033[0m'
    CYAN = lambda x: '\033[36m' + str(x) + '\033[0m'
    WHITE = lambda x: '\033[37m' + str(x) + '\033[0m'
    UNDERLINE = lambda x: '\033[4m' + str(x) + '\033[0m'
    RESET = lambda x: '\033[0m' + str(x)

# print(style.GREEN("No node features is given, use dummy featuers") + style.RESET(""))