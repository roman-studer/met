import uuid
from dataclasses import dataclass
import yaml
import os
import re
from pathlib import Path
from dataclasses_json import DataClassJsonMixin
from abc import ABCMeta


@dataclass
class Config(DataClassJsonMixin, metaclass=ABCMeta):
    """
    Class holding configuration from config.yaml file.
    Config is also the parent class of all other configuration classes.
    It implements methods to load an arbitrary yaml file.
    """
    config_dir: str
    name: str
    random_state: int

    def __init__(self):
        super(DataClassJsonMixin, self).__init__()
        self.base = 'met'
        self.name = 'main'
        self.load_yaml(self.name)

    def load_yaml(self, name: str):
        built_path = Path(re.sub(rf"{self.base}.*", f'{self.base}/configuration', str(os.getcwd())) + '/' + name + '.yaml')
        config = yaml.full_load(built_path.read_text(encoding="UTF-8"))
        self.__dict__.update(config)


@dataclass()
class Paths(Config):
    """
    Class holding configuration from paths.yaml file
    """
    working_dir: str
    data_dir: str
    data_dir: str
    config_dir: str

    def __init__(self):
        super().__init__()
        self.name = 'paths'
        Config.load_yaml(self, self.name)

    def __post_init__(self):
        self.validate_path_len()

    def validate_path_len(self):
        for path in self.__dict__:
            assert len(path) < 260, f'Length of path {path} is over 260 characters. This is not allowed on Windows by ' \
                                    'default and can be changed in the registry.'


@dataclass()
class Image(Config):
    """
    Class holding configuration from image.yaml file
    """
    width: int
    height: int
    channels: int

    def __init__(self):
        super().__init__()
        self.name = 'image'
        Config.load_yaml(self, self.name)

    def get_configured_image_shape(self) -> tuple:
        """
        Returns the configured image shape as a tuple (w,h,c)
        :return: (w,h,c) tuple
        """
        return self.width, self.height, self.channels


class WandBConfig(Config):
    """
    Class holding configuration from wandb.yaml file
    """
    project_name: str
    entity: str
    api_key: str
    run_name: str

    def __init__(self):
        super().__init__()
        self.name = 'wandb'
        Config.load_yaml(self, self.name)

        if self.run_name is None:
            self.run_name = 'test' + uuid.uuid4().hex


class Hyperparameters(Config):
    """
    Class holding configuration from hyperparameters.yaml file
    """
    def __init__(self):
        super().__init__()
        self.name = 'hyperparameters'
        Config.load_yaml(self, self.name)
