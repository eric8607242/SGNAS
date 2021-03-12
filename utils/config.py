"""Config utilities for yml file.
Modified from
https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/config.py
"""
import os
import argparse
import yaml


class LoaderMeta(type):
    """Constructor for supporting `!include` and `!path`."""

    def __new__(mcs, __name__, __bases__, __dict__):
        cls = super().__new__(mcs, __name__, __bases__, __dict__)
        cls.add_constructor("!include", cls.construct_include)
        return cls


class Loader(yaml.Loader, metaclass=LoaderMeta):
    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)

    def construct_include(self, node):
        filename = os.path.abspath(
            os.path.join(self._root, self.construct_scalar(node)))
        file_type = os.path.splitext(filename)[1].lstrip('.')
        with open(filename, 'r') as f:
            if file_type in ('yaml', 'yml'):
                return yaml.load(f, Loader)
            else:
                return ''.join(f.readlines())


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, list):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """
        Convert object to yaml dict
        """
        yaml_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = value
                else:
                    yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        ret_str = []
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                ret_str.append("{}:".format(key))
                child_ret_str = value.__repr__().split("\n")
                for item in child_ret_str:
                    ret_str.append("     " + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append("{}:".format(key))
                    for item in value:
                        child_ret_str = item.__repr__().split(".\n")
                        for item in child_ret_str:
                            ret_str.append("     " + item)
                else:
                    ret_str.append("{}: {}".format(key, value))
            else:
                ret_str.append("{}: {}".format(key, value))
        return "\n".join(ret_str)


class Config(AttrDict):
    def __init__(self, file_path):
        assert os.path.exists(
            file_path), "File {} not exits.".format(file_path)

        with open(file_path, "r") as f:
            cfg_dict = yaml.load(f, Loader)
        super(Config, self).__init__(cfg_dict)


def get_config(cfg_path):
    CONFIG = Config(cfg_path)
    return CONFIG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, help="path to the config file")
    args = parser.parse_args()

    CONFIG = get_config(args.cfg_path)
    print(CONFIG)
