import yaml
import os
from easydict import EasyDict


def load_config(net_name):
    filepath = os.path.join('./data/cfgs/', net_name + '.yml')
    with open(filepath, mode='r') as f:
        cfg = yaml.load(f.read())
        cfg = EasyDict(cfg)
    return cfg
