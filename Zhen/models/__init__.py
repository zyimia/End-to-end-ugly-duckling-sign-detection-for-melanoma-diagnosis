import torch
from collections import OrderedDict
from .resnet_siim import ResNet_SIIM
from .efficient_siim import Efficient_SIIM
from .efficient_siimd import Efficient_SIIMD

key2model = {
    'efficient-siim': Efficient_SIIM,
    'resnet-siim': ResNet_SIIM,
    'efficient-siimd': Efficient_SIIMD
}


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]  # remove `module`
        new_state_dict[name] = v
    return new_state_dict


def get_model(model_name, in_channel, n_classes, *param):

    if model_name not in key2model.keys():
        print(key2model)
        raise ModelErr('model does not exists in the given list')

    model = key2model[model_name](in_channel, n_classes, *param)

    return model


class ModelErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
