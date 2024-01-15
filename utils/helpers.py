"""shout-out to https://github.com/lucidrains/x-transformers/tree/main/x_transformers"""
import json
from pathlib import Path
import warnings
import torch
from functools import partial
from inspect import isfunction
import PIL.Image as pil_image
from torchvision.transforms import PILToTensor

# helpers
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def always(val):
    def inner(*args, **kwargs):
        return val
    return inner


def not_equals(val):
    def inner(x):
        return x != val
    return inner


def equals(val):
    def inner(x):
        return x == val
    return inner


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


# keyword argument helpers

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def load_image_from_disk(path: Path):
    return pil_image.open(path).convert('RGB')

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


pil_to_tensor = PILToTensor()

def convert_pil_to_tensor(image) -> torch.Tensor:
    with warnings.catch_warnings():
        # to filter PyTorch UserWarning as described here: https://github.com/pytorch/vision/issues/2194
        warnings.simplefilter("ignore")
        return pil_to_tensor(image)