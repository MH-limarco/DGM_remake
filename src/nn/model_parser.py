import contextlib
from collections.abc import Iterable

import os
import re
import yaml
import copy

from os.path import isabs

import torch
import torch.nn as nn
from torch_geometric import nn as pyGnn

from src import get_project_root
from src import nn as DGMnn

root = get_project_root()
cfg_root = r"src/cfg/"

os_re = re.compile(r'(?<!^)[/,\,//,\\]')
env_format = os_re.split(__file__)

def _format_check_(format):
    return 'norm' if ":" in format else 'wsl' if "/mnt" == format else 'not_abs'

def _alignment_path_(path_format):
    root_f = _format_check_(env_format[0])
    path_f = _format_check_(path_format[0])
    assert root_f != 'not_abs'

    if path_f != 'not_abs':
        if root_f == 'norm' and path_f == 'wsl':
            PATH = os.path.join(f"{path_format[1][0].upper()}:{os.sep}", *path_format[2:])
        elif root_f == 'wsl' and path_f == 'norm':
            PATH = os.path.join(env_format, path_format[0][0].lower(), *path_format[1:])
        else:
            PATH = os.path.join(*path_format)
    else:
        PATH = os.path.join(root, cfg_root, *path_format)
    return PATH

def read_yaml(PATH):
    path_format = os_re.split(PATH)
    PATH = _alignment_path_(path_format)

    with open(PATH) as f:
        _yaml_dict = yaml.safe_load(f)
        if _yaml_dict.get('scales') is None:
            #call
            _yaml_dict['scales'] = {'n': [1.00, 1024]}

    _model_dict = {"model": []}
    _model_dict['nc'] = _yaml_dict.pop('nc')
    _model_dict['activation'] = _yaml_dict.pop('activation')
    _model_dict['scales'] = _yaml_dict.pop('scales')
    for blocks in _yaml_dict.values():
        _model_dict['model'] += blocks

    return _model_dict


def parser(PATH, in_c, nc=None):
    arg_i = copy.deepcopy([in_c])
    model_dict = read_yaml(PATH)
    if nc is not None:
        model_dict['nc'] = nc

    _model_, save_idx, from_idx, input_edges, output_edges = [], [], [], [], []
    for idx, (f, m, args) in enumerate(model_dict['model']):
        if 'nc' in args:
            args[args.index('nc')] = model_dict['nc']

        input_edge, output_edge = True, False

        if m in ['dDGM']:
            output_edge = True

        elif m in ['JumpingKnowledge']:
            input_edge = False
            arg_i += [sum([arg_i[i] for i in f])]

        elif m in ["APPNP", "GatedGraphConv"]:

            if m == "APPNP":
                args.insert(0, arg_i[f])
                arg_i += [arg_i[f]]
            if m == "GatedGraphConv":
                args.insert(0, arg_i[f])
                arg_i += [args[1] if not isinstance(args[1], Iterable) else args[1][-1]]



        else:
            args.insert(0, arg_i[f])
            arg_i += [args[1] if not isinstance(args[1], Iterable) else args[1][-1]]

        input_edges.append(input_edge)
        output_edges.append(output_edge)


        m = getattr(DGMnn, m)
        _model_.append(m(*args))

        if f != -1:
            save_idx += f if isinstance(f, Iterable) else [f]
        from_idx.append(f if isinstance(f, Iterable) else [f])
    return seq(_model_, save_idx, from_idx, input_edges, output_edges)


def seq(model, save_idx, from_idx, input_edges, output_edges):
    x, A = "x", "edge_index"
    x_l, A_l = f"{x}{0}", f"{A}{0}"

    _seq_ = []
    for idx, (m, f, i_edge, o_edge) in enumerate(zip(model, from_idx, input_edges, output_edges)):
        if f == [-1]:
            _in_ = [x_l] if not i_edge else [x_l, A_l]
        elif len(f) < 2:
            _in_ = [f'{x}{f[0]}'] if not i_edge else [f"{x}{f[0]}", f"{A}{0}"]
        else:
            #_in_ = [f'{k}{i}' for k in ([x, A] if i_edge else [x]) for i in f]
            _in_ = [f'{x}{i}' for i in f]

        if idx in save_idx:
            x_l, A_l = f"{x}{idx}", f"{A}{0}"

        _out_ = [x_l] if not o_edge else [x_l, A_l]
        s = f"{','.join(_in_)} -> {','.join(_out_)}"
        print(s)
        _seq_.append((m, s))

    return pyGnn.Sequential(','.join([f"{x}{0}", f"{A}{0}"]), _seq_)



if __name__ == "__main__":
    #path = r"N:\python-code\DGM_v2\src\cfg\test.yaml"
    #path = r"/mnt/n/python-code\DGM_v2\src\cfg\test.yaml"
    path = r"test2.yaml"

    in_dim = 10
    x = torch.rand((64, in_dim)).cpu()
    edge_index = torch.randint(64, size=(2, 20)).cpu()

    _test_ = parser(path, in_dim)
    _test_ = pyGnn.summary(_test_, x, edge_index, max_depth=1)
    print(_test_)

    #path = r"src/cfg/test.yaml"
    #read_yaml(path)