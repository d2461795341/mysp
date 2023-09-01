import os
import torch
import numpy as np
import random
import os
import yaml

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


def stable_softmax(x):
    # 减去每行的最大值，以防止上溢出
    x_max, _ = torch.max(x, dim=1, keepdim=True)
    x -= x_max

    # 计算 Softmax 分子
    exp_x = torch.exp(x)

    # 计算 Softmax 分母
    sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)

    # 计算 Softmax 值
    softmax_x = exp_x / sum_exp_x

    return softmax_x