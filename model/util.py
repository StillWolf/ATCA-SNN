"""
-*- coding: utf-8 -*-

@Time    : 2021/5/4 11:05

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : util.py
"""

import torch
import numpy as np
import random
def setup_seed(seed):
    print('set random seed as '+str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True