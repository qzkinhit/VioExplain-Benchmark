# coding=utf-8
"""全局随机种子控制 —— 复现的命门。

原型代码（main.py / compare.py / yset.py）全程没有 random.seed，导致每次结果不同。
本模块统一设定 random 与 numpy 的种子，保证「同代码 + 同 seed = 同结果」。
"""
import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """同时固定 Python random 与 numpy 的随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def derive_seed(base: int, run_idx: int) -> int:
    """多 run 实验：由 base 派生每个 run 的独立种子，整体仍可复现。"""
    return base + run_idx
