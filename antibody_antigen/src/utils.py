from __future__ import print_function, division

import os
from datetime import datetime

import numpy as np
import torch
import torch.utils.data
from torch.optim import Optimizer

path = os.getcwd()


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, lchain, hchain, antigen, label):
        self.lchain = lchain
        self.hchain = hchain
        self.antigen = antigen
        self.label = label

    def __len__(self):
        return len(self.lchain)

    def __getitem__(self, i):
        return self.lchain[i], self.hchain[i], self.antigen[i], self.label[i]


def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    x2 = [a[2] for a in args]
    y = [a[3] for a in args]
    return x0, x1, x2, torch.stack(y, 0)


def log(m, file=None, timestamped=True, print_also=False):
    curr_time = f"[{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}] "  # 代码根据timestamped参数决定是否在日志信息前面添加时间戳
    log_string = f"{curr_time if timestamped else ''}{m}"  # m是输入参数，代表要打印的日志信息
    if file is None:
        print(log_string)
    else:
        # file参数不为None，则将日志信息写入文件，并根据print_also参数决定是否同时打印到标准输出中
        print(log_string, file=file)
        if print_also:
            print(log_string)
        file.flush()  # 如果写入文件后，需要调用file.flush()方法将缓冲区中的数据刷新到磁盘上


def NormalizeData(train_tensor):
    min_val = torch.min(train_tensor)
    max_val = torch.max(train_tensor)
    # print('min,max:',min_val, max_val)
    normalized_train_tensor = (train_tensor - min_val) / (max_val - min_val)
    return normalized_train_tensor


# r2 score
def r_squared(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()  # 将张量转换为numpy数组
    y_pred = y_pred.cpu().detach().numpy()
    mean_value = np.mean(y_true)
    tss = np.sum((y_true - mean_value) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - rss / tss

    return r_squared


def pearsonr(y_true, y_pred):
    """
    Calculates Pearson correlation coefficient between two arrays.

    Args:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
        float: Pearson correlation coefficient.
    """
    mean_yt = np.mean(y_true)
    mean_yp = np.mean(y_pred)

    yt_dev = y_true - mean_yt
    yp_dev = y_pred - mean_yp
    numerator = np.sum(yt_dev * yp_dev)
    denominator = np.sqrt(np.sum(yt_dev ** 2)) * np.sqrt(yp_dev ** 2)
    # Handle the case where the denominator is zero (to avoid division by zero)
    if denominator == 0:
        return 0.0

    correlation = numerator / denominator

    return correlation


# # def pearsonr(y_true, y_pred):
# #     y_true = y_true.cpu().detach().numpy()
# #     y_pred = y_pred.cpu().detach().numpy()
# #     mean_yt = np.mean(y_true)
# #     mean_yp = np.mean(y_pred)

# #     yt_dev = y_true - mean_yt
# #     yp_dev = y_pred - mean_yp

# #     numerator = np.sum(yt_dev * yp_dev)
# #     denominator = np.sqrt(np.sum(yt_dev ** 2)) * np.sqrt(np.sum(yp_dev ** 2))

# #     correlation = numerator / denominator

# #     return correlation


class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):  # rho调整
        defaults = dict(rho=rho, **kwargs)
        self.base_optimizer = base_optimizer(params, **kwargs)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            rho = group['rho']
            grad_norm = torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SAM does not support sparse gradients')
                p_data = p.data
                grad_norm = torch.norm(grad)
                if grad_norm != 0:
                    grad /= grad_norm
                    perturbation = torch.zeros_like(p_data).normal_(0, rho * grad_norm)
                    p_data.add_(perturbation)
                    self.base_optimizer.step(closure)
                    p_data.sub_(perturbation)
        return loss
