# -*-coding:utf-8 -*-

# File       : utils.py
# Author     : hingmauc
# Time       : 2024/7/23 14:32
# Description：

import torch
import cv2
import numpy as np


def get_if(Yf_tensor, vi_3_tensor):
    vi_ycbcr = rgb_to_ycbcr(vi_3_tensor)
    cb = vi_ycbcr[:, 1:2, :, :]
    cr = vi_ycbcr[:, 2:3, :, :]
    If_ycbcr = torch.cat([Yf_tensor, cb, cr], dim=1)
    If_tensor = ycbcr_to_rgb(If_ycbcr)
    return If_tensor


def rgb_to_ycbcr(img_rgb):
    R = img_rgb[:, 0:1, :, :]
    G = img_rgb[:, 1:2, :, :]
    B = img_rgb[:, 2:3, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    return torch.cat([Y, Cb, Cr], dim=1)


def ycbcr_to_rgb(img_ycbcr):
    Y = img_ycbcr[:, 0:1, :, :]
    Cb = img_ycbcr[:, 1:2, :, :]
    Cr = img_ycbcr[:, 2:3, :, :]
    R = Y + 1.402 * (Cr - 128 / 255.0)
    G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    B = Y + 1.772 * (Cb - 128 / 255.0)
    return torch.cat([R, G, B], dim=1)


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img
