# -*-coding:utf-8 -*-

# File       : wave_fusion.py
# Author     : hingmauc
# Time       : 2024/3/1 11:23
# Description:

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

Pad_Mode = ['constant', 'reflect', 'replicate', 'circular']
class DWT_2D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'db1',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False):
        super(DWT_2D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_ll = self.filt_low[:, None] * self.filt_low[None,:]
        self.filter_lh = self.filt_low[:, None] * self.filt_high[None,:]
        self.filter_hl = self.filt_high[:, None] * self.filt_low[None,:]
        self.filter_hh = self.filt_high[:, None] * self.filt_high[None,:]
        self.filter_ll = self.filter_ll[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_lh = self.filter_lh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hl = self.filter_hl[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hh = self.filter_hh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        if torch.cuda.is_available():
            self.filter_ll = self.filter_ll.cuda()
            self.filter_lh = self.filter_lh.cuda()
            self.filter_hl = self.filter_hl.cuda()
            self.filter_hh = self.filter_hh.cuda()
        if self.trainable:
            self.filter_ll = nn.Parameter(self.filter_ll)
            self.filter_lh = nn.Parameter(self.filter_lh)
            self.filter_hl = nn.Parameter(self.filter_hl)
            self.filter_hh = nn.Parameter(self.filter_hh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2,
                              self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 4
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv2d(input, self.filter_ll, stride = self.stride, groups = self.groups),\
               F.conv2d(input, self.filter_lh, stride = self.stride, groups = self.groups),\
               F.conv2d(input, self.filter_hl, stride = self.stride, groups = self.groups),\
               F.conv2d(input, self.filter_hh, stride = self.stride, groups = self.groups)


class IDWT_2D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'db1',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False):
        super(IDWT_2D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.dec_lo)
        band_high = torch.tensor(wavelet.dec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_ll = self.filt_low[:, None] * self.filt_low[None,:]
        self.filter_lh = self.filt_low[:, None] * self.filt_high[None,:]
        self.filter_hl = self.filt_high[:, None] * self.filt_low[None,:]
        self.filter_hh = self.filt_high[:, None] * self.filt_high[None,:]
        self.filter_ll = self.filter_ll[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_lh = self.filter_lh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hl = self.filter_hl[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hh = self.filter_hh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        up_filter = torch.zeros((self.stride))
        up_filter[0] = 1
        up_filter = up_filter[:, None] * up_filter[None,:]
        up_filter = up_filter[None, None, :, :].repeat(self.out_channels, 1, 1, 1)
        self.register_buffer('up_filter', up_filter)
        if torch.cuda.is_available():
            self.filter_ll = self.filter_ll.cuda()
            self.filter_lh = self.filter_lh.cuda()
            self.filter_hl = self.filter_hl.cuda()
            self.filter_hh = self.filter_hh.cuda()
        if self.trainable:
            self.filter_ll = nn.Parameter(self.filter_ll)
            self.filter_lh = nn.Parameter(self.filter_lh)
            self.filter_hl = nn.Parameter(self.filter_hl)
            self.filter_hh = nn.Parameter(self.filter_hh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 0, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 0, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 + 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 + 1]

    def forward(self, LL, LH, HL, HH):
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        assert LL.size()[0] == LH.size()[0] == HL.size()[0] == HH.size()[0]
        assert LL.size()[1] == LH.size()[1] == HL.size()[1] == HH.size()[1] == self.in_channels
        LL = F.conv_transpose2d(LL, self.up_filter, stride = self.stride, groups = self.in_channels)
        LH = F.conv_transpose2d(LH, self.up_filter, stride = self.stride, groups = self.in_channels)
        HL = F.conv_transpose2d(HL, self.up_filter, stride = self.stride, groups = self.in_channels)
        HH = F.conv_transpose2d(HH, self.up_filter, stride = self.stride, groups = self.in_channels)
        LL = F.pad(LL, pad = self.pad_sizes, mode = self.pad_type)
        LH = F.pad(LH, pad = self.pad_sizes, mode = self.pad_type)
        HL = F.pad(HL, pad = self.pad_sizes, mode = self.pad_type)
        HH = F.pad(HH, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv2d(LL, self.filter_ll, stride = 1, groups = self.groups) + \
               F.conv2d(LH, self.filter_lh, stride = 1, groups = self.groups) + \
               F.conv2d(HL, self.filter_hl, stride = 1, groups = self.groups) + \
               F.conv2d(HH, self.filter_hh, stride = 1, groups = self.groups)



from model import TransformerBlock_Dual
class WaveFusionModule(nn.Module):
    def __init__(self, in_channels,
                 wave_name = 'db1',
                 num_heads = [8, 4, 2, 1],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(WaveFusionModule, self).__init__()

        self.dwt = DWT_2D(in_channels=in_channels,wavename=wave_name)
        self.idwt = IDWT_2D(in_channels=in_channels,wavename=wave_name)

        self.base_enhance = TransformerBlock_Dual(dim=in_channels, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type)
        self.detail_enhance = TransformerBlock_Dual(dim=in_channels, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type)

        self.base_fuser = TransformerBlock_Dual(dim=in_channels, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type)
        self.detail_fuser = TransformerBlock_Dual(dim=in_channels, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, feature_vi, feature_ir):
        coeffs_visible = self.dwt(feature_vi)
        coeffs_infrared = self.dwt(feature_ir)

        coeffs_fused = []

        for idx, (cv, ci) in enumerate(zip(coeffs_visible, coeffs_infrared)):
            if idx == 0:
                enhance_base_vi = self.base_enhance(cv)
                enhance_base_ir = self.base_enhance(ci)
                coeffs_fused.append(self.base_fuser(enhance_base_vi + enhance_base_ir))
            else:
                enhance_detail_vi = self.detail_enhance(cv)
                enhance_datail_ir = self.detail_enhance(ci)
                coeffs_fused.append(self.detail_fuser(enhance_detail_vi + enhance_datail_ir))

        fused_feature = self.idwt(*coeffs_fused)
        return fused_feature