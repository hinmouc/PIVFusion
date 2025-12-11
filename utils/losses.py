# -*-coding:utf-8 -*-

# File       : losses.py
# Author     : hingmauc
# Time       : 2024/10/23 14:29
# Description：

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContrastLoss(nn.Module):
    def __init__(self, filter_type='sobel'):
        super(ContrastLoss, self).__init__()
        if filter_type == 'sobel':
            self.kernel_x = torch.tensor([[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]], dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0)
            self.kernel_y = torch.tensor([[-1, -2, -1],
                                          [0, 0, 0],
                                          [1, 2, 1]], dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0)

    def high_pass_filter(self, image, kernel):
        if image.is_cuda:
            kernel = kernel.to(image.device)
        else:
            kernel = kernel

        channels = image.shape[1]
        kernel = kernel.repeat(channels, 1, 1, 1)

        filtered = F.conv2d(image, kernel, padding=1, groups=channels)
        return filtered

    def forward(self, original_image, enhanced_image):
        original_filtered_x = self.high_pass_filter(original_image, self.kernel_x)
        original_filtered_y = self.high_pass_filter(original_image, self.kernel_y)

        enhanced_filtered_x = self.high_pass_filter(enhanced_image, self.kernel_x)
        enhanced_filtered_y = self.high_pass_filter(enhanced_image, self.kernel_y)

        original_contrast = original_filtered_x.std(dim=[2, 3]) + original_filtered_y.std(dim=[2, 3])
        enhanced_contrast = enhanced_filtered_x.std(dim=[2, 3]) + enhanced_filtered_y.std(dim=[2, 3])

        loss = F.l1_loss(original_contrast, enhanced_contrast)
        return loss

class GlobalContrastLoss(nn.Module):
    def __init__(self):
        super(GlobalContrastLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)

    def forward(self, source, target):
        source_std, source_mean = torch.std_mean(source)
        target_std, target_mean = torch.std_mean(target)

        source_dynamic_range = torch.max(source) - torch.min(source)
        target_dynamic_range = torch.max(target) - torch.min(target)

        contrast_loss = F.l1_loss(source_std, target_std) + F.l1_loss(source_mean, target_mean)
        dynamic_range_loss = F.l1_loss(source_dynamic_range, target_dynamic_range)

        total_loss = contrast_loss + dynamic_range_loss
        return total_loss

class HistogramMatchLoss(nn.Module):
    def __init__(self, bins=256, smooth=False, smooth_sigma=1.0):
        super().__init__()
        self.bins = bins
        self.smooth = smooth
        self.smooth_sigma = smooth_sigma

    def gaussian_smooth(self, hist):
        kernel_size = int(self.smooth_sigma * 3) * 2 + 1
        kernel = torch.normal(mean=0, std=self.smooth_sigma, size=(kernel_size,))
        kernel /= kernel.sum()
        return torch.nn.functional.conv1d(hist.view(1, 1, -1), kernel.view(1, 1, -1), padding=kernel_size // 2).view(-1)

    def forward(self, source, target):
        source_hist = torch.histc(source, bins=self.bins, min=0, max=1)
        target_hist = torch.histc(target, bins=self.bins, min=0, max=1)

        if self.smooth:
            source_hist = self.gaussian_smooth(source_hist)
            target_hist = self.gaussian_smooth(target_hist)

        epsilon = 1e-8
        source_hist_norm = source_hist / source_hist.sum()+ epsilon
        target_hist_norm = target_hist / target_hist.sum()+ epsilon

        loss = F.mse_loss(source_hist_norm, target_hist_norm)
        return loss


class CombinedContrastLoss(nn.Module):
    def __init__(self, hist_weight=0.4, global_weight=0.2, local_weight=0.4):
        super().__init__()
        self.hist_weight = hist_weight
        self.global_weight = global_weight
        self.local_weight = local_weight

        self.hist_loss = HistogramMatchLoss()
        self.global_loss = GlobalContrastLoss()
        self.local_loss = ContrastLoss()

    def forward(self, input, target):
        hist_loss = self.hist_loss(input, target)
        global_loss = self.global_loss(input, target)
        local_loss = self.local_loss(input, target)
        combined_loss = (
                self.hist_weight * hist_loss +
                self.global_weight * global_loss +
                self.local_weight * local_loss
                )
        return combined_loss


class Losses1(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = kornia.losses.SSIMLoss(11, reduction='mean')

    def gradient(self, x):
        return kornia.filters.SpatialGradient()(x)

    def forward(self, vi, ir, vi_hat, ir_hat):
        total_loss = 1 * self.mse(vi,vi_hat) + 5 * self.ssim(vi,vi_hat)+\
                     1 * self.mse(ir,ir_hat) + 5 * self.ssim(ir,ir_hat)

        print("Total Loss:", total_loss.item())
        return total_loss


class Losses2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lumin_loss = CombinedContrastLoss().to(device)

    def gradient(self, input):
        kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view(1, 1, 2, 2).to(input.device)
        kernel_y = kernel_x.transpose(2, 3)
        gradient_x = torch.abs(F.conv2d(input, kernel_x, stride=1, padding=1))
        gradient_y = torch.abs(F.conv2d(input, kernel_y, stride=1, padding=1))
        gradient_x = (gradient_x - torch.min(gradient_x)) / (torch.max(gradient_x) - torch.min(gradient_x) + 0.0001)
        gradient_y = (gradient_y - torch.min(gradient_y)) / (torch.max(gradient_y) - torch.min(gradient_y) + 0.0001)
        return gradient_x + gradient_y

    def angle(self, a, b):
        vector = a * b
        up = torch.sum(vector)
        down = torch.sqrt(torch.sum(a ** 2)) * torch.sqrt(torch.sum(b ** 2))
        down += 1e-8
        theta = torch.acos(up / down)
        return theta

    def forward(self, Y_f, vi, ir, vi_y_en):
        total_loss = 20 * F.l1_loss(self.gradient(Y_f) , torch.max(self.gradient(ir), self.gradient(vi)))+\
                     0.5 * torch.mean(self.angle(Y_f,vi))+\
                     1.1 * self.lumin_loss(Y_f, vi_y_en)

        print("Total Loss:", total_loss.item())
        return total_loss
