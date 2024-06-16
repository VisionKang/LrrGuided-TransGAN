import torch
import torch.nn.functional as F
from math import exp
import torch.nn as nn
from args_fusion import args
import numpy as np

# 这个特殊的msssim可以试试, 但先用普通的, swinfuse里用的
"""------------------------------------------------------------------------------------------------------------------"""
# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()
#
#
# def create_window(window_size, channel=1):  # 11, 1
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # (11, 1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # (11, 11)-->(1, 1, 11, 11)
#     # 只能在第0维扩展一个维数; 如果不增加维数，只是增加维度，要增加的原维度必须是1才可以在该维度增加维度，其他值均不可以
#     window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  # (channel, 1, 11, 11)
#     return window
#
#
# def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
#     # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
#     if val_range is None:
#         if torch.max(img1) > 128:
#             max_val = 255
#         else:
#             max_val = 1  # 1
#
#         if torch.min(img1) < -0.5:
#             min_val = -1  # -1
#         else:
#             min_val = 0
#         L = max_val - min_val  # 2
#     else:
#         L = val_range
#
#     padd = 0
#     (_, channel, height, width) = img1.size()  # channel:1
#     if window is None:
#         real_size = min(window_size, height, width)  # min(11, 224, 224)-->11
#         window = create_window(real_size, channel=channel).to(img1.device)  # (1, 1, 11, 11)
#
#     mu1 = F.conv2d(img1, window, padding=padd, groups=channel)  # [out_channels, in_channels/groups, Kheight, Kweight]
#     mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
#
#     mu1_sq = mu1.pow(2)  # E(X)**2
#     mu2_sq = mu2.pow(2)  # E(Y)**2
#     mu1_mu2 = mu1 * mu2  # E(X)*E(Y)
#
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq  # E(X**2)
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq  # E(Y**2)
#     sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2  # E(X*Y)
#
#     C1 = (0.01 * L) ** 2
#     C2 = (0.03 * L) ** 2
#
#     v1 = 2.0 * sigma12 + C2
#     v2 = sigma1_sq + sigma2_sq + C2
#     cs = torch.mean(v1 / v2)  # contrast sensitivity, 对比度相似性的平均值
#
#     ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)  # SSIM完整公式
#
#     if size_average:  # True
#         ret = ssim_map.mean()  # 一个数，SSIM的平均值
#     else:
#         ret = ssim_map.mean(1).mean(1).mean(1)  # 一个包含4个值的tensor
#
#     if full:  # True
#         return ret, cs
#     return ret
#
#
# def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
#     device = img1.device
#     weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)  # 和为1.0001，大约为1
#     levels = weights.size()[0]  # [5]-->5
#     mssim = []
#     mcs = []  # 这玩意干嘛的
#     for _ in range(levels):
#         sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
#         mssim.append(sim)  # 5个(SSIM平均值)tensor的列表
#         mcs.append(cs)  # 5个(对比度相似性的平均值)tensor的列表
#
#         img1 = F.avg_pool2d(img1, (2, 2))
#         img2 = F.avg_pool2d(img2, (2, 2))
#
#     mssim = torch.stack(mssim)  # tensor:(5,)
#     mcs = torch.stack(mcs)  # tensor:(5,)
#
#     # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
#     if normalize:
#         mssim = (mssim + 1) / 2
#         mcs = (mcs + 1) / 2
#
#     pow1 = mcs ** weights  # 5个相对应位置的mcs的weights次方, tensor:(5,)
#     pow2 = mssim ** weights  # mssim的weights次方, tensor:(5,)
#     # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
#     output = torch.prod(pow1[:-1] * pow2[-1])  # [:-1]:自始至最后一个元素前的所有元素, output是一个数字
#     return output
"""------------------------------------------------------------------------------------------------------------------"""

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1, 1, 11, 11]
    return window

# 计算 ssim 损失函数, img1:源图, img2:融合图像
def mssim(img1, img2, window_size=11, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1  # 1

        if torch.min(img1) < -0.5:
            min_val = -1  # -1
        else:
            min_val = 0
        L = max_val - min_val  # 2
    else:
        L = val_range

    padd = window_size // 2  # 为了保持图像大小不变, 还是224

    (_, channel, height, width) = img1.size()

    # 滤波器窗口
    window = create_window(window_size, channel=channel).to(img1.device)  # [channel, 1, 11, 11]-->[1, 1, 11, 11]
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    # cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret

# 方差计算
def std(img, window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)  # [1, 1, 9, 9]
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

def final_ssim(img_ir, img_vis, img_fuse):

    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    std_ir = std(img_ir)  # 方差
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)  # 返回一个用标量值0填充的tensor，其大小与input相同
    one = torch.ones_like(std_vi)  # 返回一个用标量值1填充的tensor，其大小与input相同

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)  # 按照一定的规则合并两个tensor
    map2 = torch.where((std_ir - std_vi) > 0, zero, one)  # 我觉得应该是torch.where((std_ir - std_vi) > 0, zero, one), 原版是>=, 但我改了

    ssim = map1 * ssim_ir + map2 * ssim_vi

    return ssim.mean()


def L_con_ssim(ir, vis, fuse):
    l_ssim = 1 - final_ssim(ir, vis, fuse)
    return l_ssim

def L_con_int(ir, vis, fuse, a):  # # 元素平方相加再开方
    loss_ir_int = torch.pow(torch.pow((ir-fuse), 2).sum(), 0.5)
    loss_vis_int = torch.pow(torch.pow((vis-fuse), 2).sum(), 0.5)
    l_int = (a * loss_ir_int + (1-a) * loss_vis_int) / (224 * 224)
    return l_int

def L_con_grad(ir, vis, fuse, b):
    kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, -1, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
    kernel = kernel.cuda()
    ir_grad = F.conv2d(ir, kernel.repeat(1, 1, 1, 1), stride=1, padding=1)
    vis_grad = F.conv2d(vis, kernel.repeat(1, 1, 1, 1), stride=1, padding=1)
    fuse_grad = F.conv2d(fuse, kernel.repeat(1, 1, 1, 1), stride=1, padding=1)
    loss_ir_grad = torch.pow(torch.pow((ir_grad-fuse_grad), 2).sum(), 0.5)
    loss_vis_grad = torch.pow(torch.pow((vis_grad-fuse_grad), 2).sum(), 0.5)
    l_grad = (b * loss_ir_grad + (1 - b) * loss_vis_grad) / (224 * 224)
    return l_grad

def L_con(ir, vis, fuse, a, b, alpha, beta, gamma):
    loss_int = L_con_int(ir, vis, fuse, a)
    loss_grad = L_con_grad(ir, vis, fuse, b)
    loss_ssim = L_con_ssim(ir, vis, fuse)
    l_con = alpha * loss_int + beta * loss_grad + gamma * loss_ssim
    return l_con

def L_adv_G(score_g_ir, score_g_vis):
    l_adv_g_ir = (-torch.log(score_g_ir + args.eps)).mean()
    l_adv_g_vis = (-torch.log(score_g_vis + args.eps)).mean()
    l_adv_g = l_adv_g_ir + l_adv_g_vis
    return l_adv_g

def L_G(ir, vis, fuse, a, b, alpha, beta, gamma, score_g_ir, score_g_vis, lamda):
    l_g = L_adv_G(score_g_ir, score_g_vis) + lamda * L_con(ir, vis, fuse, a, b, alpha, beta, gamma)
    return l_g

def L_D_ir(score_ir, score_g_ir):
    l_d_ir = (-torch.log(score_ir + args.eps)).mean()
    l_d_g_ir = (-torch.log(1.0 - score_g_ir + args.eps)).mean()
    d_ir_loss = l_d_ir + l_d_g_ir
    return d_ir_loss

def L_D_vis(score_vis, score_g_vis):
    l_d_vis = (-torch.log(score_vis + args.eps)).mean()
    l_d_g_vis = (-torch.log(1.0 - score_g_vis + args.eps)).mean()
    d_vis_loss = l_d_vis + l_d_g_vis
    return d_vis_loss

