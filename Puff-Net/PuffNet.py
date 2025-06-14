import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Optional, List
from torch import Tensor
from einops import rearrange
from function import normal, normal_style
from function import calc_mean_std
import numbers
# import scipy.stats as stats
from ViT_helper import DropPath, to_2tuple, trunc_normal_

# 图像转Patch嵌入模块
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)    # 将图像尺寸转换为二元组形式
        patch_size = to_2tuple(patch_size)  # Patch尺寸同上
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # 使用卷积将图像映射为 Patch 特征
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 前向传播：直接卷积获得 patch 表示
        x = self.proj(x)

        return x


# 解码器：用于从特征图还原图像
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),  # 最终输出为RGB图像
)

# 预定义 VGG 网络前层（用于感知损失）
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

# 多层感知机（全连接层序列）
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)  # 中间层维度
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



# Puff 模型：包括特征提取、分离、transformer融合、解码与训练损失计算
class Puff(nn.Module):
    """ This is the style transform transformer module """

    def __init__(self, encoder, decoder, PatchEmbed, BaseFeatureExtraction, DetailFeatureExtraction, transformer, args):

        super().__init__()
        enc_layers = list(encoder.children())  # 提取VGG子模块
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        # 冻结VGG参数
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.base = BaseFeatureExtraction           # 风格提取器
        self.detail = DetailFeatureExtraction       # 内容提取器
        self.mse_loss = nn.MSELoss()                # 使用MSE损失计算内容和风格差异
        self.transformer = transformer              # Transformer主结构
        hidden_dim = transformer.d_model            # 隐藏维度（未使用）
        self.decode = decoder                       # 解码器（生成图像）
        self.embedding = PatchEmbed                 # patch嵌入模块
        self.is_train = args.train                  # 是否为训练模式

    def encode_with_intermediate(self, input):
        # 获取VGG的多层特征输出（不含原图）
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        # 内容损失：原图与生成图的特征距离
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        # 风格损失：特征图的均值与标准差之间的距离
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)

    def forward(self, samples_c, samples_s):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # 前向传播，支持训练与推理两种模式
        if self.is_train:
            # content-style seperate
            # 内容图和风格图分别提取不同特征（Detail/Base）
            content_input = self.detail(samples_c)
            style_input = self.base(samples_s)
            content_input_style = self.base(samples_c)
            style_input_content = self.detail(samples_s)

            # features used to calcate loss
            # 获取VGG的中间层特征用于损失计算
            content_feats = self.encode_with_intermediate(samples_c)
            style_feats = self.encode_with_intermediate(samples_s)

            # seperate loss calcate
            cc_feats = self.encode_with_intermediate(content_input)
            cs_feats = self.encode_with_intermediate(content_input_style)
            sc_feats = self.encode_with_intermediate(style_input_content)
            ss_feats = self.encode_with_intermediate(style_input)
            # 四种方向组合的特征损失
            loss_cc = self.calc_content_loss(normal(cc_feats[-1]), normal(content_feats[-1])) + self.calc_content_loss(
                normal(cc_feats[-2]), normal(content_feats[-2]))
            loss_sc = self.calc_content_loss(normal(sc_feats[-1]), normal(style_feats[-1])) + self.calc_content_loss(
                normal(sc_feats[-2]), normal(style_feats[-2]))
            loss_cs = self.calc_style_loss(cs_feats[0], content_feats[0])
            loss_ss = self.calc_style_loss(ss_feats[0], style_feats[0])
            # 综合特征损失
            loss_fe = loss_ss + loss_cs + 0.7 * loss_cc + 0.7 * loss_sc

            # Linear projection
            # Patch Embedding 映射
            style = self.embedding(style_input)
            content = self.embedding(content_input)
            style_content = self.embedding(style_input_content)
            content_style = self.embedding(content_input_style)
            # style = self.embedding(samples_s)
            # content = self.embedding(samples_c)

            # postional embedding is calculated in transformer.py
            pos_s = None
            pos_c = None
            mask = None

            # Generation
            # 主Transformer模块生成融合特征
            hs = self.transformer(style, mask, content, pos_c, pos_s)
            Ics = self.decode(hs)  # 风格迁移图像

            # 内容损失
            Ics_feats = self.encode_with_intermediate(Ics)
            loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + self.calc_content_loss(
                normal(Ics_feats[-2]), normal(content_feats[-2]))
            # Style loss
            # 风格损失（5层）
            loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
            for i in range(1, 5):
                loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

            # Identity Loss：同源图重建（Icc, Iss）
            Icc = self.decode(self.transformer(content_style, mask, content, pos_c, pos_c))
            Iss = self.decode(self.transformer(style, mask, style_content, pos_s, pos_s))

            # Identity losses lambda 1
            # 图像级 identity 损失
            loss_lambda1 = self.calc_content_loss(Icc, samples_c) + self.calc_content_loss(Iss, samples_s)

            # Identity losses lambda 2
            # 特征级 identity 损失
            Icc_feats = self.encode_with_intermediate(Icc)
            Iss_feats = self.encode_with_intermediate(Iss)
            loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + self.calc_content_loss(Iss_feats[0],
                                                                                                           style_feats[0])
            for i in range(1, 5):
                loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(
                    Iss_feats[i], style_feats[i])
            # Please select and comment out one of the following two sentences
            # 返回生成图像 + 损失项
            return Ics, content_input, style_input, loss_fe, loss_c, loss_s, loss_lambda1, loss_lambda2  # train

        else:
            # 推理模式：仅生成迁移图像
            # content-style seperate
            content_input = self.detail(samples_c)
            style_input = self.base(samples_s)
            # Linear projection
            style = self.embedding(style_input)
            content = self.embedding(content_input)

            # postional embedding is calculated in transformer.py
            pos_s = None
            pos_c = None
            mask = None

            # Generation
            hs = self.transformer(style, mask, content, pos_c, pos_s)
            Ics = self.decode(hs)
            return Ics

