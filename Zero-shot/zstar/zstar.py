import os
from math import sqrt

import torch
import torch.nn.functional as F
import numpy as np
import shutil

from einops import rearrange

from .zstar_utils import AttentionBase

# 跨注意力重加权控制器，用于在指定步骤/层级对 cross-attention 模块进行干预
class ReweightCrossAttentionControl(AttentionBase):
    def __init__(self, start_step=4, end_step=1000, start_layer=10, end_layer=1000, layer_idx=None, step_idx=None, total_steps=50, content_img_name=None):
        """
        构造函数，初始化要干预的 diffusion 步骤和网络层
        
        参数：
        - start_step / end_step: 开始和结束的 diffusion 步骤索引
        - start_layer / end_layer: 开始和结束的 attention 层索引
        - layer_idx: 明确指定需要重加权的层索引列表
        - step_idx: 明确指定需要重加权的步骤列表
        - total_steps: diffusion 总步数（通常是 50）
        - content_img_name: 如果需要用掩码辅助，就需要指定内容图路径（从而推导出 mask.npy）
        """
        super().__init__()
        self.total_layers = 16  # 模型中的 cross-attention 总层数
        self.total_steps = total_steps
        self.start_step = max(0, start_step)
        self.end_step = min(end_step, total_steps)
        self.start_layer = max(0, start_layer)
        self.end_layer = min(end_layer, self.total_layers)
        self.layer_idx = layer_idx if layer_idx is not None else list(
            range(self.start_layer, self.end_layer))
        self.step_idx = step_idx if step_idx is not None else list(
            range(self.start_step, self.end_step))
        self.content_img_name = content_img_name
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)

    # 用于计算 q, k 相似度（sim）的辅助函数（标准 attention 相似度公式）
    def get_batch_sim(self, type, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)  # 将 Q 变成 [H, BN, D]
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")  # sim = QK^T
        return sim

    # 引入 mask 衰减机制的注意力相似度计算（通常用于 spatial control）
    def get_batch_sim_with_mask(self, type, cc_sim, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        head_num = sim.shape[0]
        pixel_size = sim.shape[1]
        h = w = int(sqrt(pixel_size))  # 把 1D flatten 像素映射为 2D
        sim_reshaped = sim.reshape(head_num, h, w, pixel_size)
        cc_sim_reshaped = cc_sim.reshape(head_num, h, w, pixel_size)
        # 提取最小和最大值，形成动态范围
        min_cc_sim_reshaped, _ = torch.min(
            cc_sim_reshaped, dim=3, keepdim=True)
        max_sim_reshaped, _ = torch.max(sim_reshaped, dim=3, keepdim=True)
        start = 0.5
        end = -0.5
        length = w
        # 尝试加载图像对应的 mask，如果失败则使用默认值
        if self.content_img_name is not None:
            mask_path = self.content_img_name.replace(".jpg", "_mask.npy")
            mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
            mask[mask < 0.5] = -1.0
            mask[mask > 0.5] = 1.0
            mask *= -1.0  # 翻转掩码值
        else:
            print("ERROR: mask npy not found!!!")
            mask = torch.tensor([[1., 1., 1., 1.],
                                [1., -1., -1., 1.],
                                [1., -1., -1., 1.],
                                [-1., -1., -1., -1.]], dtype=torch.float32).cuda()
        # 插值调整大小
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(
            h, w), mode='bilinear', align_corners=False)
        gradual_vanished_array = mask.reshape(1, h, w, 1).to(sim.device)
        # 将 mask 衰减项叠加到 sim 上
        gradual_vanished_mask = (
            min_cc_sim_reshaped - max_sim_reshaped)[:, :, :, :] * gradual_vanished_array
        sim_reshaped[:, :length, :, :] += gradual_vanished_mask
        sim = sim_reshaped.reshape(head_num, pixel_size, pixel_size)
        return sim

    # latest
    # 自定义 forward 函数，重写 attention 输出
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        # 如果当前 step 或 layer 不需要重加权，使用默认逻辑
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        # 按 batch 结构拆分 q/k/v 为 style 和 content 两部分
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        # 计算所有组合的相似度
        style_style_out_u_sim = self.get_batch_sim(
            "ss", qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        style_style_out_c_sim = self.get_batch_sim(
            "ss", qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        content_content_out_u_sim = self.get_batch_sim(
            "cc", qu[-num_heads:], ku[-num_heads:], vu[-num_heads:], sim[-num_heads:], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        content_content_out_c_sim = self.get_batch_sim(
            "cc", qc[-num_heads:], kc[-num_heads:], vc[-num_heads:], sim[-num_heads:], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        style_content_out_u_sim = self.get_batch_sim(
            "sc", qu[:num_heads], ku[-num_heads:], vu[-num_heads:], sim[-num_heads:], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        style_content_out_c_sim = self.get_batch_sim(
            "sc", qc[:num_heads], kc[-num_heads:], vc[-num_heads:], sim[-num_heads:], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        content_style_out_u_sim = self.get_batch_sim(
            "cs", qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        content_style_out_c_sim = self.get_batch_sim(
            "cs", qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        # 手动放大 style-content 和 content-style 的相似度权重
        content_style_out_u_sim *= 1.2
        content_style_out_c_sim *= 1.2

        style_content_out_u_sim *= 1.5
        style_content_out_c_sim *= 1.5

        # 计算 batch size（用于 rearrange）
        b = qu[-num_heads:].shape[0] // num_heads

        # 拼接内容-风格和内容-内容的注意力图和 V
        content_style_content_content_out_u_sim = torch.cat(
            (content_style_out_u_sim, content_content_out_u_sim), 2)
        content_style_content_content_out_c_sim = torch.cat(
            (content_style_out_c_sim, content_content_out_c_sim), 2)
        vu_cscc_concat = torch.cat((vu[:num_heads], vu[-num_heads:]), 1)
        vc_cscc_concat = torch.cat((vc[:num_heads], vc[-num_heads:]), 1)

        # 拼接风格-内容和风格-风格的注意力图和 V
        style_content_style_style_out_u_sim = torch.cat(
            (style_content_out_u_sim, style_style_out_u_sim), 2)
        style_content_style_style_out_c_sim = torch.cat(
            (style_content_out_c_sim, style_style_out_c_sim), 2)
        vu_scss_concat = torch.cat((vu[-num_heads:], vu[:num_heads]), 1)
        vc_scss_concat = torch.cat((vc[-num_heads:], vc[:num_heads]), 1)

        # 归一化后乘 V 得到输出
        content_style_content_content_out_u_sim = content_style_content_content_out_u_sim.softmax(
            -1)
        content_style_content_content_out_c_sim = content_style_content_content_out_c_sim.softmax(
            -1)

        # 将 heads 还原回 batch 维度
        mixup_out_u = torch.einsum(
            "h i j, h j d -> h i d", content_style_content_content_out_u_sim, vu_cscc_concat)
        mixup_out_u = rearrange(mixup_out_u, "h (b n) d -> b n (h d)", b=b)
        mixup_out_c = torch.einsum(
            "h i j, h j d -> h i d", content_style_content_content_out_c_sim, vc_cscc_concat)
        mixup_out_c = rearrange(mixup_out_c, "h (b n) d -> b n (h d)", b=b)

        style_content_style_style_out_u_sim = style_content_style_style_out_u_sim.softmax(
            -1)
        style_content_style_style_out_c_sim = style_content_style_style_out_c_sim.softmax(
            -1)
        original_out_u = torch.einsum(
            "h i j, h j d -> h i d", style_content_style_style_out_u_sim, vu_scss_concat)
        original_out_u = rearrange(
            original_out_u, "h (b n) d -> b n (h d)", b=b)
        original_out_c = torch.einsum(
            "h i j, h j d -> h i d", style_content_style_style_out_c_sim, vc_scss_concat)
        original_out_c = rearrange(
            original_out_c, "h (b n) d -> b n (h d)", b=b)

        # 拼接最终结果（前一半 batch 是 style，后一半 batch 是 content）
        out = torch.cat([original_out_u, mixup_out_u,
                        original_out_c, mixup_out_c], dim=0)
        return out
