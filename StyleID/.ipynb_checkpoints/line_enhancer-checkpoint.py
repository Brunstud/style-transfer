import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# 将形状为 [1, 3, H, W] 且范围为 [0, 1] 的 Tensor 保存为 PNG 图片。
def save_image(tensor, path):
    np_img = tensor.squeeze().clamp(0, 1).cpu().numpy()  # [3,H,W]
    np_img = (np_img * 255).astype(np.uint8).transpose(1, 2, 0)  # [H,W,3]
    img = Image.fromarray(np_img)
    img.save(path)
    
# ======== XDoG config ========
XDoG_config = {
    'sigma': 0.6,
    'k': 2.5,
    'gamma': 0.97 + 0.01 * np.random.rand(),
    'eps': -15 / 255.0,
    'phi': 1e8
}

def unsharp_mask(image, times=3):
    """多轮USM锐化"""
    for _ in range(times):
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    return image

def apply_sobel(gray):
    """计算Sobel边缘图"""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    edge = (magnitude / magnitude.max() * 255).astype(np.uint8)
    return edge

def apply_xdog(gray):
    """基于XDoG的二值边缘提取"""
    sigma = XDoG_config['sigma']
    k = XDoG_config['k']
    gamma = XDoG_config['gamma']
    eps = XDoG_config['eps']
    phi = XDoG_config['phi']

    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = g1 - gamma * g2
    dog /= dog.max()
    e = 1 + np.tanh(phi * (dog - eps))
    e[e >= 1] = 1
    edge = 1 - e  # 翻转黑白
    edge = (edge * 255).astype(np.uint8)
    return edge

def enhance_line(img_tensor, method="sobel", save_path=None, threshold=50):
    """
    图像线条增强模块
    - 输入: img_tensor (B, C, H, W), 归一化至 [-1,1]
    - 输出: 增强后图像 tensor，仍为 [-1,1]
    """
    assert img_tensor.dim() == 4, "Expect 4D tensor"
    img = ((img_tensor + 1) / 2).clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)
    img_uint8 = (img * 255).astype(np.uint8)

    # 多轮锐化
    sharp = unsharp_mask(img_uint8.copy(), times=3)
    gray = cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY)

    if method == "sobel":
        edge = apply_sobel(gray)
    elif method == "xdog":
        edge = apply_xdog(gray)
    else:
        raise ValueError("method should be 'sobel' or 'xdog'")

    # 阈值化并膨胀
    _, imap = cv2.threshold(edge, threshold, 255, cv2.THRESH_BINARY)
    imap = cv2.dilate(imap, None, iterations=1)
    imap = np.expand_dims(imap, axis=2) / 255.0

    # 融合成pseudo-GT
    pseudo_gt = img * (1 - imap) + sharp / 255.0 * imap
    pseudo_gt_tensor = torch.from_numpy(pseudo_gt.transpose(2, 0, 1)).unsqueeze(0).float()
    pseudo_gt_tensor = 2 * pseudo_gt_tensor - 1

    # 可选保存图像
    if save_path:
        save_image((pseudo_gt_tensor + 1) / 2, save_path)
        print(f"Saved enhanced image to {save_path}")

    return pseudo_gt_tensor
