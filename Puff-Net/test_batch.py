import os.path
from collections import OrderedDict

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image

import PuffNet
import transformer
import extraction
from argument import args
# from thop import profile

args.train = False

CONTENT_FOLDER = args.content_data
STYLE_FOLDER = args.style_data
OUTPUT_FOLDER = args.output
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Device
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

vgg = PuffNet.vgg
vgg.load_state_dict(torch.load("vgg_normalised.pth"))
vgg = nn.Sequential(*list(vgg.children())[:44])

base = extraction.BaseFeatureExtraction()
detail = extraction.DetailFeatureExtraction()
decoder = PuffNet.decoder
embedding = PuffNet.PatchEmbed()
Trans = transformer.Transformer()
decoder.eval()
embedding.eval()
Trans.eval()

# model load
for module, name in zip([base, decoder, detail, embedding, Trans],
                        ['base', 'decoder', 'detail', 'embedding', 'transformer']):
    state_dict = torch.load(f'model/{name}_iter_{args.iter}.pth')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k] = v
    module.load_state_dict(new_state_dict)
    module.eval()


network = PuffNet.Puff(vgg, decoder, embedding, base, detail, Trans, args)
network.eval()
network.to(device)

# ToTensor
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# You can try our model by using it
content_list = sorted([f for f in os.listdir(CONTENT_FOLDER) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
style_list = sorted([f for f in os.listdir(STYLE_FOLDER) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
print(f"Found {len(content_list)} content images and {len(style_list)} style images.")

with torch.no_grad():
    for content_name in content_list:
        for style_name in style_list:
            # 构建输出文件名
            out_name = f"{os.path.splitext(content_name)[0]}_stylized_{os.path.splitext(style_name)[0]}.png"
            save_path = os.path.join(OUTPUT_FOLDER, out_name)
            
            # === 🚫 如果文件已存在，跳过 ===
            # if os.path.exists(save_path):
            #     print(f"Skip existing: {save_path}")
            #     continue

            # 加载图像
            content_path = os.path.join(CONTENT_FOLDER, content_name)
            style_path = os.path.join(STYLE_FOLDER, style_name)
            content_img = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
            style_img = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)

            # 生成结果
            output = network(content_img, style_img).to('cpu')

            # 保存图片
            save_image(output, save_path)
            print(f"Saved: {save_path}")
