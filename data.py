import random
import string

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

# 字符集：数字+大小写字母+特殊字符
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
CHARSET = list(string.digits) + list(string.ascii_letters) + list(string.punctuation) + [' '] + [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
VOCAB_SIZE = len(CHARSET)
char2idx = {char: idx for idx, char in enumerate(CHARSET)}
idx2char = {idx: char for char, idx in char2idx.items()}
pad_id   = char2idx[PAD_TOKEN]
sos_id   = char2idx[SOS_TOKEN]
eos_id   = char2idx[EOS_TOKEN]

class OCRDataset(Dataset):
    def __init__(self, num_samples=1000, img_height=32, min_width=50, max_chars=50):
        self.num_samples = num_samples
        self.img_height = img_height
        self.min_width = min_width
        self.max_chars = max_chars

        try:
            self.font = ImageFont.truetype("arial.ttf", 20)
        except OSError:
            print("无法加载 'arial.ttf' 字体，将使用系统默认字体。")
            self.font = ImageFont.load_default()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机生成文字
        text = ''.join(random.choices(CHARSET[:-3], k=random.randint(10, self.max_chars)))

        # 计算文本像素宽度（用 getbbox 最准确）
        bbox = self.font.getbbox(text)                # (left, top, right, bottom)
        text_w = bbox[2] - bbox[0]                    # 文字宽度
        pad_w = random.randint(20, 60)            # 左右边距 20~60 像素
        img_w = text_w + pad_w                        # 真实所需宽度
        width = max(img_w, self.min_width)            # 不低于全局最小

        # 创建图像
        img = Image.new('L', (width, self.img_height), 255)
        draw = ImageDraw.Draw(img)

        # 绘制文字
        draw.text((5, 5), text, font=self.font, fill=0)

        # 转换为tensor
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # 添加通道维度

        # 文字转索引
        text_indices = [char2idx[c] for c in text]
        text_tensor = torch.tensor(text_indices, dtype=torch.long)

        return {
            'image': img_tensor,
            'text': text_tensor,
            'text_length': len(text),
            'width': width
        }

def collate_fn(batch):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    widths = [item['width'] for item in batch]

    # ✅ 将宽度向上对齐到 16 的倍数对齐（ViT patch）
    def round_up_to_patch_size(w, patch_size=16):
        return ((w + patch_size - 1) // patch_size) * patch_size

    max_width = max(widths)
    max_width = round_up_to_patch_size(max_width)

    # 填充图片到相同宽度
    padded_images = []
    for img in images:
        _, h, w = img.shape
        padding = max_width - w
        padded_img = torch.nn.functional.pad(img, (0, padding), value=1.0)
        padded_images.append(padded_img)

    images_tensor = torch.stack(padded_images)

    # 处理文字序列（不变）
    text_lengths = [len(text) for text in texts]
    max_text_len = max(text_lengths)
    padded_texts = []
    for text in texts:
        padding = max_text_len - len(text)
        text = torch.cat([torch.tensor([sos_id]), text, torch.tensor([eos_id])])
        padded = torch.nn.functional.pad(text, (0, max_text_len - len(text) + 2), value=pad_id)
        padded_texts.append(padded)
    texts_tensor = torch.stack(padded_texts)

    return {
        'images': images_tensor,
        'texts': texts_tensor,
        'text_lengths': torch.tensor(text_lengths),
        'widths': torch.tensor([max_width] * len(batch))  # 统一宽度
    }
