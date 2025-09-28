import math

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, register_model

@register_model
def vit_tiny_patch8_224(pretrained=False, **kwargs):
    # 去掉 VisionTransformer 不认识的参数
    # 这些参数是 timm create_model 为了下载权重、缓存等额外塞进来的
    for k in ('pretrained_cfg', 'pretrained_cfg_overlay', 'cache_dir'):
        kwargs.pop(k, None)

    model = VisionTransformer(
        img_size=224,
        patch_size=8,
        embed_dim=192,   # tiny 级别
        depth=12,
        num_heads=3,     # 192 // 64
        mlp_ratio=2,
        **kwargs
    )
    return model

class TransformerDecoder(nn.Module):
    """独立的Transformer解码器类"""
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_seq_length=1024):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_length)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, memory_key_mask=None):
        # tgt: [batch, tgt_seq_len]
        # memory: [batch, src_seq_len, d_model]

        # 嵌入和位置编码
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)

        # 生成目标序列的掩码（防止看到未来信息）
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # 生成内存掩码（如果需要）
        if memory_key_mask is None:
            memory_key_mask = (memory.abs().sum(-1) == 0) # [B, src_len]

        # Transformer解码
        output = self.transformer_decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_mask
        )

        # 输出层
        output = self.fc_out(output)

        return output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, C]
        x = x + self.pe[:, :x.size(1), :]   # 自动广播到 [B, T, C]
        return self.dropout(x)

class ViTTransformerOCR(nn.Module):
    """完整的OCR模型：ViT编码器 + Transformer解码器"""
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6, pad_id=-1, sos_id=-1, eos_id=-1):
        super().__init__()

        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        # 使用timm创建ViT模型（支持单通道）
        # 创建 ViT 模型（不限制输入尺寸）
        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            in_chans=1,
            num_classes=0,          # 去掉分类头
            img_size=None,
            dynamic_img_size=True,
            global_pool='',         # ✅ 关键：禁止池化，保留 patch 序列
        )

        # 获取ViT的特征维度
        self.vit_feature_dim = self.vit.embed_dim

        # 将ViT输出映射到解码器维度
        self.linear_mapping = nn.Linear(self.vit_feature_dim, d_model)

        # Transformer解码器
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers
        )

    def forward(self, images, target_texts=None, max_length=50, mode='train'):
        # 编码器部分
        # images: [batch, 1, H, W]
        vit_features = self.vit(images)  # [batch, num_patches, vit_feature_dim]
        memory = self.linear_mapping(vit_features)  # [batch, num_patches, d_model]

        if mode == 'train' and target_texts is not None:
            # 训练模式：使用teacher forcing
            # 在目标序列前添加起始符（这里用0表示）
            batch_size = target_texts.size(0)
            device = target_texts.device

            # 创建起始符
            start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            decoder_input = torch.cat([start_token, target_texts], dim=1)

            # 移除最后一个token作为输入
            decoder_input = decoder_input[:, :-1]

            # 解码
            output = self.decoder(decoder_input, memory)
            return output

        else:  # 推理模式
            batch_size = images.size(0)
            device = images.device
            # 设置起始符 <sos> 的 id
            decoder_input = torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=device)

            outputs = []                 # 每个时间步的预测 token
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)  # 标记已结束序列

            for _ in range(max_length):
                # 解码
                output = self.decoder(decoder_input, memory)        # [batch, seq_len, vocab_size]

                # 获取最后一个时间步的预测
                next_token = output[:, -1:, :].argmax(dim=-1)       # [batch, 1]

                # 防止越界（保留）
                next_token = next_token.clamp(0, self.decoder.fc_out.out_features - 1)

                # 关键：如果某条样本已经 finished，强制写 PAD，保持长度对齐
                next_token = torch.where(finished, torch.full_like(next_token, self.pad_id), next_token)
                outputs.append(next_token)

                # 更新 finished 标记（只要出现一次 EOS 就永久 finished）
                finished |= (next_token.squeeze(1) == self.eos_id)

                # 提前退出条件：全部序列都停
                if finished.all():
                    break

                # 准备下一步输入
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # 拼成 [B, L] 返回
            if outputs:
                outputs = torch.cat(outputs, dim=1)
            else:
                outputs = torch.empty(batch_size, 0, dtype=torch.long, device=device)

            return outputs
