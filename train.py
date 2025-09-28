#!/usr/bin/env python3
"""
支持每 epoch 训练后验证的 OCR 训练脚本
"""
import math
import os
import editdistance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import VOCAB_SIZE, OCRDataset, collate_fn, idx2char, pad_id, sos_id, eos_id
from model import ViTTransformerOCR

# ---------- 配置 ----------
IMG_H, IMG_MIN_W = 32, 50                       # 更宽场景
MAX_CHARS    = 50                               # 最多字符数
D_MODEL      = 384                              # tiny 用 384 省显存
BATCH_SIZE   = 320
ACCUM_STEP   = 1
NUM_EPOCHS   = 60
LR           = 1e-4
NUM_WORKERS  = 16
NUM_TRAIN    = 95_000                           # 100k 里 95k 训练 5k 验证
NUM_VAL      = 5_000
CHECKPOINT     = 'ocr_tiny_best.pth'            # 断点文件
CHECKPOINT_CER = 'ocr_best_cer.pth'
CHECKPOINT_EM  = 'ocr_best_em.pth'
PRETRAINED_VIT = 'vit_tiny_patch16_224'
# --------------------------

def cer_score(pred, gold, length):
    """
    pred:  模型输出的一维 tensor，长度 = max_len，可能含 PAD/EOS
    gold:  真值一维 tensor，长度 = max_len，可能含 PAD/EOS
    length: 真值实际字符数（不含 BOS/EOS/PAD）
    返回：CER = editdistance / len(gold_text)
    """
    # 1. 把 tensor 转成 list，去掉 PAD 和 EOS
    def _clean(seq):
        out = []
        for idx in seq.tolist():
            if idx == pad_id:          # PAD
                continue
            if idx == eos_id:          # EOS，直接截断
                break
            if idx in idx2char:        # 正常字符
                out.append(idx2char[idx])
        return out

    pred_clean = _clean(pred)
    gold_clean = _clean(gold[:length])   # 只取有效长度

    pred_str = ''.join(pred_clean)
    gold_str = ''.join(gold_clean)

    if not gold_str:
        return 0.0

    return editdistance.eval(pred_str, gold_str) / len(gold_str)

def exact_match(pred, gold, length):
    # 有效长度
    len_gold = length.item() if isinstance(length, torch.Tensor) else length
    len_pred = (pred != pad_id).logical_and(pred != eos_id).sum().item()

    min_len = min(len_pred, len_gold)
    if min_len == 0:                       # 空串
        return len_pred == len_gold

    return torch.equal(pred[:min_len], gold[:min_len])

def count_params(model, name):
    c = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{name:20s}: {c/1e6:6.2f} M')
    return c

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss, running_cer, running_em, samples = 0.0, 0.0, 0, 0
    pbar = tqdm(loader, desc='Validate')
    for batch in pbar:
        images = batch['images'].cuda(non_blocking=True)
        texts  = batch['texts'].cuda(non_blocking=True)
        tgt_len = batch['text_lengths']

        with autocast('cuda'):
            logits = model(images, texts, mode='train')
            loss = criterion(logits.view(-1, logits.size(-1)), texts.view(-1))

        preds = logits.argmax(-1)
        bs = images.size(0)
        em_cnt = 0
        for p, t, L in zip(preds, texts, tgt_len):
            running_cer += cer_score(p, t, L)
            em_cnt += exact_match(p, t, L)
        running_em += em_cnt
        samples += bs
        running_loss += loss.item() * bs
        pbar.set_postfix(loss=f'{running_loss/samples:.4f}',
                         cer=f'{running_cer/samples:.4f}',
                         acc=f'{running_em/samples:.2%}')
    return running_loss/samples, running_cer/samples, running_em/samples

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    running_loss, running_cer, running_em, samples = 0.0, 0.0, 0, 0
    pbar = tqdm(loader, desc=f'Train E{epoch}')
    for idx, batch in enumerate(pbar):
        images = batch['images'].cuda(non_blocking=True)
        texts  = batch['texts'].cuda(non_blocking=True)
        tgt_len = batch['text_lengths']

        with autocast('cuda'):
            logits = model(images, texts, mode='train')
            loss = criterion(logits.view(-1, logits.size(-1)), texts.view(-1))

        scaler.scale(loss).backward()
        if (idx + 1) % ACCUM_STEP == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        preds = logits.argmax(-1)
        bs = images.size(0)
        em_cnt = 0
        for p, t, L in zip(preds, texts, tgt_len):
            running_cer += cer_score(p, t, L)
            em_cnt += exact_match(p, t, L)
        running_em += em_cnt
        samples += bs
        running_loss += loss.item() * bs
        pbar.set_postfix(loss=f'{running_loss/samples:.4f}',
                         cer=f'{running_cer/samples:.4f}',
                         acc=f'{running_em/samples:.2%}')
    return running_loss/samples, running_cer/samples, running_em/samples

def main():
    # 已有 amp，再打开：
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    device = torch.device('cuda')
    # 1. 数据集拆分
    full_set = OCRDataset(num_samples=NUM_TRAIN + NUM_VAL,
                        img_height=IMG_H,
                        min_width=IMG_MIN_W,
                        max_chars=MAX_CHARS)
    train_set, val_set = random_split(full_set, [NUM_TRAIN, NUM_VAL],
                                      generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, collate_fn=collate_fn)

    # 2. 模型
    model = ViTTransformerOCR(vocab_size=VOCAB_SIZE,
                              d_model=D_MODEL,
                              nhead=6,
                              num_decoder_layers=6,
                              pad_id=pad_id, sos_id=sos_id, eos_id=eos_id).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.AdamW([
        {'params': model.vit.parameters(),            'lr': 1e-5},
        {'params': model.linear_mapping.parameters(), 'lr': 5e-5},
        {'params': model.decoder.parameters(),        'lr': 2e-4},
    ], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler    = GradScaler(init_scale=2048, growth_interval=100)

    vit   = model.vit
    dec   = model.decoder
    map_m = model.linear_mapping

    total = 0
    total += count_params(vit,   'ViT-Tiny')
    total += count_params(map_m, 'LinearMapping')
    total += count_params(dec,   'TransformerDec')
    print('-'*40)
    print(f'Total Trainable: {total/1e6:6.2f} M')

    # 3. 断点续训
    start_epoch = 0
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        print(f'🔥 续训 epoch {start_epoch}')

    best_cer, best_em, best_em_save = 1.0, 0.0, 0.0
    patience_loss, patience_em = 10, 10   # 连续不改善轮数
    best_train_loss = float('inf')
    counter_loss, counter_em = 0, 0

    # 4. 训练
    # model = torch.compile(model, mode='max-autotune')   # 训练步 10-15 % 提速
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f'\n----- Epoch {epoch} -----')
        train_loss, train_cer, train_em = train_one_epoch(model, train_loader,
                                                          criterion, optimizer, scaler, epoch)
        val_loss, val_cer, val_em = validate(model, val_loader, criterion)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
        print(f'LR={current_lr:.6f} GradNorm={total_norm:.2f}')
        print(f'Train  loss={train_loss:.4f}  CER={train_cer:.4f}  EM={train_em:.2%}')
        print(f'Val    loss={val_loss:.4f}  CER={val_cer:.4f}  EM={val_em:.2%}')

        # ---------- 1. 训练 loss 异常 ----------
        if not math.isfinite(train_loss):
            print('❌ Train loss 异常，立即终止训练')
            break

        # ---------- 2. 前 3  epoch 不监控 ----------
        if epoch < 3:
            continue

        # ---------- 3. Train loss 10 轮不下降 ----------
        if train_loss < best_train_loss - 1e-4:
            best_train_loss = train_loss
            counter_loss = 0
        else:
            counter_loss += 1
        if counter_loss >= patience_loss:
            print(f'⚠️  Train loss 连续 {patience_loss}  epoch 无下降，提前停止')
            break

        # ---------- 4. Val_EM 达标或 10 轮不涨 ----------
        if val_em >= 0.992:
            print('🎉  Val_EM 达到 99.2 %，训练提前完成')
            break
        if val_em > best_em + 1e-4:
            best_em = val_em
            counter_em = 0
        else:
            counter_em += 1
        if counter_em >= patience_em:
            print(f'⚠️  Val_EM 连续 {patience_em}  epoch 无提升，提前停止')
            break

        # ---------- 5. 保存最优 ----------
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch,
                    'best_cer': best_cer, 'best_em': best_em}, CHECKPOINT)
        if val_cer > 0 and val_cer < best_cer:
            best_cer = val_cer
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                        'scaler': scaler.state_dict(), 'epoch': epoch,
                        'best_cer': best_cer, 'best_em': best_em}, CHECKPOINT_CER)
            print(f'✅ 最佳 CER 模型已保存  CER={best_cer:.2%}')
        if val_em > best_em_save:      # 与上面 best_em 分开，避免提前停止干扰
            best_em_save = val_em
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                        'scaler': scaler.state_dict(), 'epoch': epoch,
                        'best_cer': best_cer, 'best_em': best_em}, CHECKPOINT_EM)
            print(f'🚀 最佳 Exact-Match 模型已保存  EM={best_em:.2%}')

    print('\n训练结束（可能提前停止）')

if __name__ == '__main__':
    main()