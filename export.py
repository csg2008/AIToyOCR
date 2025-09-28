#!/usr/bin/env python3
"""
导出 ONNX + 长文本性能测试
"""
import os
import time

import editdistance
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFont

from data import CHARSET, VOCAB_SIZE, idx2char, pad_id, sos_id, eos_id
from model import ViTTransformerOCR

EXPORT_FILE  = 'ocr_tiny_dynamic.onnx'
CHECKPOINT   = 'ocr_best_em.pth'   # 训练得到的权重
IMG_HEIGHT   = 32
IMG_WIDTH    = 512
MAX_CHARS    = 1500
STEP         = 10
FONT         = ImageFont.truetype("arial.ttf", 20)   # 如需更精细可换 TTF

# ---------- 1. 导出 ONNX ----------
@torch.no_grad()
def export_onnx():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = ViTTransformerOCR(vocab_size=VOCAB_SIZE,
                              d_model=384, nhead=6,
                              num_decoder_layers=6,
                              pad_id=pad_id, sos_id=sos_id, eos_id=eos_id).cuda().eval()
    ckpt = torch.load(CHECKPOINT, map_location='cuda')
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

    # 动态轴
    batch   = {0: 'B'}
    width   = {2: 'W'}          # 只让宽度动态，高度固定 32
    seq_len = {1: 'L'}

    dummy_img   = torch.randn(1, 1, IMG_HEIGHT, IMG_WIDTH).cuda()
    dummy_max   = torch.tensor([128])          # 最大输出长度占位符

    # 把模型包一层 forward 用于 ONNX
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.vocab_size = model.decoder.fc_out.out_features

        def forward(self, images, max_len):
            pred = self.model(images, mode='test', max_length=max_len[0].item())
            # pred: [B, L]  可能含非法值
            pred = torch.clamp(pred, 0, self.vocab_size - 1)
            return pred

    wrapped = ONNXWrapper(model)
    wrapped(dummy_img, dummy_max)   # 跑一次建图

    # 真正提速：先用 jit.trace 拿图
    traced = torch.jit.trace(wrapped, (dummy_img, dummy_max))

    torch.onnx.export(
        traced,
        (dummy_img, dummy_max),
        EXPORT_FILE,
        input_names=['images', 'max_len'],
        output_names=['pred_ids'],
        dynamic_axes={'images': batch | width,
                      'max_len': {0: 'B'},
                      'pred_ids': batch | seq_len},
        opset_version=14,
        dynamo=False,
        do_constant_folding=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    print(f'✅ ONNX 已导出 → {EXPORT_FILE}')
    return EXPORT_FILE

# ---------- 2. 生成单张长图 ----------
def gen_image(text):
    w = max(IMG_WIDTH, len(text) * 16)            # 粗略估算宽度
    img = Image.new('L', (w, IMG_HEIGHT), 255)
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), text, font=FONT, fill=0)
    tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor, text

# ---------- 3. ONNX 推理 ----------
def onnx_infer(session, img_tensor, max_len=2000):
    img_np = img_tensor.cpu().numpy()
    max_len_np = np.array([max_len], dtype=np.int64)
    tic = time.time()
    pred_ids = session.run(None, {'images': img_np, 'max_len': max_len_np})[0]
    toc = time.time()
    pred_str = ''.join([idx2char[i] for i in pred_ids[0] if i in idx2char])
    return pred_str, toc - tic

# ---------- 4. 主流程 ----------
def main():
    if not os.path.exists(EXPORT_FILE):
        export_onnx()

    # 加载 ONNX
    ort_sess = ort.InferenceSession(EXPORT_FILE, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    print('len', 'cer', 'time(ms)', 'pred_preview', sep='\t')
    for n_char in range(10, MAX_CHARS + 1, STEP):
        gt_text = ''.join(np.random.choice(list(CHARSET), n_char))
        img_tensor, _ = gen_image(gt_text)
        pred_text, dt = onnx_infer(ort_sess, img_tensor, max_len=n_char + 20)
        cer = editdistance.eval(pred_text, gt_text) / max(len(gt_text), 1)
        print(n_char, f'{cer:.3f}', f'{dt*1000:.1f}',
              pred_text[:30] + ('...' if len(pred_text) > 30 else ''), sep='\t')

if __name__ == '__main__':
    main()