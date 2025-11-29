import os
import sys
import torch
import matplotlib.pyplot as plt
from holmesvau.holmesvau_utils import load_model, generate, show_smapled_video

mllm_path = './ckpts/HolmesVAU-2B'
sampler_path = './holmesvau/ATS/anomaly_scorer.pth'

# OS に応じてデバイスを自動選択
# 注: MPS は bicubic 補間など未実装の演算が多く、このモデルでは動作しないため macOS は CPU を使用
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")
model, tokenizer, generation_config, sampler = load_model(mllm_path, sampler_path, device)

video_path = "./examples/robbery.mp4"
prompt = "Could you specify the anomaly events present in the video?"
pred, history, frame_indices, anomaly_score = generate(video_path, prompt, model, tokenizer, generation_config, sampler, select_frames=12, use_ATS=True)
print('\nUser:', prompt, '\nHolmesVAU:', pred)


