import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer
from holmesvau.ATS.Temporal_Sampler import Temporal_Sampler
from holmesvau.internvl_utils import build_transform, get_index, dynamic_preprocess


# decord.VideoReader の cv2 ベース代替実装
class VideoReader:
    """decord.VideoReader 互換ラッパー（cv2 ベース）"""
    def __init__(self, path, ctx=None, num_threads=1):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self._len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __len__(self):
        return self._len

    def get_avg_fps(self):
        return self._fps

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError(f"Cannot read frame {idx}")
        # decord は RGB で返すので BGR→RGB 変換
        return _FrameWrapper(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def __del__(self):
        if self.cap:
            self.cap.release()


class _FrameWrapper:
    """decord の frame.asnumpy() 互換"""
    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr


def cpu(n):
    """decord.cpu() のダミー"""
    return None


def load_model(mllm_path, sampler_path, device):
    # CUDA: bfloat16 + Flash Attention、CPU: bfloat16（MPS は未サポート）
    use_flash_attn = device.type == 'cuda'

    model = AutoModel.from_pretrained(
        mllm_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(mllm_path, trust_remote_code=True, use_fast=False)
    model = model.to(device)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    sampler = Temporal_Sampler(sampler_path, device)
    return model, tokenizer, generation_config, sampler

def get_pixel_values(vr, frame_indices, input_size=448, max_num=1):
    transform = build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def generate(video_path, prompt, model, tokenizer, generation_config, sampler, dense_sample_freq=16, select_frames=12, use_ATS=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    print("Frame Number: ", len(vr))
    if use_ATS and len(vr) > dense_sample_freq * select_frames:
        # dense sampling 
        print("Anomaly-fouced Temporal Sampling...")
        dense_frame_indices = list(range(len(vr)))[::dense_sample_freq]
        pixel_values, num_patches_list = get_pixel_values(vr, dense_frame_indices)
        # anomaly-focused sampling
        anomaly_score, sampled_idxs = sampler.density_aware_sample(pixel_values, model, select_frames)
        sparse_pixel_values = pixel_values[sampled_idxs]
        frame_indices, num_patches_list = [dense_frame_indices[i] for i in sampled_idxs], [num_patches_list[i] for i in sampled_idxs]
        print('Sampled frames: ', frame_indices)
    else:
        # uniform sampling
        print("Uniform Sampling...")
        frame_indices = get_index(bound=None, fps=float(vr.get_avg_fps()), max_frame=len(vr)-1, first_idx=0, num_segments=select_frames)
        frame_indices = list(map(int, frame_indices))
        sparse_pixel_values, num_patches_list = get_pixel_values(vr, frame_indices)
        anomaly_score = None
        
    # generate
    history = None
    # モデルの dtype に合わせる（MPS は bfloat16 未対応）
    sparse_pixel_values = sparse_pixel_values.to(model.dtype).to(model.device)
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + prompt
    # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
    response, history = model.chat(tokenizer, sparse_pixel_values, question, generation_config,
                                num_patches_list=num_patches_list, history=history, return_history=True)
    return response, history, frame_indices, anomaly_score

def show_smapled_video(vr, idx_list=None, segment=None, labels=None):
    if idx_list is None:
        if segment is None:
            idx_list = np.linspace(0, len(vr)-1, 8, dtype=int)
        else:
            idx_list = np.linspace(segment[0], segment[1]-1, 8, dtype=int)
    all_frame = []
    for i in idx_list:
        frame = vr[i].asnumpy() #[h,w,3]
        all_frame.append(frame)
    h,w,c = all_frame[0].shape
    frame_show = np.zeros((h, w*len(all_frame), c), dtype=np.uint8)
    for i in range(len(all_frame)):
        frame_show[:, i*w:(i+1)*w, :] = all_frame[i]
        frame_show[:, i*w:i*w+5, :] = 255

    plt.figure(figsize=(20,10))
    plt.imshow(frame_show)
    plt.axis('off')
    plt.show()
    plt.close()