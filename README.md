フォーク元: https://github.com/pipixin321/HolmesVAU

このリポジトリは [HolmesVAU](https://arxiv.org/abs/2412.06171) を macOS で推論できるように修正したものです。

訓練等の対応は行なっておりません。

## 主な変更点

- `decord` → `opencv-python` に置き換え（macOS arm64 で decord がビルドできないため）
- MPS（Apple Silicon GPU）は未対応演算が多いため、macOS では CPU で動作
- デバイス選択の自動化（CUDA があれば CUDA、なければ CPU）

## 推論手順

### 1. 環境構築

[uv](https://docs.astral.sh/uv/) を使用します。

```bash
# uv のインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

### 2. モデルのダウンロード

2.1 [HolmesVAU-2B](https://huggingface.co/ppxin321/HolmesVAU-2B) をダウンロード
```bash
cd ./ckpts
git lfs install
git clone https://huggingface.co/ppxin321/HolmesVAU-2B
```

### 3. 推論の実行
```bash
uv run inference.py
```


## 注意事項
- CPU での推論は **8時間程度** かかります（サンプル動画 robbery.mp4 の場合）

## 推論の流れ

1. **動画読み込み**: OpenCV で動画を読み込み、総フレーム数を取得（例: 2730 フレーム）
2. **密サンプリング**: 16 フレームごとにサンプリング → 約 170 フレーム
3. **ViT 特徴抽出**: 各フレームを Vision Transformer に通して特徴量を抽出（← ここが重い）
4. **異常スコア計算**: Anomaly Scorer で各フレームの異常スコアを算出
5. **疎サンプリング**: 異常スコアに基づいて 12 フレームを選択（Anomaly-focused Temporal Sampling）
6. **LLM 推論**: 選択されたフレームを Multimodal LLM に入力し、異常イベントを説明
