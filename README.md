# pretrain_llama3

LLaMA (Large Language Model Meta AI)アーキテクチャを基にした事前学習用の実装です。DeepSpeedを使用した分散学習に対応しています。

## 特徴

- LLaMAアーキテクチャの完全な実装
  - RMSNorm
  - Rotary Position Embedding (RoPE)
  - SwiGLU Activation
  - Group Query Attention

- 高度な分散学習対応
  - DeepSpeed Stage 3による最適化
  - CPU Offloading
  - 混合精度学習(FP16)
  - 勾配の累積

- モデル設定
  - デフォルトで4096次元、32層、32ヘッド
  - 語彙数32,000
  - 最大シーケンス長2048
  - Group Query Attention (8 KV heads)

## 必要要件

torch>=2.0.0 deepspeed>=0.10.0 transformers>=4.30.0 numpy>=1.24.0 tqdm>=4.65.0 wandb>=0.15.0 # オプション: 実験管理用

## インストール

```bash
pip install -r requirements.txt
```

使用方法
学習の実行
# 環境変数の設定
export TRAIN_DATA_PATH=/path/to/your/training/data

# DeepSpeedを使用した分散学習の実行
deepspeed --num_gpus=8 train.py

設定のカスタマイズ
config/config.pyで以下のようなハイパーパラメータを調整できます：

モデル構造: 次元数、レイヤー数、ヘッド数など
学習設定: バッチサイズ、エポック数、学習率など
DeepSpeed設定: ZeRO stage、オフロード設定など
主要コンポーネント
model/llama.py: LLaMAモデルの実装
model/dataset.py: データセット処理
config/config.py: 設定管理
train.py: 学習実行スクリプト
GPU要件
推奨: NVIDIA H100 x 8
DeepSpeedのStage 3とCPU offloadingにより、限られたGPUメモリでも大規模モデルの学習が可能
