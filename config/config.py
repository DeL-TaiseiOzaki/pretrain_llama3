from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class Config:
    # モデルの構造パラメータ
    dim: int = 4096                # 埋め込みの次元
    n_layers: int = 32            # トランスフォーマーレイヤーの数
    n_heads: int = 32             # アテンションヘッドの数
    n_kv_heads: int = 8           # キー/バリューヘッドの数
    vocab_size: int = 32000       # 語彙サイズ
    multiple_of: int = 256        # 隠れ層の次元数の倍数
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5        # Layer Normalizationのイプシロン
    rope_theta: float = 10000.0   # RoPEのθパラメータ
    max_seq_len: int = 2048       # 最大シーケンス長

    # 学習パラメータ
    batch_size: int = 32          # バッチサイズ
    epochs: int = 100             # エポック数
    learning_rate: float = 1e-4   # 学習率
    weight_decay: float = 0.1     # Weight decay
    warmup_steps: int = 2000      # Warmupステップ数
    log_interval: int = 10        # ログ出力間隔
    eval_interval: int = 1000     # 評価間隔
    save_interval: int = 1000     # モデル保存間隔
    checkpoint_dir: str = 'checkpoints'

    # DeepSpeed設定
    ds_config: Dict[str, Any] = None

    def __post_init__(self):
        # DeepSpeed設定の初期化
        self.ds_config = {
            "train_batch_size": self.batch_size,
            "train_micro_batch_size_per_gpu": self.batch_size // 8,  # H100 8枚で分散
            "gradient_accumulation_steps": 1,
            
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8
                }
            },

            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": self.epochs * 1000,  # 要調整
                    "warmup_num_steps": self.warmup_steps
                }
            },

            "fp16": {
                "enabled": True,
                "auto_cast": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },

            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5
            },

            "gradient_clipping": 1.0,
            "steps_per_print": self.log_interval,
            "wall_clock_breakdown": False
        }

    @property
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')