# config.py
import torch

CONFIG = {
    # === 実験設定 ===
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dimension": 256,
    "epochs": 20,
    "batch_size": 128,
    "margin": 0.3,
    "lr1": 1e-4,#encoder用の初期学習率
    "lr2": 1e-4,#aggregator用の学習率
    "weight_decay": 1e-4,

    # === Transformer構造 ===
    "num_layers": 4,
    "num_heads": 4,
    "ff_dim": 512,
    "max_len": 264,
    "aggregator_hidden_dim": 128, #MLPを追加する場合の隠れ層次元

    # === Ablationスイッチ ===
    "use_positional_embedding": True,   # Falseにするとnn.Identity()で位置エンコーディングなし
    "use_mlp_aggregator": True,         # Falseにすると単純平均でMLPなし
    
    # === データパス ===
    "category_dir": "data/compressed_rank32",
    "dataset_dir": "data/image_base_dataset",
}
