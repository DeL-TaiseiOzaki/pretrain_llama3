import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import json
import random

class TextDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = 2048):
        self.max_length = max_length
        
        # データの読み込み
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.json'):
                self.samples = [json.loads(line) for line in f]
            else:
                self.samples = f.readlines()
                
        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # JSONの場合
        if isinstance(sample, dict):
            if 'text' in