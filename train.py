import os
import torch
import torch.distributed as dist
import deepspeed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from config.config import Config
from model.llama import Llama

class TextDataset(Dataset):
    def __init__(self, data_path, max_length):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # ここで実際のデータ処理を実装
        # トークン化などの処理を行う
        pass

def setup_distributed():
    """分散学習環境のセットアップ"""
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NPROCS'])
    rank = int(os.environ['SLURM_PROCID'])
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, rank

def main():
    # 設定の読み込み
    config = Config()
    
    # 分散学習のセットアップ
    local_rank, world_size, rank = setup_distributed()
    
    # モデルの初期化
    model = Llama(config).cuda()
    
    # データセットの準備
    train_dataset = TextDataset(
        data_path=os.environ.get('TRAIN_DATA_PATH'),
        max_length=config.max_seq_len
    )
    
    # データローダーの設定
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # DeepSpeedの初期化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=config.ds_config
    )
    
    # 学習ループ
    for epoch in range(config.epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            
            outputs = model_engine(batch['input_ids'])
            loss = outputs[1]
            
            model_engine.backward(loss)
            model_engine.step()
            
            if step % config.log_interval == 0 and local_rank == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
        
        # チェックポイントの保存
        if local_rank == 0:
            model_engine.save_checkpoint(config.checkpoint_dir, f"epoch_{epoch}")

if __name__ == "__main__":
    main()