import os
import sys
import time
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import pyarrow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from train_model import TrainConfig, load_and_preprocess_data, focal_loss
from data_engine import DayTradingDataEngine
from composite_ai import CausalMultiTimeframeAI
from train_model import load_and_preprocess_data, focal_loss

# 全域變數用來快取資料，避免每個 Sweep 重新處理
cached_train_loader = None
cached_val_loader = None
cached_alpha_weights = None
cached_input_dim = None

def init_data():
    global cached_train_loader, cached_val_loader, cached_alpha_weights, cached_input_dim
    print("📥 [SWEEP] 正在載入並處理全域訓練資料 (僅執行一次)...")
    config = TrainConfig()
    engine = DayTradingDataEngine()
    df_real_chips = engine.fetch_real_historical_chips(days=730)
    
    # 這裡借用原本 train_model.py 中的 load_and_preprocess_data 
    cached_train_loader, cached_val_loader, cached_alpha_weights, cached_input_dim, _ = load_and_preprocess_data(config, df_real_chips)
    print("✅ 資料載入完成！")

def sweep_train():
    # WandB 會自動把配置注入
    run = wandb.init()
    config = wandb.config
    
    # 建立模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalMultiTimeframeAI(
        input_dim=cached_input_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
        seq_len_1m=40, # 預設 TrainConfig.window_size_1m
        seq_len_15m=20 # 預設 TrainConfig.window_size_15m
    ).to(device)
    
    # 設定優化器與學習率排程
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # 如果支援混合精度則啟用
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # 我們不變動 batch_size，如果需要變動，就要從 dataset 重新建立 loader
    # 為了簡化，這邊直接用緩存的 loader
    
    epochs = 30 # Sweep 通常不需要跑滿 150 epoch，可提早停止以節省時間
    
    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            model.train()
            epoch_loss = 0
            
            for b_X1, b_X15, b_y, b_Tdir, b_Tt in cached_train_loader:
                b_X1, b_X15, b_y = b_X1.to(device), b_X15.to(device), b_y.to(device)
                b_Tdir, b_Tt = b_Tdir.to(device), b_Tt.to(device)

                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        out0, out1, prop_logit = model(b_X1, b_X15, b_Tdir)
                        loss0 = focal_loss(out0, b_y, cached_alpha_weights)
                        loss1 = focal_loss(out1, b_y, cached_alpha_weights)
                        base_loss = torch.where(b_Tt == 0.0, loss0, loss1)
                        
                        ps = torch.sigmoid(prop_logit).squeeze(-1)
                        ipw_weights = torch.where(b_Tt == 1.0, 1.0 / (ps + 1e-4), 1.0 / (1.0 - ps + 1e-4))
                        ipw_weights = torch.clamp(ipw_weights, min=0.1, max=10.0)
                        causal_loss = (base_loss * ipw_weights).mean()
                        loss_propensity = F.binary_cross_entropy_with_logits(prop_logit.squeeze(-1), b_Tt)
                        
                        loss = causal_loss + 0.5 * loss_propensity
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out0, out1, prop_logit = model(b_X1, b_X15, b_Tdir)
                    loss0 = focal_loss(out0, b_y, cached_alpha_weights)
                    loss1 = focal_loss(out1, b_y, cached_alpha_weights)
                    base_loss = torch.where(b_Tt == 0.0, loss0, loss1)
                    
                    ps = torch.sigmoid(prop_logit).squeeze(-1)
                    ipw_weights = torch.where(b_Tt == 1.0, 1.0 / (ps + 1e-4), 1.0 / (1.0 - ps + 1e-4))
                    ipw_weights = torch.clamp(ipw_weights, min=0.1, max=10.0)
                    causal_loss = (base_loss * ipw_weights).mean()
                    loss_propensity = F.binary_cross_entropy_with_logits(prop_logit.squeeze(-1), b_Tt)
                    
                    loss = causal_loss + 0.5 * loss_propensity
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                epoch_loss += loss.item()
                
            avg_train_loss = epoch_loss / len(cached_train_loader)
            
            # 驗證階段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for b_X1, b_X15, b_y, b_Tdir, b_Tt in cached_val_loader:
                    b_X1, b_X15, b_y = b_X1.to(device), b_X15.to(device), b_y.to(device)
                    b_Tdir, b_Tt = b_Tdir.to(device), b_Tt.to(device)
                    out0, out1, prop_logit = model(b_X1, b_X15, b_Tdir)
                    
                    loss0 = focal_loss(out0, b_y, cached_alpha_weights)
                    loss1 = focal_loss(out1, b_y, cached_alpha_weights)
                    base_loss = torch.where(b_Tt == 0.0, loss0, loss1)
                    
                    ps = torch.sigmoid(prop_logit).squeeze(-1)
                    ipw_weights = torch.where(b_Tt == 1.0, 1.0 / (ps + 1e-4), 1.0 / (1.0 - ps + 1e-4))
                    ipw_weights = torch.clamp(ipw_weights, min=0.1, max=10.0)
                    causal_loss = (base_loss * ipw_weights).mean()
                    loss_propensity = F.binary_cross_entropy_with_logits(prop_logit.squeeze(-1), b_Tt)
                    loss = causal_loss + 0.5 * loss_propensity
                    
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(cached_val_loader)
            scheduler.step()
            
            epoch_duration = time.time() - epoch_start_time
            
            # 紀錄至 WandB
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch_duration": epoch_duration
            })
            print(f"[Sweep Run] Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    except Exception as e:
        print(f"Sweep iteration failed: {e}")
    finally:
        # 清理記憶體，防止連續 Sweep 跑到 OOM
        del model
        del optimizer
        del scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    load_dotenv()
    
    # 預先載入資料，避免每次嘗試都重新計算
    init_data()
    
    # 定義 Sweep 搜尋空間與策略
    sweep_config = {
        'method': 'bayes', # 貝葉斯優化
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'lr': {
                'min': 1e-5,
                'max': 1e-3
            },
            'dropout': {
                'min': 0.1,
                'max': 0.5
            },
            'd_model': {
                'values': [128, 256, 512]
            },
            'nhead': {
                'values': [4, 8]
            },
            'num_layers': {
                'values': [2, 3, 4]
            },
            'weight_decay': {
                'values': [1e-5, 1e-4, 1e-3]
            }
        }
    }
    
    print("🚀 啟動 WandB Sweep 超參數自動最佳化...")
    sweep_id = wandb.sweep(sweep_config, project="finance_v3_day_trading_sweep")
    
    # 開始執行 Agent，設定 count=3 作為測試 (正式優化可改為更高的次數)
    wandb.agent(sweep_id, function=sweep_train, count=3)
