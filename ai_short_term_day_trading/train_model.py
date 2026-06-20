import os
import sys
from dotenv import load_dotenv
import time

# 強制控制台使用 UTF-8 輸出，防止 Emoji 造成 Windows CMD 崩潰
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import pyarrow # 必須在 torch 之前載入以防止 Windows C++ DLL Segfault

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import time as time_lib
import copy
import wandb
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from torch.utils.data import TensorDataset, DataLoader

# Pre-load/config for performance and stability
pd.options.mode.string_storage = 'python'
try:
    import pyarrow
except ImportError:
    pass

from data_engine import DayTradingDataEngine
from composite_ai import CausalMultiTimeframeAI
from model_manager import TradingModelManager

class TrainConfig:
    """Hyperparameters and configuration for training."""
    # Data parameters
    history_days: int = 730
    window_size_1m: int = 40
    window_size_15m: int = 20
    future_window: int = 10

    # Model architecture
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 3
    dropout: float = 0.3

    # Training hyperparameters
    lr: float = 0.0005
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 150
    early_stop_patience: int = 20

    # Environment
    num_workers: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    base_dir: Path = Path(__file__).parent
    model_dir: Path = base_dir / "saved_models"

def vectorized_triple_barrier(df, price_col='Close', vol_col='atr', t_horizon=10, tp_mult=2.0, sl_mult=1.5):
    """
    三重屏障法 (Triple Barrier Method) - 優化停損停利機制
    """
    future_max = df[price_col].rolling(window=t_horizon).max().shift(-t_horizon)
    future_min = df[price_col].rolling(window=t_horizon).min().shift(-t_horizon)

    p0 = df[price_col]
    atr = df[vol_col].replace(0, 10.0)

    hit_tp = future_max >= (p0 + tp_mult * atr)
    hit_sl = future_min <= (p0 - sl_mult * atr)
    final_p = df[price_col].shift(-t_horizon)

    labels = pd.Series(3, index=df.index)
    hit_both = hit_tp & hit_sl

    cond_strong_up = hit_tp & (~hit_both)
    cond_strong_down = hit_sl & (~hit_both)

    cond_med_up = (~hit_tp) & (~hit_sl) & (final_p > p0 + 0.5 * atr)
    cond_med_down = (~hit_tp) & (~hit_sl) & (final_p < p0 - 0.5 * atr)
    cond_weak_up = (~hit_tp) & (~hit_sl) & (final_p > p0) & (final_p <= p0 + 0.5 * atr)
    cond_weak_down = (~hit_tp) & (~hit_sl) & (final_p < p0) & (final_p >= p0 - 0.5 * atr)

    labels[cond_strong_up] = 6
    labels[cond_strong_down] = 0
    labels[cond_med_up] = 5
    labels[cond_med_down] = 1
    labels[cond_weak_up] = 4
    labels[cond_weak_down] = 2

    if 'date' in df.columns:
        is_same_day = df['date'].dt.date == df['date'].shift(-t_horizon).dt.date
        labels[~is_same_day] = np.nan
    else:
        labels.iloc[-t_horizon:] = np.nan

    labels[final_p.isna()] = np.nan
    return labels

def load_and_preprocess_data(config: TrainConfig, df_daily_chips_input: Optional[pd.DataFrame] = None):
    engine = DayTradingDataEngine()
    print(f"📥 下載台指期貨歷史 K 線進行 AI 訓練 (過去 {config.history_days} 天)...")
    df_raw = engine.fetch_intraday_data(days=config.history_days)

    if df_raw is None or df_raw.empty:
        raise ValueError("❌ 無法獲取期貨歷史 K 線數據")

    if df_daily_chips_input is not None:
        print("🔗 融合外部籌碼數據...")
        df = engine.integrate_institutional_chips(df_raw, df_daily_chips_input)
    else:
        print("⚠️ 未偵測到外部籌碼數據，啟用基本結構適配...")
        df = df_raw.copy()
        chip_cols = ['foreign_net_oi', 'dealer_net_oi', 'pc_ratio']
        for col in chip_cols:
            df[col] = 1.0 if col == 'pc_ratio' else 0.0

    # Labeling (三重屏障)
    print("🏷️ 產生三重屏障標籤...")
    df['label'] = vectorized_triple_barrier(df, t_horizon=config.future_window)
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)

    # 因果干預標籤 (Treatment)
    df['rolling_vol_std'] = df['mock_volume'].rolling(window=60, min_periods=1).std()
    threshold_vol = df['rolling_vol_std'].fillna(10) * 1.5
    df['Treatment_T'] = np.where(df['mock_volume'] > threshold_vol, 1.0, 0.0)
    df['Treatment_Direction'] = np.where(df['Treatment_T'] == 1.0, np.sign(df['Close'] - df['Open']), 0.0)

    # 選擇特徵
    absolute_cols = ['Open', 'High', 'Low', 'Close', 'vwap', 'bb_upper', 'bb_lower', 'Volume']
    intermediate_cols = ['Amount', 'mock_volume', 'vol_price', 'cum_vol_price', 'cum_vol', 'tr', 'atr', 'daily_open', 'yesterday_close', 'sma20', 'vwap_5', 'ma_20', 'recent_high_20', 'recent_low_20']
    constant_cols = ['foreign_net_oi', 'dealer_net_oi']
    exclude_cols = (['date', 'time', 'date_only', 'day_of_week', 'label', 'Treatment_T', 'Treatment_Direction', 'rolling_vol_std'] + absolute_cols + intermediate_cols + constant_cols)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    df_feat = df[feature_cols].copy()
    df_numeric = df_feat.select_dtypes(include=[np.number])

    # 移除常數特徵
    constant_mask = df_numeric.nunique() <= 1
    if constant_mask.any():
        dropped = constant_mask[constant_mask].index.tolist()
        df_numeric = df_numeric.drop(columns=dropped)
        feature_cols = [c for c in feature_cols if c in df_numeric.columns]

    print(f"🔍 最終特徵數量: {len(feature_cols)}")

    # 歸一化
    train_size = int(len(df_numeric) * 0.8)
    df_train_part = df_numeric.iloc[:train_size]
    median = df_train_part.median()
    iqr = df_train_part.quantile(0.75) - df_train_part.quantile(0.25)
    iqr = iqr.replace(0, 1.0)

    df_normalized = ((df_numeric - median) / iqr).fillna(0).clip(-10, 10).astype(np.float32)

    norm_params = {"mean": median.to_dict(), "std": iqr.to_dict(), "feature_cols": feature_cols}
    input_dim = df_normalized.shape[1]

    # 準備 MTF (Multi-Timeframe) 數據
    print("📊 準備多時間尺度 (MTF) 張量...")
    df_1m = df_normalized.copy()
    df_1m['date'] = df['date'].reset_index(drop=True)

    df_15m_historical = df_1m.set_index('date').resample('15min', closed='right', label='right').last().dropna()
    data_1m = df_1m.drop(columns='date').values
    data_15m_hist = df_15m_historical.values
    label_values = df['label'].values
    dir_values = df['Treatment_Direction'].values
    t_values = df['Treatment_T'].values

    indices_15m = df_15m_historical.index.get_indexer(df_1m['date'], method='pad')

    X_1m, X_15m, Y, T_dir, T_t = [], [], [], [], []
    win_1m, win_15m = config.window_size_1m, config.window_size_15m

    for i in range(win_1m - 1, len(df_1m)):
        idx_15m = indices_15m[i]
        if idx_15m >= win_15m - 1:
            X_1m.append(data_1m[i - win_1m + 1 : i + 1])
            hist_15m = data_15m_hist[idx_15m - (win_15m - 2) : idx_15m + 1]
            current_1m_state = data_1m[i].reshape(1, -1)

            if len(hist_15m) > 0:
                seq_15m = np.concatenate([hist_15m, current_1m_state], axis=0)
            else:
                seq_15m = np.tile(current_1m_state, (win_15m, 1))

            X_15m.append(seq_15m[-win_15m:])
            Y.append(label_values[i])
            T_dir.append(dir_values[i])
            T_t.append(t_values[i])

    X_1m = np.array(X_1m)
    X_15m = np.array(X_15m)
    Y = np.array(Y)
    T_dir = np.array(T_dir).reshape(-1, 1)
    T_t = np.array(T_t)

    # 序列劃分與 Gap 處理
    split_idx = int(len(X_1m) * 0.8)
    gap = win_1m

    X1_train, X15_train = torch.as_tensor(X_1m[:split_idx], dtype=torch.float32), torch.as_tensor(X_15m[:split_idx], dtype=torch.float32)
    Y_train = torch.as_tensor(Y[:split_idx], dtype=torch.long)
    T_dir_train, T_t_train = torch.as_tensor(T_dir[:split_idx], dtype=torch.float32), torch.as_tensor(T_t[:split_idx], dtype=torch.float32)

    X1_val, X15_val = torch.as_tensor(X_1m[split_idx + gap:], dtype=torch.float32), torch.as_tensor(X_15m[split_idx + gap:], dtype=torch.float32)
    Y_val = torch.as_tensor(Y[split_idx + gap:], dtype=torch.long)
    T_dir_val, T_t_val = torch.as_tensor(T_dir[split_idx + gap:], dtype=torch.float32), torch.as_tensor(T_t[split_idx + gap:], dtype=torch.float32)

    train_dataset = TensorDataset(X1_train, X15_train, Y_train, T_dir_train, T_t_train)
    val_dataset = TensorDataset(X1_val, X15_val, Y_val, T_dir_val, T_t_val)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 計算權重
    class_counts = np.bincount(Y[:split_idx], minlength=7).astype(float)
    weights = np.zeros(7)
    active_classes = class_counts > 0
    weights[active_classes] = 1.0 / np.sqrt(class_counts[active_classes])
    if weights.sum() > 0:
        weights = weights / weights.sum() * active_classes.sum()
    alpha_weights = torch.tensor(weights, dtype=torch.float32).to(config.device)

    return train_loader, val_loader, alpha_weights, input_dim, norm_params

def focal_loss(inputs, targets, alpha, gamma=1.0, label_smoothing=0.1):
    """Focal Loss with Label Smoothing for robust multi-class classification."""
    ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none', label_smoothing=label_smoothing)(inputs, targets)
    pt = torch.exp(-ce_loss)
    return (((1 - pt) ** gamma) * ce_loss)

def train_trading_model(df_daily_chips_input: Optional[pd.DataFrame] = None, engine=None):
    config = TrainConfig()
    if config.device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    try:
        # Load logic will assign input_dim, let's fix the scoping
        pass
    except Exception as e:
        print(e)

    # Actually call logic
    if engine is None:
        from data_engine import DayTradingDataEngine
        engine = DayTradingDataEngine()

    df_raw = engine.fetch_intraday_data(days=config.history_days)
    if df_daily_chips_input is not None:
        df = engine.integrate_institutional_chips(df_raw, df_daily_chips_input)
    else:
        df = df_raw.copy()
        for col in ['foreign_net_oi', 'dealer_net_oi', 'pc_ratio']:
            df[col] = 1.0 if col == 'pc_ratio' else 0.0

    df['label'] = vectorized_triple_barrier(df, t_horizon=config.future_window)
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)

    df['rolling_vol_std'] = df['mock_volume'].rolling(window=60, min_periods=1).std()
    threshold_vol = df['rolling_vol_std'].fillna(10) * 1.5
    df['Treatment_T'] = np.where(df['mock_volume'] > threshold_vol, 1.0, 0.0)
    df['Treatment_Direction'] = np.where(df['Treatment_T'] == 1.0, np.sign(df['Close'] - df['Open']), 0.0)

    exclude_cols = ['date', 'time', 'date_only', 'day_of_week', 'label', 'Treatment_T', 'Treatment_Direction', 'rolling_vol_std', 'Open', 'High', 'Low', 'Close', 'vwap', 'bb_upper', 'bb_lower', 'Volume', 'Amount', 'mock_volume', 'vol_price', 'cum_vol_price', 'cum_vol', 'tr', 'atr', 'daily_open', 'yesterday_close', 'sma20', 'vwap_5', 'ma_20', 'recent_high_20', 'recent_low_20', 'foreign_net_oi', 'dealer_net_oi', 'foreign_spot_net', 'dealer_spot_net']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    df_feat = df[feature_cols].copy()
    df_numeric = df_feat.select_dtypes(include=[np.number])
    constant_mask = df_numeric.nunique() <= 1
    if constant_mask.any():
        df_numeric = df_numeric.drop(columns=constant_mask[constant_mask].index.tolist())
        feature_cols = [c for c in feature_cols if c in df_numeric.columns]

    train_size = int(len(df_numeric) * 0.8)
    median = df_numeric.iloc[:train_size].median()
    iqr = (df_numeric.iloc[:train_size].quantile(0.75) - df_numeric.iloc[:train_size].quantile(0.25)).replace(0, 1.0)
    df_normalized = ((df_numeric - median) / iqr).fillna(0).clip(-10, 10).astype(np.float32)
    norm_params = {"mean": median.to_dict(), "std": iqr.to_dict(), "feature_cols": feature_cols}

    input_dim = df_normalized.shape[1]
    df_1m = df_normalized.copy()
    df_1m['date'] = df['date'].reset_index(drop=True)
    df_15m_historical = df_1m.set_index('date').resample('15min', closed='right', label='right').last().dropna()
    data_1m = df_1m.drop(columns='date').values
    data_15m_hist = df_15m_historical.values
    label_values = df['label'].values
    dir_values = df['Treatment_Direction'].values
    t_values = df['Treatment_T'].values
    indices_15m = df_15m_historical.index.get_indexer(df_1m['date'], method='pad')

    X_1m, X_15m, Y, T_dir, T_t = [], [], [], [], []
    win_1m, win_15m = config.window_size_1m, config.window_size_15m
    for i in range(win_1m - 1, len(df_1m)):
        idx_15m = indices_15m[i]
        if idx_15m >= win_15m - 1:
            X_1m.append(data_1m[i - win_1m + 1 : i + 1])
            hist_15m = data_15m_hist[idx_15m - (win_15m - 2) : idx_15m + 1]
            current_1m_state = data_1m[i].reshape(1, -1)
            seq_15m = np.concatenate([hist_15m, current_1m_state], axis=0) if len(hist_15m) > 0 else np.tile(current_1m_state, (win_15m, 1))
            X_15m.append(seq_15m[-win_15m:])
            Y.append(label_values[i])
            T_dir.append(dir_values[i])
            T_t.append(t_values[i])

    X_1m, X_15m, Y, T_dir, T_t = np.array(X_1m), np.array(X_15m), np.array(Y), np.array(T_dir).reshape(-1, 1), np.array(T_t)
    split_idx = int(len(X_1m) * 0.8)
    gap = win_1m
    X1_train = torch.as_tensor(X_1m[:split_idx], dtype=torch.float32)
    X15_train = torch.as_tensor(X_15m[:split_idx], dtype=torch.float32)
    Y_train = torch.as_tensor(Y[:split_idx], dtype=torch.long)
    T_dir_train = torch.as_tensor(T_dir[:split_idx], dtype=torch.float32)
    T_t_train = torch.as_tensor(T_t[:split_idx], dtype=torch.float32)

    X1_val = torch.as_tensor(X_1m[split_idx + gap:], dtype=torch.float32)
    X15_val = torch.as_tensor(X_15m[split_idx + gap:], dtype=torch.float32)
    Y_val = torch.as_tensor(Y[split_idx + gap:], dtype=torch.long)
    T_dir_val = torch.as_tensor(T_dir[split_idx + gap:], dtype=torch.float32)
    T_t_val = torch.as_tensor(T_t[split_idx + gap:], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X1_train, X15_train, Y_train, T_dir_train, T_t_train), batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X1_val, X15_val, Y_val, T_dir_val, T_t_val), batch_size=config.batch_size, shuffle=False)

    class_counts = np.bincount(Y[:split_idx], minlength=7).astype(float)
    weights = np.zeros(7)
    active_classes = class_counts > 0
    weights[active_classes] = 1.0 / np.sqrt(class_counts[active_classes])
    if weights.sum() > 0: weights = weights / weights.sum() * active_classes.sum()
    alpha_weights = torch.tensor(weights, dtype=torch.float32).to(config.device)

    model = CausalMultiTimeframeAI(
        input_dim=input_dim, d_model=config.d_model, nhead=config.nhead,
        num_layers=config.num_layers, dropout=config.dropout,
        seq_len_1m=config.window_size_1m, seq_len_15m=config.window_size_15m
    ).to(config.device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda') if config.device.type == 'cuda' else None

    print("🚀 開始訓練 Causal MTF 模型...")
    load_dotenv()
    wandb.init(
        project="finance_v3_day_trading",
        config={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_layers": config.num_layers,
            "dropout": config.dropout
        }
    )

    # 驗證階段使用標準 CE（不含 IPW），避免 propensity score 膨脹造成假性 val_loss 上升
    val_ce_criterion = nn.CrossEntropyLoss(weight=alpha_weights, label_smoothing=0.1)

    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0

        for b_X1, b_X15, b_y, b_Tdir, b_Tt in train_loader:
            b_X1, b_X15, b_y = b_X1.to(config.device), b_X15.to(config.device), b_y.to(config.device)
            b_Tdir, b_Tt = b_Tdir.to(config.device), b_Tt.to(config.device)

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    out0, out1, prop_logit = model(b_X1, b_X15, b_Tdir)

                    loss0 = focal_loss(out0, b_y, alpha_weights)
                    loss1 = focal_loss(out1, b_y, alpha_weights)
                    base_loss = torch.where(b_Tt == 0.0, loss0, loss1)

                    # IPW (Inverse Probability Weighting) - 僅訓練階段使用
                    ps = torch.sigmoid(prop_logit).squeeze(-1)
                    ipw_weights = torch.where(b_Tt == 1.0, 1.0 / (ps + 1e-4), 1.0 / (1.0 - ps + 1e-4))
                    ipw_weights = torch.clamp(ipw_weights, min=0.2, max=5.0)

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
                loss0 = focal_loss(out0, b_y, alpha_weights)
                loss1 = focal_loss(out1, b_y, alpha_weights)
                base_loss = torch.where(b_Tt == 0.0, loss0, loss1)

                ps = torch.sigmoid(prop_logit).squeeze(-1)
                ipw_weights = torch.where(b_Tt == 1.0, 1.0 / (ps + 1e-4), 1.0 / (1.0 - ps + 1e-4))
                ipw_weights = torch.clamp(ipw_weights, min=0.2, max=5.0)

                causal_loss = (base_loss * ipw_weights).mean()
                loss_propensity = F.binary_cross_entropy_with_logits(prop_logit.squeeze(-1), b_Tt)

                loss = causal_loss + 0.5 * loss_propensity
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            epoch_loss += loss.item()

            # 計算 Train Accuracy（取 factual 預測）
            with torch.no_grad():
                preds = torch.where(b_Tt.unsqueeze(-1).expand_as(out0) == 0.0, out0, out1).argmax(dim=-1)
                train_correct += (preds == b_y).sum().item()
                train_total += b_y.size(0)

        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # ===== 驗證階段：使用標準 CE（不含 IPW），精確衡量泛化能力 =====
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for b_X1, b_X15, b_y, b_Tdir, b_Tt in val_loader:
                b_X1, b_X15, b_y = b_X1.to(config.device), b_X15.to(config.device), b_y.to(config.device)
                b_Tdir, b_Tt = b_Tdir.to(config.device), b_Tt.to(config.device)
                out0, out1, prop_logit = model(b_X1, b_X15, b_Tdir)

                # 驗證 Loss：標準加權 CE，不含 IPW
                factual_logits = torch.where(b_Tt.unsqueeze(-1).expand_as(out0) == 0.0, out0, out1)
                loss = val_ce_criterion(factual_logits, b_y)
                val_loss += loss.item()

                # 計算 Val Accuracy
                preds = factual_logits.argmax(dim=-1)
                val_correct += (preds == b_y).sum().item()
                val_total += b_y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_duration_sec": epoch_duration
        })

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_duration:.2f}s")

        if patience_counter >= config.early_stop_patience:
            print(f"🛑 Early Stopping at Epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    model.eval()
    model = model.to('cpu')

    if not config.model_dir.exists():
        config.model_dir.mkdir(parents=True)

    manager = TradingModelManager(model_dir=str(config.model_dir))

    # 計算 Causal Metrics
    model.eval()
    with torch.no_grad():
        # 避免 OOM，只取最後 2000 筆資料來計算 Causal Metrics
        sample_size = min(2000, len(X1_val))
        out0, out1, _ = model(X1_val[-sample_size:], X15_val[-sample_size:], T_dir_val[-sample_size:])
        y0_pred = out0.argmax(dim=-1)
        y1_pred = out1.argmax(dim=-1)
        cate = (y1_pred - y0_pred).float().mean().item()
        factual_loss = best_loss
        counterfactual_var = out0.var(dim=0).mean().item()

    causal_metrics = {
        "ATE_estimation": cate,
        "factual_loss": factual_loss,
        "counterfactual_variance": counterfactual_var
    }

    manager.save_model(model, optimizer, {"loss": best_loss, "causal": causal_metrics}, {"window_size": config.window_size_1m, "window_size_15m": config.window_size_15m})

    norm_params_path = config.model_dir / "norm_params.json"
    with open(norm_params_path, "w", encoding='utf-8') as f:
        json.dump(norm_params, f, ensure_ascii=False, indent=4)

    print(f"✅ 訓練完畢！模型與參數已儲存於: {config.model_dir}")
    wandb.finish()

if __name__ == "__main__":
    engine = DayTradingDataEngine()
    df_real_chips = engine.fetch_real_historical_chips(days=730)
    train_trading_model(df_daily_chips_input=df_real_chips, engine=engine)
