import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
pd.options.mode.string_storage = 'python' # Disable pyarrow to prevent Shioaji thread crash
import pyarrow # Pre-load pyarrow to prevent access violation with Shioaji threads
import numpy as np
import os
import json
import time as time_lib
from data_engine import DayTradingDataEngine
from composite_ai import CompositeDayTradingAI
from model_manager import TradingModelManager

def train_trading_model(df_daily_chips_input=None):
    # 自動偵測 GPU，若失敗則回退 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  系統偵測運算設備: {device.type.upper()}")

    engine = DayTradingDataEngine()
    print("📥 下載台指期貨歷史 K 線進行 AI 訓練...")
    df_raw = engine.fetch_intraday_data(days=180)

    # 必須在這裡先定義 window_size，否則後面的檢查會報錯
    window_size = 40

    if df_raw is None or df_raw.empty:
        print("❌ 無法獲取期貨歷史 K 線數據")
        return

    # 1. 籌碼數據融合
    if df_daily_chips_input is not None:
        if hasattr(engine, 'integrate_institutional_chips'):
            df = engine.integrate_institutional_chips(df_raw, df_daily_chips_input)
            if df.isnull().values.any():
                print(f"⚠️ 警告：資料中尚有 {df.isnull().sum().sum()} 個空值，將自動填補為 0")
                df.fillna(0, inplace=True)

            print(f"DEBUG: df_raw 形狀 {df_raw.shape}")
            print(f"DEBUG: df 合併後形狀 {df.shape}")
            print(f"✅ 準備訓練！資料總列數: {len(df)}")
        else:
            df = df_raw
    else:
        print("⚠️ 未偵測到外部籌碼日誌，啟用基本結構適配中...")
        df = df_raw.copy()
        for col in ['foreign_net_oi', 'dealer_net_oi', 'foreign_oi_zscore', 'dealer_oi_zscore', 'foreign_oi_momentum', 'dealer_oi_momentum']:
            df[col] = 0.0
        df['pc_ratio'] = 1.0
        df['pc_ratio_momentum'] = 0.0

    exclude_cols = ['date', 'time', 'date_only', 'day_of_week', 'label', 'future_max', 'future_min', 'max_up_ret', 'max_down_ret']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df_feat = df[feature_cols].copy()

    # ==========================================
    # 🚀 突破口 A：非線性濾波 (Log Transform)
    # ==========================================
    log_features = ['mock_volume', 'macd_hist', 'vwap_bias']
    for col in log_features:
        if col in df_feat.columns:
            df_feat[col] = np.sign(df_feat[col]) * np.log1p(np.abs(df_feat[col]))

    # ==========================================
    # 🚀 突破口 B：穩健標準化 (Robust Scaling) - 防止未來函數洩漏 (Look-ahead Bias)
    # ==========================================
    # i. 確保只對數值欄位操作
    df_numeric = df_feat.select_dtypes(include=[np.number])
    
    # ii. 切分訓練集與驗證集 (時間序列必須按順序切分，不能隨機 shuffle)
    train_size = int(len(df_numeric) * 0.8)
    df_train = df_numeric.iloc[:train_size]
    
    # iii. 僅使用訓練集的統計量來計算正規化參數
    median = df_train.median()
    iqr = df_train.quantile(0.75) - df_train.quantile(0.25)
    iqr = iqr.replace(0, 1.0)
    
    # iv. 執行正規化 (全體資料皆使用訓練集的參數)
    df_normalized = (df_numeric - median) / iqr
    df_normalized = df_normalized.fillna(0)

    input_dim = df_normalized.shape[1]
    print(f"✅ 正規化完成，最終輸入特徵維度 (input_dim): {input_dim}")

    # 3. 建立標籤
    future_window = 5
    df['future_max'] = df['Close'].shift(-future_window).rolling(window=future_window).max()
    df['future_min'] = df['Close'].shift(-future_window).rolling(window=future_window).min()
    df['max_up_ret'] = (df['future_max'] - df['Close']) / df['Close']
    df['max_down_ret'] = (df['Close'] - df['future_min']) / df['Close']

    df.dropna(subset=['future_max', 'future_min'], inplace=True)
    df_normalized = df_normalized.iloc[:len(df)]

    T_L1 = 0.0010
    T_L2 = 0.0020
    T_L3 = 0.0035

    def classify_trend(row):
        up = row['max_up_ret']
        down = row['max_down_ret']
        
        # 0: Strong Down (-3), 1: Med Down (-2), 2: Weak Down (-1)
        # 3: Hold (0)
        # 4: Weak Up (1), 5: Med Up (2), 6: Strong Up (3)
        if up > down:
            if up > T_L3: return 6
            if up > T_L2: return 5
            if up > T_L1: return 4
            return 3
        else:
            if down > T_L3: return 0
            if down > T_L2: return 1
            if down > T_L1: return 2
            return 3

    df['label'] = df.apply(classify_trend, axis=1)

    # 4. 時序滑動視窗構建
    print(f"📊 檢查樣本數: {len(df_normalized)}")
    if len(df_normalized) < window_size:
        print(f"❌ 嚴重錯誤：資料處理後樣本數僅 {len(df_normalized)}，小於 window_size ({window_size})！請檢查資料來源與日期對齊。")
        return # 提前結束，避免除以零

    X, y = [], []

    data_values = df_normalized.values
    label_values = df['label'].values

    for i in range(window_size, len(data_values)):
        X.append(data_values[i-window_size:i])
        y.append(label_values[i])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    # X 形狀是 (Batch, Window, Features) = (N, 40, 25)
    # 不需在這裡轉置，因為 CompositeDayTradingAI 內部會自動處理 transpose

    y = torch.tensor(np.array(y), dtype=torch.long)

    # 驗證形狀是否正確
    print(f"DEBUG: X shape before model: {X.shape}") # 應該要是 (N, 40, 25)

    # 5. 模型初始化與訓練
    input_dim = X.shape[2]  # 特徵維度是在最後一個維度
    print(f"✅ 模型輸入維度自動對齊: {input_dim}")
    model = CompositeDayTradingAI(input_dim=input_dim, d_model=256, nhead=16, num_layers=4)
    model = model.to(device)

    # 優化：維持固定的初始動能，不再一開始就讓它溜滑梯
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
    epochs = 150

    # 突破 1：高原退火法 (只有當 Loss 連續 5 個 Epoch 降不下來時，才把學習率乘以 0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    counts = df['label'].value_counts().sort_index().values
    weights = 1.0 / (counts + 1e-8)
    weights = np.sqrt(weights)
    weights = weights / weights.sum() * 3.0
    alpha_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # ==========================================
    # 🚀 突破 2 & 3：實作 Focal Loss 函數
    # ==========================================
    def focal_loss(inputs, targets, alpha, gamma=2.0):
        ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        f_loss = ((1 - pt) ** gamma) * ce_loss
        return f_loss.mean()

    print(f"🚀 開始【Focal Loss 深度特徵版】分類訓練... 輸入維度: {input_dim}, 總樣本數: {len(X)}")
    
    # 優化：使用 DataLoader 處理批次，並加入 shuffle 防過擬合
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(X, y)
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type=='cuda'))

    # 優化：設定 Early Stopping 機制與 AMP 混合精度訓練以加速並防止過擬合
    best_loss = float('inf')
    early_stop_patience = 20
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    model.train()
    start_time = time_lib.time()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # 優化記憶體
            
            # 使用混合精度加速 (若有 GPU)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_X)
                    loss = focal_loss(outputs, batch_y, alpha=alpha_weights)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                loss = focal_loss(outputs, batch_y, alpha=alpha_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)

        # 讓 scheduler 根據這個 Epoch 的 Loss 來決定要不要降學習率
        scheduler.step(avg_epoch_loss)

        # Early Stopping 檢查
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            # 可以在此暫存最好的 model_state_dict，此處從簡只計算 best_loss
        else:
            patience_counter += 1

        if (epoch+1) % 10 == 0:
            elapsed = time_lib.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}], Focal Loss: {avg_epoch_loss:.4f} | LR: {current_lr:.6f} | 耗時: {elapsed:.2f} 秒")
            
        if patience_counter >= early_stop_patience:
            print(f"🛑 觸發 Early Stopping，訓練提早結束於 Epoch {epoch+1}，最佳 Loss: {best_loss:.4f}")
            break

    model = model.to('cpu')

    manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
    manager.save_model(model, optimizer, {"loss": avg_epoch_loss}, {"window_size": window_size, "epochs": epochs})

    norm_params = {"mean": median.to_dict(), "std": iqr.to_dict(), "feature_cols": feature_cols}
    with open(os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json"), "w", encoding='utf-8') as f:
        json.dump(norm_params, f)
    print(f"✅ Focal Loss 強化模型訓練完畢！")

if __name__ == "__main__":
    from data_engine import DayTradingDataEngine
    engine = DayTradingDataEngine()

    # 自動抓取過去 180 天的真實籌碼與買賣超！
    df_real_chips = engine.fetch_real_historical_chips(days=180)

    # 將真實籌碼傳入訓練引擎
    train_trading_model(df_daily_chips_input=df_real_chips)