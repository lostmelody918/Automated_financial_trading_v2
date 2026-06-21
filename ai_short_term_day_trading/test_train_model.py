import pyarrow
import torch
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from train_model import train_trading_model, TrainConfig, load_and_preprocess_data

def create_mock_df(size=200):
    """建立模擬的台指期 K 線數據"""
    dates = pd.date_range(start='2024-01-01', periods=size, freq='min')
    df = pd.DataFrame({
        'date': dates,
        'Open': np.random.uniform(18000, 19000, size),
        'High': np.random.uniform(18000, 19000, size),
        'Low': np.random.uniform(18000, 19000, size),
        'Close': np.random.uniform(18000, 19000, size),
        'Volume': np.random.randint(100, 1000, size),
        'mock_volume': np.random.randint(100, 1000, size),
        'macd_hist': np.random.normal(0, 10, size),
        'vwap_bias': np.random.normal(0, 0.01, size),
        'atr': np.random.normal(15, 2, size)
    })
    # 確保 High >= Low
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
    return df

def test_training_pipeline():
    print("🧪 開始測試訓練流水線...")
    
    # 設定測試配置
    class TestConfig(TrainConfig):
        epochs = 1
        batch_size = 16
        window_size_1m = 10
        window_size_15m = 5
        model_dir = Path("./test_saved_models")
    
    config = TestConfig()
    if config.model_dir.exists():
        shutil.rmtree(config.model_dir)
    config.model_dir.mkdir()

    # 模擬 DataEngine
    mock_df = create_mock_df(size=200)
    
    with patch('train_model.DayTradingDataEngine') as MockEngine:
        instance = MockEngine.return_value
        instance.fetch_intraday_data.return_value = mock_df
        
        # 1. 測試資料處理
        print("   - 測試資料預處理...")
        train_loader, val_loader, alpha_weights, input_dim, norm_params = load_and_preprocess_data(config)
        assert len(train_loader) > 0
        assert "mean" in norm_params
        print("   ✅ 資料處理成功")

        # 3. 測試完整訓練流程 (1 epoch)
        print("   - 測試完整訓練流程 (1 Epoch)...")
        with patch('train_model.TrainConfig', return_value=config):
            # Patch DataLoader to use our config sizes
            with patch('train_model.load_and_preprocess_data', return_value=(train_loader, val_loader, alpha_weights, input_dim, norm_params)):
                train_trading_model()
        
        # 4. 驗證產出物
        assert (config.model_dir / "norm_params.json").exists()
        # TradingModelManager 會存成 trading_model_v1.pth 之類的
        model_files = list(config.model_dir.glob("*.pth"))
        assert len(model_files) > 0
        print("   ✅ 訓練流程測試成功，產出物已驗證")

    # 清理測試目錄
    if config.model_dir.exists():
        shutil.rmtree(config.model_dir)
    print("\n🎉 所有測試通過！")

if __name__ == "__main__":
    try:
        test_training_pipeline()
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
