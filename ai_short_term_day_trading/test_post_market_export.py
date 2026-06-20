import os
import shutil
import pytest
import pandas as pd
pd.options.mode.string_storage = 'python' # Disable pyarrow to prevent Shioaji thread crash
import pyarrow # Pre-load pyarrow to prevent access violation
import numpy as np
from datetime import datetime, timedelta
from post_market_export import PostMarketExporter

# ==========================================
# Mock Data 生成區
# ==========================================
def generate_mock_market_data():
    """生成包含日盤與極端狀態 (NaN, 空值) 的模擬主市場資料"""
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day, 8, 45)
    
    dates = [today_start + timedelta(minutes=i) for i in range(300)]
    # 加入未排序與重複，測試 sanitize 功能
    dates.append(today_start + timedelta(minutes=150)) 
    
    close_prices = np.linspace(20000, 20200, len(dates))
    close_prices += np.random.normal(0, 10, len(dates)) # 加點雜訊
    
    # 製造極端資料 (NaN, Inf)
    close_prices[10] = np.nan
    close_prices[20] = np.inf
    
    df = pd.DataFrame({
        'date': dates,
        'Close': close_prices,
        'Volume': np.random.randint(1, 100, len(dates)),
        'mock_feature_1': np.random.random(len(dates))
    })
    return df

def generate_mock_options_history():
    """生成夜盤跨日盤的模擬選擇權資料"""
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    night_start = datetime(yesterday.year, yesterday.month, yesterday.day, 15, 0)
    
    dates = [night_start + timedelta(minutes=i*5) for i in range(150)] # 夜盤
    day_start = datetime(now.year, now.month, now.day, 8, 45)
    dates += [day_start + timedelta(minutes=i*5) for i in range(60)] # 日盤
    
    df_opt = pd.DataFrame({
        'date': dates,
        'Close': np.linspace(100, 200, len(dates)),
        'Bid': np.linspace(99, 199, len(dates)),
        'Ask': np.linspace(101, 201, len(dates)),
        'Volume': np.random.randint(10, 500, len(dates))
    })
    
    return {'TXO20000C': df_opt}

def generate_mock_trade_log():
    """生成單筆交易區間詳細紀錄"""
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day, 9, 0)
    
    trade1 = {
        'symbol': 'TXO20000C',
        'strategy_label': 'AI_Breakout_V1',
        'direction': 1,
        'entry_time': (today_start + timedelta(minutes=10)).isoformat(),
        'entry_price': 20050.0,
        'exit_time': (today_start + timedelta(minutes=60)).isoformat(),
        'exit_price': 20100.0,
        'hard_tp_price': 20150.0,
        'hard_sl_price': 20000.0,
        'pnl': 50 * 50,
        'feat_vol_surge': 2.5
    }
    
    # 一筆沒有 exit_time 的未平倉或邊角情況
    trade2 = {
        'symbol': 'TXO20200P',
        'strategy_label': 'AI_Reversal_V2',
        'direction': -1,
        'entry_time': (today_start + timedelta(minutes=120)).isoformat(),
        'entry_price': 20120.0,
        'exit_time': None,
        'exit_price': 0,
        'hard_tp_price': 20050.0,
        'hard_sl_price': 20200.0,
        'pnl': -10 * 50,
        'feat_vol_surge': 1.1
    }
    return [trade1, trade2]

# ==========================================
# Pytest 測試案例
# ==========================================

@pytest.fixture
def temp_export_dir():
    dir_path = "temp_pytest_reports"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def test_exporter_full_pipeline(temp_export_dir):
    """驗證: Mock Data 能否成功跑通雙版本匯出且無 Runtime Error"""
    exporter = PostMarketExporter(export_dir=temp_export_dir)
    
    df_market = generate_mock_market_data()
    dict_opt = generate_mock_options_history()
    trade_log = generate_mock_trade_log()
    
    # 執行主匯出邏輯
    exporter.execute_export(df_market, dict_opt, trade_log)
    
    # 驗證 CLI 版本輸出
    files = os.listdir(temp_export_dir)
    assert any("cli_market_data_" in f for f in files), "缺失 CLI market_data"
    assert any("cli_options_data_" in f for f in files), "缺失 CLI options_data"
    assert any("cli_trade_log_" in f for f in files), "缺失 CLI trade_log"
    
    # 驗證 Human 版本輸出
    assert any("human_report_" in f and f.endswith('.md') for f in files), "缺失 Human report MD"
    assert any("human_chart_" in f and f.endswith('.png') for f in files), "缺失 Human chart PNG"

def test_exporter_edge_cases(temp_export_dir):
    """驗證: 極端防呆機制 (空資料、全 NaN)"""
    exporter = PostMarketExporter(export_dir=temp_export_dir)
    
    # 傳入空的 DF 與空的 log
    df_empty = pd.DataFrame()
    dict_empty = {}
    log_empty = []
    
    try:
        exporter.execute_export(df_empty, dict_empty, log_empty)
    except Exception as e:
        pytest.fail(f"邊角情況 (Empty Data) 觸發了崩潰: {e}")
        
    # 傳入完全無時間的 DF
    df_no_time = pd.DataFrame({'Close': [100, 200], 'Volume': [1, 2]})
    try:
        exporter.execute_export(df_no_time, dict_empty, log_empty)
    except Exception as e:
        pytest.fail(f"邊角情況 (No Time Column) 觸發了崩潰: {e}")
