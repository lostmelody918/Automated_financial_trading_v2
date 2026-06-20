import pandas as pd
import numpy as np

def extract_treatment_variable(df_tick, df_kline):
    """
    從 Tick 級別資料中萃取微觀結構干預變數 (OFI, Order Flow Imbalance)
    並與 K 線特徵融合。
    
    :param df_tick: 包含 datetime, price, volume, is_buyer_maker 的 DataFrame
    :param df_kline: 已計算好特徵的 1 分鐘 K 線 DataFrame，需包含 date 欄位
    :return: 融合 T 和 Treatment_Direction 的 DataFrame
    """
    df = df_tick.copy()
    
    # 確保 datetime 格式
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 判斷內外盤與計算主動買賣量
    # is_buyer_maker == False -> 外盤 (主動買)
    df['active_buy_vol'] = np.where(df['is_buyer_maker'] == False, df['volume'], 0)
    df['active_sell_vol'] = np.where(df['is_buyer_maker'] == True, df['volume'], 0)
    
    # 將 Tick 資料降頻對齊到 1 分鐘
    df.set_index('datetime', inplace=True)
    df_min = df.resample('1min').agg({
        'active_buy_vol': 'sum',
        'active_sell_vol': 'sum'
    }).dropna()
    
    # 計算每分鐘的「淨主動攻擊量」(OFI)
    df_min['net_active_vol'] = df_min['active_buy_vol'] - df_min['active_sell_vol']
    
    # 定義干預變數 (T)
    # 計算過去 60 期的滾動標準差作為動態門檻 (如果不足 60 期則用 expanding)
    df_min['rolling_std'] = df_min['net_active_vol'].rolling(window=60, min_periods=1).std()
    
    # 避免 std 為 0 導致除以零或無干預，設定一個最低門檻 (例如 10 口)
    threshold = df_min['rolling_std'].fillna(10) * 1.5 
    
    df_min['Treatment_T'] = np.where(df_min['net_active_vol'].abs() > threshold, 1.0, 0.0)
    
    df_min['Treatment_Direction'] = np.where(
        df_min['Treatment_T'] == 1.0,
        np.sign(df_min['net_active_vol']),
        0.0
    )
    
    df_min = df_min.reset_index()
    df_min.rename(columns={'datetime': 'date'}, inplace=True)
    
    # 資料融合：將結果 DataFrame 與現有的 1 分鐘 K 線特徵 DataFrame 進行 merge
    # 保留左邊 K 線的資料，若某些時間沒有 Tick 資料則填補 0
    df_merged = pd.merge(df_kline, df_min[['date', 'Treatment_T', 'Treatment_Direction']], on='date', how='left')
    df_merged['Treatment_T'] = df_merged['Treatment_T'].fillna(0.0)
    df_merged['Treatment_Direction'] = df_merged['Treatment_Direction'].fillna(0.0)
    
    return df_merged
