import pandas as pd
import numpy as np
import os

class TickDataProcessor:
    """
    處理 Tick 級別資料的處理器
    負責將 Tick 資料轉換為包含因果特徵 (Treatment) 的 K 線資料
    """
    def __init__(self, volume_threshold=500):
        # 定義大單干預的門檻，例如淨主動買賣量超過 500 口視為大單干預
        self.volume_threshold = volume_threshold

    def process_ticks_to_klines(self, tick_df: pd.DataFrame, timeframe='1min') -> pd.DataFrame:
        """
        將 Tick 資料轉換成 K 線，並計算淨主動買賣量與干預方向 (Treatment Direction)
        
        預期的 tick_df 欄位:
        - datetime: 時間戳記
        - price: 成交價
        - volume: 成交量
        - bid_price: 最佳買價 (選填，若無則用 price 變動推估)
        - ask_price: 最佳賣價 (選填，若無則用 price 變動推估)
        - tick_type: 'B' (主動買), 'S' (主動賣) (選填，若無則由 price 與 bid/ask 決定)
        """
        df = tick_df.copy()
        
        if 'datetime' not in df.columns:
            raise ValueError("Tick data 必須包含 'datetime' 欄位")
            
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # 決定主動買賣方向 (若無 tick_type 欄位)
        if 'tick_type' not in df.columns:
            df['tick_type'] = 'Unknown'
            if 'bid_price' in df.columns and 'ask_price' in df.columns:
                # 依據最佳買賣價判斷
                df.loc[df['price'] >= df['ask_price'], 'tick_type'] = 'B'
                df.loc[df['price'] <= df['bid_price'], 'tick_type'] = 'S'
            else:
                # 簡單推估：價格上漲為買，下跌為賣 (僅為替代方案)
                df['price_diff'] = df['price'].diff()
                df.loc[df['price_diff'] > 0, 'tick_type'] = 'B'
                df.loc[df['price_diff'] < 0, 'tick_type'] = 'S'
                # 平盤情況延續前一次的方向
                df['tick_type'] = df['tick_type'].replace('Unknown', np.nan).ffill().fillna('Unknown')
                
        # 計算主動買賣量
        df['aggress_buy_vol'] = np.where(df['tick_type'] == 'B', df['volume'], 0)
        df['aggress_sell_vol'] = np.where(df['tick_type'] == 'S', df['volume'], 0)
        
        # 聚合為 K 線
        # 定義聚合規則
        resample_rule = {
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum',
            'aggress_buy_vol': 'sum',
            'aggress_sell_vol': 'sum'
        }
        
        # 排除可能不存在的欄位
        kline_df = df.resample(timeframe).agg(resample_rule)
        
        # 重新命名欄位
        kline_df.columns = ['open', 'high', 'low', 'close', 'volume', 'aggress_buy_vol', 'aggress_sell_vol']
        kline_df.dropna(subset=['close'], inplace=True) # 移除無交易的 K 線
        
        # 計算淨主動買賣量 (Net Aggressive Volume)
        kline_df['net_aggressive_vol'] = kline_df['aggress_buy_vol'] - kline_df['aggress_sell_vol']
        
        # 計算干預變數 Treatment (T)
        # T=1 (大買單干預), T=-1 (大賣單干預), T=0 (無干預)
        conditions = [
            (kline_df['net_aggressive_vol'] >= self.volume_threshold),
            (kline_df['net_aggressive_vol'] <= -self.volume_threshold)
        ]
        choices = [1, -1]
        kline_df['treatment_dir'] = np.select(conditions, choices, default=0)
        
        return kline_df

if __name__ == "__main__":
    # 測試腳本
    print("Testing TickDataProcessor...")
    # 建立虛擬 Tick 資料
    np.random.seed(42)
    times = pd.date_range("2023-01-01 09:00:00", periods=1000, freq="1s")
    prices = np.random.normal(loc=15000, scale=10, size=1000)
    volumes = np.random.randint(1, 100, size=1000)
    tick_types = np.random.choice(['B', 'S'], size=1000)
    
    # 注入一些極端的大單以觸發 T=1 或 T=-1
    volumes[50:60] = 500  # 模擬 09:00 區間的大單
    tick_types[50:60] = 'B'
    
    volumes[150:160] = 500  # 模擬大賣單
    tick_types[150:160] = 'S'
    
    mock_tick_df = pd.DataFrame({
        "datetime": times,
        "price": prices,
        "volume": volumes,
        "tick_type": tick_types
    })
    
    processor = TickDataProcessor(volume_threshold=1000)
    kline_df = processor.process_ticks_to_klines(mock_tick_df, timeframe="1min")
    
    print("\n生成的 K 線特徵與因果變數 (前 5 筆):")
    print(kline_df[['close', 'volume', 'net_aggressive_vol', 'treatment_dir']].head())
    
    # 檢查是否有成功生成大單干預
    print("\n大買單干預次數:", (kline_df['treatment_dir'] == 1).sum())
    print("大賣單干預次數:", (kline_df['treatment_dir'] == -1).sum())
    print("無干預次數:", (kline_df['treatment_dir'] == 0).sum())
