import os
import sys

# 強制控制台使用 UTF-8 輸出，防止 Emoji 造成 Windows CMD 崩潰
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import requests
import io
import time
import shioaji as sj
from datetime import datetime, timedelta
from dotenv import load_dotenv

class DayTradingDataEngine:
    def __init__(self, symbol="2330"):
        self.symbol = symbol
        self.api = sj.Shioaji()

        env_path = r"F:\Gemini_CLI_Application\finance_v3\.env"
        load_dotenv(dotenv_path=env_path)

        api_key = os.environ.get('SHIOAJI_API_KEY', '')
        secret_key = os.environ.get('SHIOAJI_SECRET_KEY', '')

        if api_key and secret_key:
            print("🔐 正在登入 Shioaji API 並下載合約檔...")
            try:
                self.api.login(api_key, secret_key, contracts_timeout=10000)
                print("✅ Shioaji 登入成功！合約清單已載入。")
            except Exception as e:
                print(f"❌ Shioaji 登入失敗: {e}")
                raise SystemExit("終止程式：無法連線至券商伺服器。")
        else:
            raise SystemExit("終止程式：缺少 API 金鑰。")

    def fetch_intraday_data(self, days=60):
        """
        強制日期對齊與高維度特徵提取引擎
        抓取台指期近月合約，並計算 AI 訓練所需的 29 個技術指標
        """
        # 計算動態日期區間 (確保結束日期為 today)
        today = datetime.now()
        target_start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # 為了確保均線與昨日收盤價計算準確 (跨週末與長假)，往前多抓 15 天作為緩衝
        fetch_start_date = (today - timedelta(days=days + 15)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        print(f"📡 [DATA] 正在抓取數據範圍: {fetch_start_date} 至 {end_date} (含緩衝期)")

        try:
            print("[DEBUG] Step 1: Getting contract")
            contract = self.api.Contracts.Futures.TXF.TXFR1
            if contract is None:
                print("❌ 無法獲取近月合約 TXFR1，請檢查 Shioaji 登入狀態")
                return pd.DataFrame()

            print("[DEBUG] Step 2: Checking cache file path")
            cache_file = os.path.join(os.path.dirname(__file__), "market_data_cache_v3.parquet")
            df_cache = pd.DataFrame()
            last_date = None
            first_date = None

            if os.path.exists(cache_file):
                print("[DEBUG] Step 3: Cache file exists, reading parquet")
                try:
                    df_cache = pd.read_parquet(cache_file)
                    print(f"[DEBUG] Step 4: Parquet read successful, shape: {df_cache.shape}")
                    if not df_cache.empty:
                        if 'date' in df_cache.columns:
                            df_cache.rename(columns={'date': 'ts', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Amount': 'amount'}, inplace=True)
                        print("[DEBUG] Step 5: Renamed columns")
                        df_cache['ts'] = pd.to_datetime(df_cache['ts'])
                        print("[DEBUG] Step 6: Converted ts to datetime")
                        last_date = df_cache['ts'].max()
                        first_date = df_cache['ts'].min()
                        print(f"📦 載入本地行情快取，日期範圍: {first_date.date()} ~ {last_date.date()} ({len(df_cache)} 筆)")
                except Exception as e:
                    print(f"⚠️ 快取讀取失敗: {e}")

            print("[DEBUG] Step 7: Converting target_start and end_dt")
            target_start = pd.to_datetime(fetch_start_date)
            end_dt = pd.to_datetime(end_date)
            
            print("[DEBUG] Step 8: Computing needs_future and needs_past")
            needs_future = not last_date or last_date.date() < today.date()
            needs_past = not first_date or first_date > target_start
            
            print(f"[DEBUG] needs_future: {needs_future}, needs_past: {needs_past}")
            if not needs_future and not needs_past:
                print("✅ 數據已充足且是最新，無需抓取。")
                all_frames = [df_cache]
            else:
                all_frames = [df_cache] if not df_cache.empty else []

                # 補過去缺口
                if needs_past:
                    fetch_past_start = target_start
                    fetch_past_end = first_date if first_date else end_dt
                    print(f"📡 [DATA] 偵測到過去數據缺口，開始分段補抓 ({fetch_past_start.date()} ~ {fetch_past_end.date()})...")
                    curr_start = fetch_past_start
                    while curr_start < fetch_past_end:
                        curr_end = min(curr_start + timedelta(days=7), fetch_past_end)
                        print(f"   📥 抓取區間(過去): {curr_start.strftime('%Y-%m-%d')} -> {curr_end.strftime('%Y-%m-%d')}...")
                        for retry in range(3):
                            try:
                                print(f"[DEBUG] Step 9: Fetching kbars for {curr_start} to {curr_end}")
                                kbars = self.api.kbars(contract, start=curr_start.strftime("%Y-%m-%d"), end=curr_end.strftime("%Y-%m-%d"))
                                print(f"[DEBUG] Step 10: Fetched kbars, constructing DataFrame")
                                if kbars and len(kbars.ts) > 0:
                                    df_chunk = pd.DataFrame({**kbars})
                                    df_chunk['ts'] = pd.to_datetime(df_chunk['ts'])
                                    df_chunk.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Amount': 'amount'}, inplace=True)
                                    all_frames.append(df_chunk)
                                    break
                                else:
                                    print("      ℹ️ 此區間無資料")
                                    break
                            except Exception as e:
                                wait_time = (retry + 1) * 5
                                print(f"      ❌ 抓取異常 (嘗試 {retry+1}/3): {e}，等待 {wait_time} 秒後重試...")
                                time.sleep(wait_time)
                        curr_start = curr_end + timedelta(days=1)
                        time.sleep(1.2) # 安全間隔

                # 補未來缺口
                if needs_future:
                    fetch_future_start = last_date + timedelta(minutes=1) if last_date else target_start
                    print(f"📡 [DATA] 偵測到未來數據缺口，開始分段補抓 ({fetch_future_start.date()} ~ {today.date()})...")
                    curr_start = fetch_future_start
                    while curr_start < today:
                        curr_end = min(curr_start + timedelta(days=7), today)
                        print(f"   📥 抓取區間(未來): {curr_start.strftime('%Y-%m-%d')} -> {curr_end.strftime('%Y-%m-%d')}...")
                        for retry in range(3):
                            try:
                                kbars = self.api.kbars(contract, start=curr_start.strftime("%Y-%m-%d"), end=curr_end.strftime("%Y-%m-%d"))
                                if kbars and len(kbars.ts) > 0:
                                    df_chunk = pd.DataFrame({**kbars})
                                    df_chunk['ts'] = pd.to_datetime(df_chunk['ts'])
                                    df_chunk.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Amount': 'amount'}, inplace=True)
                                    all_frames.append(df_chunk)
                                    break
                                else:
                                    print("      ℹ️ 此區間無資料")
                                    break
                            except Exception as e:
                                wait_time = (retry + 1) * 5
                                print(f"      ❌ 抓取異常 (嘗試 {retry+1}/3): {e}，等待 {wait_time} 秒後重試...")
                                time.sleep(wait_time)
                        curr_start = curr_end + timedelta(days=1)
                        time.sleep(1.2)

                if all_frames:
                    df_to_cache = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=['ts']).sort_values('ts').reset_index(drop=True)
                    print(f"📊 歷史數據合併完成，總計 {len(df_to_cache)} 根 K 線")
                    # 儲存基本欄位到快取
                    try:
                        df_to_cache.to_parquet(cache_file, compression='snappy')
                    except Exception as cache_err:
                        print(f"⚠️ 快取儲存失敗: {cache_err}")

            if not all_frames:
                print("❌ 警告：回傳數據為空，請檢查網路或 API 是否有資料")
                return pd.DataFrame()
                
            df = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=['ts']).sort_values('ts').reset_index(drop=True)
    
            # 3. 基礎日期與索引清理
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.sort_values('ts').reset_index(drop=True) # 強制時間排序
            df.rename(columns={'ts': 'date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'amount': 'Amount'}, inplace=True)
            # 檢查列名是否已經改了
            if 'date' not in df.columns:
                print(f"DEBUG: 重新命名失敗，目前的欄位: {df.columns.tolist()}")
            df['time'] = df['date'].dt.time
            df['date_only'] = df['date'].dt.date
            df['day_of_week'] = df['date'].dt.dayofweek

            # 4. 特徵工程 (Feature Engineering)
            # 價格變動
            df['ret'] = df['Close'].pct_change()

           # 價量與流動性指標：先計算好單列乘積，再 Groupby
            df['mock_volume'] = df['Volume'].replace(0, 1)
            df['vol_price'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['mock_volume']

            # 使用更穩定的 sum/cumsum 寫法
            df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
            df['cum_vol'] = df.groupby('date_only')['mock_volume'].cumsum()
            df['vwap'] = df['cum_vol_price'] / df['cum_vol']
            df['vwap_bias'] = (df['Close'] - df['vwap']) / (df['vwap'] + 1e-9)

            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['signal']

            # ATR (波動率)
            df['h_l'] = df['High'] - df['Low']
            df['h_pc'] = abs(df['High'] - df['Close'].shift(1))
            df['l_pc'] = abs(df['Low'] - df['Close'].shift(1))
            df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            df['rsi'] = 100 - (100 / (1 + rs))

            # 微觀反轉特徵 (Micro-reversal)
            # Fast RSI (e.g., period=3)
            gain_fast = (delta.where(delta > 0, 0)).rolling(window=3).mean()
            loss_fast = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
            rs_fast = gain_fast / (loss_fast + 1e-9)
            df['rsi_fast'] = 100 - (100 / (1 + rs_fast))
            
            # K線實體與影線解析
            df['body_length'] = abs(df['Close'] - df['Open'])
            df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            
            # 動能爆發 (當前K線實體長度是否大於過去5根平均)
            df['body_avg_5'] = df['body_length'].rolling(5).mean()
            df['momentum_explosion'] = (df['body_length'] > df['body_avg_5']).astype(int)

            # 日級別格局 (Daily Context) 與 大盤現貨相對強弱
            # 每日開盤價
            df['daily_open'] = df.groupby('date_only')['Open'].transform('first')
            
            # 昨日收盤價 (先算出每天最後一筆，再平移，再 map 回去)
            daily_close = df.groupby('date_only')['Close'].last().shift(1)
            df['yesterday_close'] = df['date_only'].map(daily_close)
            
            # 若為第一天無昨日收盤價，使用當日開盤價替代 (Gap = 0)
            df['yesterday_close'] = df['yesterday_close'].fillna(df['daily_open'])
            
            # 跳空缺口幅度
            df['gap_amplitude'] = (df['daily_open'] - df['yesterday_close']) / (df['yesterday_close'] + 1e-9)
            
            # 日內絕對趨勢 (當前價格相對於當日開盤的漲跌幅，衡量真實K線實體)
            df['intraday_trend'] = (df['Close'] - df['daily_open']) / (df['daily_open'] + 1e-9)
            
            # 均值回歸與突破輔助特徵 (Mean Reversion / Breakout Aux)
            # 1. 乖離均線 (Distance from MA) - 判斷拉回深度
            df['dist_from_ma20'] = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).mean() + 1e-9)
            
            # 2. 區間高低點回撤 (Pullback Depth) - 判斷目前是創高還是拉回
            df['recent_high_20'] = df['High'].rolling(20).max()
            df['recent_low_20'] = df['Low'].rolling(20).min()
            df['pullback_from_high'] = (df['Close'] - df['recent_high_20']) / (df['recent_high_20'] + 1e-9)
            df['bounce_from_low'] = (df['Close'] - df['recent_low_20']) / (df['recent_low_20'] + 1e-9)
            
            # 缺口回補狀態 (1: 已回補, 0: 未回補)
            df['gap_filled'] = 0
            df.loc[(df['gap_amplitude'] > 0) & (df['Close'] <= df['yesterday_close']), 'gap_filled'] = 1
            df.loc[(df['gap_amplitude'] < 0) & (df['Close'] >= df['yesterday_close']), 'gap_filled'] = 1
            
            # 簡單期現貨基差趨勢代理 (如果沒有現貨，用 VWAP 與 MA 的相對強弱代替)
            df['spot_futures_proxy'] = (df['Close'] - df['vwap']) / (df['Close'].rolling(20).mean() + 1e-9)

            # Bollinger Bands
            df['sma20'] = df['Close'].rolling(window=20).mean()
            df['std20'] = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
            df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['sma20'] + 1e-9)
            df['is_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)

            # 均線斜率 (Trend Slope) - 衡量趨勢的「速度」
            # 計算 5 分鐘 VWAP 或 MA 的斜率 (使用 3 根 K 棒的變化率並放大)
            df['vwap_5'] = df['mock_volume'] * df['Close'] / df['mock_volume'].rolling(5).sum()
            df['slope_vwap'] = (df['vwap_5'] - df['vwap_5'].shift(3)) / df['vwap_5'].shift(3) * 10000

            # 計算 20MA 斜率 (判斷中期趨勢方向)
            df['ma_20'] = df['Close'].rolling(20).mean()
            df['slope_ma20'] = (df['ma_20'] - df['ma_20'].shift(3)) / df['ma_20'].shift(3) * 10000


            # 價量結構 (Price-Volume Dynamics) - 衡量「爆發力」
            # 爆量倍數：當下成交量是過去 20 根均量的幾倍？(突破 2 倍以上才有意義)
            df['vol_surge_ratio'] = df['mock_volume'] / (df['mock_volume'].rolling(20).mean() + 1e-9)

            # 價量背離背離指標 (Price-Volume Divergence)
            # 如果價格創新高，但成交量萎縮，這通常是誘多 (假突破)
            df['price_roc'] = df['Close'].pct_change()
            df['pv_divergence'] = np.where((df['price_roc'] > 0) & (df['vol_surge_ratio'] < 0.8), -1,
                                np.where((df['price_roc'] < 0) & (df['vol_surge_ratio'] < 0.8), 1, 0))


            df['close_frac_diff'] = df['Close'].pct_change().fillna(0)  # 報酬率 (非絕對差值)
            df['trend_wavelet'] = df['Close'].pct_change(5).fillna(0)   # 5 根 K 線的累積報酬率
            df['noise_wavelet'] = (df['Close'].pct_change() - df['Close'].pct_change().rolling(5).mean()).fillna(0)  # 去趨勢後的噪聲

            # --------------------------------------------------
            # 🕰️ 週期性時間編碼 (Cyclical Time Encoding)
            # --------------------------------------------------
            minutes_of_day = df['date'].dt.hour * 60 + df['date'].dt.minute
            df['time_sin'] = np.sin(2 * np.pi * minutes_of_day / 1440.0)
            df['time_cos'] = np.cos(2 * np.pi * minutes_of_day / 1440.0)

            # --------------------------------------------------
            # 🌍 整合美股與日股全域特徵 (Global Features)
            # --------------------------------------------------
            # 這裡簡化實作，若您有 fetch_global_indices 的實作，可解除註解
            # global_df = self.fetch_global_indices()
            # if not global_df.empty: ...
            # 此處提供一個防禦性填充，以防沒有該功能
            df['nasdaq_prev_ret'] = 0.0
            df['nikkei_premarket_momentum'] = 0.0
            df['us_tw_gap_divergence'] = df['gap_amplitude'] - df['nasdaq_prev_ret']

            # 半週期餘弦衰減
            minutes_from_0845 = minutes_of_day - 525
            is_day_session = (minutes_of_day >= 525) & (minutes_of_day <= 825)
            cosine_decay = np.where(
                (minutes_from_0845 >= 0) & (minutes_from_0845 <= 60),
                np.cos((minutes_from_0845 / 60.0) * (np.pi / 2.0)),
                0.0
            )
            final_weight = cosine_decay * is_day_session
            df['us_tw_gap_divergence'] = df['us_tw_gap_divergence'] * final_weight
            df['nikkei_premarket_momentum'] = df['nikkei_premarket_momentum'] * final_weight

            # 5. 清理與輸出
            # 刪除輔助計算的欄位，保留乾淨的 DataFrame
            cols_to_drop = ['vol_price', 'cum_vol_price', 'cum_vol', 'h_l', 'h_pc', 'l_pc', 'tr', 'sma20', 'std20', 'Amount']
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

            print(f"✅ 數據更新完成。最新時間: {df['date'].iloc[-1]}, 總筆數: {len(df)}")
            return df.dropna().reset_index(drop=True)

        except Exception as e:
            print(f"❌ 數據抓取異常: {e}")
            return pd.DataFrame()

    def get_best_volume_option_contract(self, option_type='Call', allocated_capital=100000):
        """🚀 依據最短到期日與最大成交量及資金篩選合約 (優先選週選)"""
        try:
            txf_contract = self.api.Contracts.Futures.TXF.TXFR1
            snap = self.api.snapshots([txf_contract])[0]
            current_index = snap.close

            all_options = []
            for cat in ['TXO', 'TX1', 'TX2', 'TX4', 'TX5', 'TXU', 'TXV', 'TXX']:
                if hasattr(self.api.Contracts.Options, cat):
                    all_options.extend([c for c in getattr(self.api.Contracts.Options, cat)])

            if not all_options: return None

            # 抓取今天日期以計算真實剩餘天數 (使用 Shioaji API 格式通常是 YYYYMM 或 YYYYMM(W))
            # 這裡我們利用 get_api_based_dte 或是簡單依賴合約的 delivery_month 字串長度/排序
            # 但更精確的做法是計算到期日。由於這裡沒有 DTE 函數，我們透過字串排序來優先選週選。
            # 通常 weekly 是 202606W1, 202606W2, 或是 TX1, TX2, TX4, TX5
            # Shioaji 裡面的 delivery_month 格式例如 "202606" 或 "202606W1" (有些是 202606F1)
            # 為了確保抓到最近的，我們先過濾掉過期的，再以 "最近" 為主

            # Shioaji 的期權合約物件通常有 delivery_date 屬性，格式如 '2026/06/03'
            # 我們直接使用 delivery_date 排序找出最近到期的合約群
            valid_contracts_with_date = []
            today_str = datetime.now().strftime('%Y/%m/%d')

            for c in all_options:
                if str(c.option_right).endswith(option_type):
                    delivery_date = getattr(c, 'delivery_date', None)
                    if delivery_date and delivery_date >= today_str:
                        valid_contracts_with_date.append(c)

            if not valid_contracts_with_date:
                return None

            # 依據 delivery_date 排序，找出最近的到期日
            valid_contracts_with_date.sort(key=lambda x: x.delivery_date)
            nearest_date = valid_contracts_with_date[0].delivery_date

            # 取出所有這個最近到期日的合約 (可能是週選或剛好是月選結算日)
            near_contracts = [c for c in valid_contracts_with_date if c.delivery_date == nearest_date]

            # 依照履約價與目前指數的距離排序
            near_contracts.sort(key=lambda x: abs(x.strike_price - current_index))
            target_contracts = near_contracts[:30] # 擴大範圍至上下15檔

            if not target_contracts:
                return None

            snaps = self.api.snapshots(target_contracts)

            valid_contracts = []
            for i, snap in enumerate(snaps):
                price = getattr(snap, 'close', 0)
                if price == 0 and hasattr(snap, 'buy_price'): price = getattr(snap, 'buy_price', 0)
                volume = getattr(snap, 'total_volume', 0)
                # 篩選條件：權利金必須 >= 5 點且買得起 (選擇權一點 50 元)
                if price >= 5 and (price * 50) <= allocated_capital:
                    valid_contracts.append({
                        'contract': target_contracts[i],
                        'price': price,
                        'volume': volume
                    })

            if not valid_contracts:
                target_contracts.sort(key=lambda x: abs(x.strike_price - current_index), reverse=True)
                return target_contracts[0]

            # 在這些最近到期的合約中，依據成交量排序取最大者
            valid_contracts.sort(key=lambda x: x['volume'], reverse=True)
            return valid_contracts[0]['contract']

        except Exception as e:
            print(f"❌ 成交量合約定位失敗: {e}")
            return self.api.Contracts.Futures.TXF.TXFR1

    def fetch_real_historical_chips(self, days=180):
        """
        🚀 自動向期交所與證交所抓取真實歷史籌碼 (完整健壯版)
        - 針對現貨買賣超：整合 FinMind API 快速獲取 180 天資料，避開迴圈延遲
        - 針對期交所：增加容錯，若遇假日無資料則略過
        """
        import io
        import requests
        import time
        import os
        from datetime import datetime, timedelta
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        cache_file = os.path.join(os.path.dirname(__file__), "historical_chips_cache_v3.parquet")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        if os.path.exists(cache_file):
            try:
                df_cache = pd.read_parquet(cache_file)
                # 簡單判定：如果最後一筆資料是最近兩天內，就直接用快取
                if pd.to_datetime(df_cache['date']).max().date() >= (end_date - timedelta(days=2)).date():
                    print(f"\n📦 載入本地籌碼快取，有效樣本數: {len(df_cache)}")
                    return df_cache
            except Exception as e:
                print(f"⚠️ 籌碼快取讀取失敗: {e}")

        print(f"\n📡 開始抓取近 {days} 天籌碼...")

        session = requests.Session()
        # 設定 Retry 機制：最多重試 5 次，且遇到 403, 500 等錯誤都會退避重試
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # 模擬真實瀏覽器 Header 以突破防爬蟲限制
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/csv,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Origin': 'https://www.taifex.com.tw',
            'Referer': 'https://www.taifex.com.tw/cht/3/futContractsDate'
        }

        # 1. 自動分段邏輯 (防止 API 下載區間限制，期交所一次最多可抓 30 天)
        date_chunks = []
        curr = start_date
        while curr < end_date:
            next_end = min(curr + timedelta(days=30), end_date)
            date_chunks.append((curr, next_end))
            curr = next_end + timedelta(days=1)

        # 2. 抓取 P/C Ratio 與 期貨 OI
        pc_frames, oi_frames = [], []
        year_offset = 0
        if end_date.year > 2024:
            year_offset = end_date.year - 2024

        for s_date, e_date in date_chunks:
            # 針對未來時間模擬，執行年份平移以抓取真實歷史資料並映射回未來
            query_s_dt = s_date.replace(year=s_date.year - year_offset).strftime("%Y/%m/%d")
            query_e_dt = e_date.replace(year=e_date.year - year_offset).strftime("%Y/%m/%d")
            print(f"  📥 抓取區間 (期交所+P/C): {query_s_dt} ~ {query_e_dt}...")

            try:
                # 抓取 PC Ratio
                r1 = session.post("https://www.taifex.com.tw/cht/3/pcRatioDown",
                                   data={"queryStartDate": query_s_dt, "queryEndDate": query_e_dt}, headers=headers, timeout=10)
                if r1.status_code == 200 and 'DOCTYPE html' not in r1.text[:100].upper() and 'alert' not in r1.text[:200]:
                    content_str = r1.content.decode('big5', errors='ignore')
                    if year_offset > 0: content_str = content_str.replace(str(e_date.year - year_offset), str(e_date.year))
                    # 修正期交所 CSV 結尾逗號導致的欄位平移問題
                    lines = [line.strip().rstrip(',') for line in content_str.split('\n') if line.strip()]
                    content_str = '\n'.join(lines)
                    try:
                        df_chunk = pd.read_csv(io.StringIO(content_str), index_col=False)
                        if not df_chunk.empty and len(df_chunk.columns) > 2:
                            pc_frames.append(df_chunk)
                    except Exception as parse_e:
                        pass # Ignore chunk errors (e.g. weekends)

                time.sleep(1) # 增加延遲避免被封鎖

                # 抓取 期貨 OI
                r2 = session.post("https://www.taifex.com.tw/cht/3/futContractsDateDown",
                                   data={"queryStartDate": query_s_dt, "queryEndDate": query_e_dt, "commodityId": "TXF"}, headers=headers, timeout=10)
                if r2.status_code == 200 and 'DOCTYPE html' not in r2.text[:100].upper() and 'alert' not in r2.text[:200]:
                    content_str = r2.content.decode('big5', errors='ignore')
                    if year_offset > 0: content_str = content_str.replace(str(e_date.year - year_offset), str(e_date.year))
                    # 修正期交所 CSV 結尾逗號導致的欄位平移問題
                    lines = [line.strip().rstrip(',') for line in content_str.split('\n') if line.strip()]
                    content_str = '\n'.join(lines)
                    try:
                        df_chunk = pd.read_csv(io.StringIO(content_str), index_col=False)
                        if not df_chunk.empty and len(df_chunk.columns) > 2:
                            oi_frames.append(df_chunk)
                    except Exception as parse_e:
                        pass # Ignore chunk errors

                time.sleep(1) # 增加延遲避免被封鎖
            except Exception as e:
                print(f"⚠️ 區段下載警告: {e}")

        df_pc, df_foreign, df_dealer = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # 3. 清理 P/C Ratio
        if pc_frames:
            df_pc = pd.concat(pc_frames, ignore_index=True)
            df_pc.columns = df_pc.columns.str.strip()
            date_col = [c for c in df_pc.columns if '日期' in c]
            pc_col = [c for c in df_pc.columns if '比率' in c or '買賣權未平倉' in c]
            
            if date_col and pc_col:
                df_pc = df_pc[[date_col[0], pc_col[0]]].copy()
                df_pc.rename(columns={date_col[0]: 'date', pc_col[0]: 'pc_ratio'}, inplace=True)
                df_pc['date'] = pd.to_datetime(df_pc['date'].astype(str).str.strip(), format='mixed', errors='coerce').dt.date
                df_pc['pc_ratio'] = pd.to_numeric(df_pc['pc_ratio'].astype(str).str.replace(',', ''), errors='coerce') / 100.0

        # 4. 清理期貨 OI
        if oi_frames:
            df_oi = pd.concat(oi_frames, ignore_index=True)
            df_oi.columns = df_oi.columns.str.strip()

            # 兼容不同欄位名稱：商品名稱 或 契約名稱
            item_col = '商品名稱' if '商品名稱' in df_oi.columns else '契約名稱'
            if item_col not in df_oi.columns:
                item_cols = [c for c in df_oi.columns if '名稱' in c]
                if item_cols: item_col = item_cols[0]

            if item_col in df_oi.columns:
                df_oi = df_oi[df_oi[item_col].astype(str).str.strip() == '臺股期貨']
                net_col = [c for c in df_oi.columns if '未平倉' in c and '口數' in c]
                if net_col:
                    net_col = net_col[0]
                    df_foreign = df_oi[df_oi['身份別'].astype(str).str.strip() == '外資及陸資'][['日期', net_col]].rename(columns={'日期': 'date', net_col: 'foreign_net_oi'})
                    df_dealer = df_oi[df_oi['身份別'].astype(str).str.strip() == '自營商'][['日期', net_col]].rename(columns={'日期': 'date', net_col: 'dealer_net_oi'})

                    for df_tmp in [df_foreign, df_dealer]:
                        if not df_tmp.empty:
                            df_tmp['date'] = pd.to_datetime(df_tmp['date'].astype(str).str.strip(), format='mixed', errors='coerce')
                            df_tmp.dropna(subset=['date'], inplace=True)
                            df_tmp.iloc[:, 1] = pd.to_numeric(df_tmp.iloc[:, 1].astype(str).str.replace(',', ''), errors='coerce')

        if not df_pc.empty: df_pc['date'] = pd.to_datetime(df_pc['date'])

        # 5. 抓取證交所現貨 (FinMind API - 優化效能)
        twse_data = []
        token = os.environ.get('FINMIND_API_TOKEN', 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoibG9zdG1lbG9keSIsImVtYWlsIjoibGVhdmU5MThAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6MH0.BmG_w18TAEobmpkAA3BO_9mPvWiVrXwYfey_n7xRUQ4')
        try:
            print("🚀 使用 FinMind API 一次性抓取現貨買賣超...")
            url = 'https://api.finmindtrade.com/api/v4/data'
            
            # 若有平移年份，則請求真實歷史資料
            query_start = start_date.replace(year=start_date.year - year_offset).strftime("%Y-%m-%d")
            query_end = end_date.replace(year=end_date.year - year_offset).strftime("%Y-%m-%d")
            
            params = {
                'dataset': 'TaiwanStockTotalInstitutionalInvestors',
                'start_date': query_start,
                'end_date': query_end,
                'token': token
            }
            res = session.get(url, params=params, timeout=15)
            data = res.json().get('data', [])
            
            if data:
                df_fm = pd.DataFrame(data)
                for dt, grp in df_fm.groupby('date'):
                    f_s = grp[grp['name'].str.contains('Foreign_Investor')]['buy'].sum() - grp[grp['name'].str.contains('Foreign_Investor')]['sell'].sum()
                    d_s = grp[grp['name'].str.contains('Dealer')]['buy'].sum() - grp[grp['name'].str.contains('Dealer')]['sell'].sum()
                    
                    # 映射回未來時間
                    dt_obj = pd.to_datetime(dt)
                    if year_offset > 0: 
                        try:
                            dt_obj = dt_obj.replace(year=dt_obj.year + year_offset)
                        except ValueError:
                            # 處理 2 月 29 日平移到非閏年的問題
                            dt_obj = dt_obj.replace(year=dt_obj.year + year_offset, day=28)
                        
                    twse_data.append({'date': dt_obj, 'foreign_spot_net': f_s, 'dealer_spot_net': d_s})
        except Exception as e:
            print(f"⚠️ FinMind 現貨資料抓取失敗: {e}")

        df_twse = pd.DataFrame(twse_data) if twse_data else pd.DataFrame(columns=['date', 'foreign_spot_net', 'dealer_spot_net'])
        if not df_twse.empty: df_twse['date'] = pd.to_datetime(df_twse['date'])

        # 6. 合併與最終清理
        if df_pc.empty and df_foreign.empty and df_twse.empty:
            print("❌ 籌碼抓取全部失敗")
            return pd.DataFrame()

        df_final = pd.DataFrame()
        if not df_pc.empty: df_final = df_pc
        if not df_foreign.empty:
            df_final = df_final.merge(df_foreign, on='date', how='outer') if not df_final.empty else df_foreign
        if not df_dealer.empty:
            df_final = df_final.merge(df_dealer, on='date', how='outer') if not df_final.empty else df_dealer
        if not df_twse.empty: 
            df_final = df_final.merge(df_twse, on='date', how='outer') if not df_final.empty else df_twse

        if df_final.empty: return pd.DataFrame()

        df_final.sort_values('date', inplace=True)

        # 關鍵：ffill 補假日 -> bfill 補遺漏開頭 -> fillna(0) 補完全缺失
        df_final.ffill(inplace=True)
        df_final.bfill(inplace=True)
        df_final.fillna(0, inplace=True)

        # 過濾日期解析失敗的 NaT
        df_daily_chips = df_final.ffill().dropna()
        print(f"✅ 籌碼抓取完成，清理後有效樣本數: {len(df_daily_chips)}")
        
        # 保存快取
        df_daily_chips.to_parquet(cache_file)
        
        return df_daily_chips.reset_index(drop=True)

    def integrate_institutional_chips(self, df_intraday, df_daily_chips):
        """
        將三大法人特徵與 K 線融合，並執行分層填補策略 (Layered Imputation)
        """
        # 1. 數據預處理與日期型別對齊
        df_chips = df_daily_chips.copy()
        df_chips['date'] = pd.to_datetime(df_chips['date']).dt.date
        df_intraday['date_only'] = pd.to_datetime(df_intraday['date']).dt.date

        # 2. 計算 Z-Score 與 Momentum (加上容錯處理)
        for col in ['foreign_net_oi', 'dealer_net_oi']:
            if col in df_chips.columns:
                df_chips[f'{col}_zscore'] = (df_chips[col] - df_chips[col].rolling(20).mean()) / (df_chips[col].rolling(20).std() + 1e-9)
                df_chips[f'{col}_momentum'] = df_chips[col].diff()
                
        # --- 強化自營商籌碼特徵 (Dealer Enhancements) ---
        if 'dealer_net_oi_momentum' in df_chips.columns and 'foreign_net_oi_momentum' in df_chips.columns:
            # 衡量自營商相對於外資的動能強弱 (自營商動能 - 外資動能)
            df_chips['dealer_relative_momentum'] = df_chips['dealer_net_oi_momentum'] - df_chips['foreign_net_oi_momentum']
            
            # 自營商極端動能分數 (如果今天大買大賣)
            df_chips['dealer_extreme_score'] = (df_chips['dealer_net_oi_momentum'] / (df_chips['dealer_net_oi_momentum'].abs().rolling(10).mean() + 1e-9)).clip(-3, 3)

        # 3. 未來函數防禦：將籌碼指標 Shift(1) (僅在非單日快照模式下執行)
        if len(df_chips) > 1:
            shift_cols = [c for c in df_chips.columns if c != 'date' and 'date' not in c]
            df_chips[shift_cols] = df_chips[shift_cols].shift(1)
        else:
            print("ℹ️ [DataEngine] 偵測到單日籌碼快照，跳過 Shift(1) 以保留當前資訊。")

        # 4. 合併數據 (使用 merge_asof 確保即使日期有落差也能抓到最新的一筆籌碼)
        df_intraday = df_intraday.sort_values('date')
        df_chips = df_chips.sort_values('date').dropna(subset=['date'])
        
        # 移除含有 NaT 的 date_only 以避免 merge_asof 報錯
        df_intraday = df_intraday.dropna(subset=['date_only'])
        
        df_intraday['date_only_dt'] = pd.to_datetime(df_intraday['date_only'])
        df_chips['date_dt'] = pd.to_datetime(df_chips['date'])
        
        df_merged = pd.merge_asof(
            df_intraday, 
            df_chips, 
            left_on='date_only_dt', 
            right_on='date_dt', 
            direction='backward',
            suffixes=('', '_y')
        )
        
        # 移除重複的 _y 欄位，並將 date 恢復正常
        cols_to_drop = ['date_only_dt', 'date_dt'] + [c for c in df_merged.columns if c.endswith('_y')]
        df_merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # 5. 分層填補策略 (Layered Imputation Strategy)
        # 定義哪些指標是「水準值」，哪些是「變動量」
        ffill_cols = ['pc_ratio', 'foreign_net_oi', 'dealer_net_oi', 'foreign_net_oi_zscore', 'dealer_net_oi_zscore']
        zero_cols = ['foreign_net_oi_momentum', 'dealer_net_oi_momentum']

        # 執行填充：加入欄位存在檢查，防止 KeyError
        for col in ffill_cols:
            if col in df_merged.columns:
                # 先用 ffill 填補當日剩下的 K 線，再用 bfill 填補可能的空頭 (如開盤第一筆)
                df_merged[col] = df_merged[col].ffill().bfill()

        for col in zero_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0)

        # 最後的安全網：如果還有沒被補到的 (例如欄位名稱未定義)，全部填 0
        df_merged.fillna(0, inplace=True)

        print(f"✅ 籌碼融合完成，最終資料筆數: {len(df_merged)}")
        # 輸出一下目前的空值檢查報告
        null_count = df_merged.isnull().sum().sum()
        print(f"ℹ️ 最終檢查：融合後資料集尚有 {null_count} 個空值 (正常情況應為 0)")

        return df_merged.reset_index(drop=True)    # 輸出一下目前的空值檢查報告
        null_count = df_merged.isnull().sum().sum()
        print(f"ℹ️ 最終檢查：融合後資料集尚有 {null_count} 個空值 (正常情況應為 0)")

        return df_merged.reset_index(drop=True)