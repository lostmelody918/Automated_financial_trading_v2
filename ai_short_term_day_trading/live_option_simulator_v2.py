import os
import time
import json
import torch
import numpy as np
import pandas as pd
import queue
import sqlite3
import threading

pd.options.mode.string_storage = 'python' # Disable pyarrow to prevent Shioaji thread crash
import pyarrow # Pre-load pyarrow to prevent access violation with Shioaji threads
from datetime import datetime, timedelta, time as datetime_time

import shioaji as sj
from datetime import datetime

from data_engine import DayTradingDataEngine
from composite_ai import CompositeDayTradingAI
from model_manager import TradingModelManager
from strategy_factory import StrategyFactory
from delta_gamma_theta import get_dynamic_bsm_bounds, get_api_based_dte

# ==========================================
# 核心參數設定
# ==========================================
INITIAL_CAPITAL = 120000
CONTRACT_MULTIPLIER = 50
FEE_SLIPPAGE_PER_CONTRACT = 100
MAX_POSITION_CAPITAL = 4000000
WINDOW_SIZE = 40

class PositionManager:
    def __init__(self, db_path="position_state.db"):
        self.db_path = db_path
        self._init_db()
        self.state = self._load_state()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # 開啟 WAL 模式提升寫入效能，避免主迴圈卡頓
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute('''
                CREATE TABLE IF NOT EXISTS position_state (
                    id INTEGER PRIMARY KEY,
                    position INTEGER,
                    entry_price REAL,
                    num_contracts INTEGER,
                    highest_price_since_entry REAL,
                    active_contract_symbol TEXT,
                    entry_time TEXT,
                    trade_capital_used REAL,
                    hard_tp_price REAL,
                    hard_sl_price REAL,
                    strategy_label TEXT
                )
            ''')
            cur = conn.execute("SELECT id FROM position_state WHERE id=1")
            if not cur.fetchone():
                conn.execute('''
                    INSERT INTO position_state (
                        id, position, entry_price, num_contracts,
                        highest_price_since_entry, active_contract_symbol,
                        entry_time, trade_capital_used, hard_tp_price, hard_sl_price, strategy_label
                    ) VALUES (1, 0, 0.0, 0, 0.0, NULL, NULL, 0.0, 0.0, 0.0, NULL)
                ''')

    def _load_state(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM position_state WHERE id=1").fetchone()
            return dict(row)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.state[k] = v
        with sqlite3.connect(self.db_path) as conn:
            set_clause = ", ".join([f"{k}=?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [1]
            conn.execute(f"UPDATE position_state SET {set_clause} WHERE id=?", values)

    def clear_position(self):
        self.update(
            position=0, entry_price=0.0, num_contracts=0,
            highest_price_since_entry=0.0, active_contract_symbol=None,
            entry_time=None, trade_capital_used=0.0, hard_tp_price=0.0, hard_sl_price=0.0,
            strategy_label=None
        )

    def get(self, key, default=None):
        return self.state.get(key, default)


class KBarAccumulator:
    def __init__(self):
        self.current_min = None
        self.O = 0
        self.H = 0
        self.L = 0
        self.C = 0
        self.V = 0

    def on_tick(self, timestamp, price, volume):
        tick_min = timestamp.replace(second=0, microsecond=0)
        if self.current_min is None:
            self.current_min = tick_min
            self.O = self.H = self.L = self.C = price
            self.V = volume
            return None

        if tick_min > self.current_min:
            finished_bar = {
                'date': self.current_min,
                'Open': self.O,
                'High': self.H,
                'Low': self.L,
                'Close': self.C,
                'Volume': self.V,
                'Amount': 0
            }
            self.current_min = tick_min
            self.O = self.H = self.L = self.C = price
            self.V = volume
            return finished_bar
        else:
            self.H = max(self.H, price)
            self.L = min(self.L, price)
            self.C = price
            self.V += volume
            return None

def is_market_open(current_time):
    return datetime_time(8, 45) <= current_time <= datetime_time(13, 45)

def is_eod_closing_time(now_dt, is_settlement=False):
    current_time = now_dt.time() if hasattr(now_dt, 'time') else now_dt
    if is_settlement:
        # 當日到期結算日，提早到 13:15 強制平倉了結，避免過度深入結算平均價計算期
        return datetime_time(13, 15) <= current_time < datetime_time(13, 45)
    # 非交割日：13:25 之後波動太小且時間價值流失快，提早到 13:25 強制平倉/不進場
    return datetime_time(13, 25) <= current_time < datetime_time(13, 45)

def load_latest_daily_chips_snapshot():
    print("📥 載入本地籌碼快照 (chips_cache.json)...")
    cache_path = os.path.join(os.path.dirname(__file__), "chips_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df_chips = pd.DataFrame(data)
            df_chips['date'] = pd.to_datetime(df_chips['date'])
            print(f"✅ 成功載入 {len(df_chips)} 天籌碼歷史。")
            return df_chips
        except Exception as e:
            print(f"⚠️ 載入 chips_cache.json 失敗: {e}")

    return pd.DataFrame({
        'date': [pd.to_datetime(datetime.now().date() - timedelta(days=1))],
        'foreign_net_oi': [0.0],
        'dealer_net_oi': [0.0],
        'trust_net_oi': [0.0],
        'pc_ratio': [1.0]
    })

def generate_eod_report(trade_log, initial_capital, current_capital, out_dir="data_learn"):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    print("\n" + "="*60 + f"\n📊  --- 今日選擇權即時模擬當沖結算報告 ---\n" + "="*60)
    total_trades = len(trade_log)
    total_pnl = current_capital - initial_capital
    print(f"💰 初始本金 : NT$ {initial_capital:,}\n💵 結算淨值 : NT$ {current_capital:,}\n📈 總淨盈虧 : NT$ {total_pnl:,.0f} ({(total_pnl / initial_capital) * 100:.2f}%)\n" + "-" * 50)
    if total_trades > 0:
        df_trades = pd.DataFrame(trade_log)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        print(f"⏱️ 總交易次數 : {total_trades} 次 | 🎯 當日勝率 : {(len(wins) / total_trades) * 100:.2f}%")
        csv_path = os.path.join(out_dir, f"daily_trade_report_{datetime.now().strftime('%Y%m%d')}.csv")
        df_trades.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"💾 已儲存今日交易明細至: {csv_path}")
    else: print("❌ 今日無交易紀錄。")
    print("="*60 + "\n")

def calculate_features(df_raw):
    df = df_raw.copy()
    if df.empty: return df

    df['time'] = df['date'].dt.time
    df['date_only'] = df['date'].dt.date
    df['day_of_week'] = df['date'].dt.dayofweek

    df['ret'] = df['Close'].pct_change()
    df['mock_volume'] = df['Volume'].replace(0, 1)
    df['vol_price'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['mock_volume']

    df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['mock_volume'].cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    df['vwap_bias'] = (df['Close'] - df['vwap']) / (df['vwap'] + 1e-9)

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']

    df['h_l'] = df['High'] - df['Low']
    df['h_pc'] = abs(df['High'] - df['Close'].shift(1))
    df['l_pc'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    gain_fast = (delta.where(delta > 0, 0)).rolling(window=3).mean()
    loss_fast = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
    rs_fast = gain_fast / (loss_fast + 1e-9)
    df['rsi_fast'] = 100 - (100 / (1 + rs_fast))

    df['body_length'] = abs(df['Close'] - df['Open'])
    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    df['body_avg_5'] = df['body_length'].rolling(5).mean()
    df['momentum_explosion'] = (df['body_length'] > df['body_avg_5']).astype(int)

    df['daily_open'] = df.groupby('date_only')['Open'].transform('first')

    daily_close = df.groupby('date_only')['Close'].last().shift(1)
    df['yesterday_close'] = df['date_only'].map(daily_close)
    df['yesterday_close'] = df['yesterday_close'].fillna(df['daily_open'])

    df['gap_amplitude'] = (df['daily_open'] - df['yesterday_close']) / (df['yesterday_close'] + 1e-9)
    df['intraday_trend'] = (df['Close'] - df['daily_open']) / (df['daily_open'] + 1e-9)

    # 均值回歸與突破輔助特徵 (Mean Reversion / Breakout Aux)
    # 1. 乖離均線 (Distance from MA) - 判斷拉回深度
    df['dist_from_ma20'] = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).mean() + 1e-9)

    # 2. 區間高低點回撤 (Pullback Depth) - 判斷目前是創高還是拉回
    df['recent_high_20'] = df['High'].rolling(20).max()
    df['recent_low_20'] = df['Low'].rolling(20).min()
    df['pullback_from_high'] = (df['Close'] - df['recent_high_20']) / (df['recent_high_20'] + 1e-9)
    df['bounce_from_low'] = (df['Close'] - df['recent_low_20']) / (df['recent_low_20'] + 1e-9)

    df['gap_filled'] = 0
    df.loc[(df['gap_amplitude'] > 0) & (df['Close'] <= df['yesterday_close']), 'gap_filled'] = 1
    df.loc[(df['gap_amplitude'] < 0) & (df['Close'] >= df['yesterday_close']), 'gap_filled'] = 1

    df['spot_futures_proxy'] = (df['Close'] - df['vwap']) / (df['Close'].rolling(20).mean() + 1e-9)

    # 3. 平滑價格與反轉突破點 (Smoothed Price & Inflection Breakout)
    df['smooth_price'] = df['Close'].ewm(span=9, adjust=False).mean()
    # 尋找反曲點 (找轉折)
    df['is_peak'] = (df['smooth_price'].shift(1) > df['smooth_price'].shift(2)) & (df['smooth_price'].shift(1) > df['smooth_price'])
    df['is_trough'] = (df['smooth_price'].shift(1) < df['smooth_price'].shift(2)) & (df['smooth_price'].shift(1) < df['smooth_price'])
    
    # 記錄前一次轉折點的值
    df['recent_peak_val'] = df['smooth_price'].where(df['is_peak']).ffill()
    df['recent_trough_val'] = df['smooth_price'].where(df['is_trough']).ffill()
    
    # 判斷突破 (1: 突破前高, -1: 跌破前低, 0: 區間內)
    df['inflection_breakout'] = np.where(df['Close'] > df['recent_peak_val'], 1,
                                  np.where(df['Close'] < df['recent_trough_val'], -1, 0))

    df['sma20'] = df['Close'].rolling(window=20).mean()
    df['std20'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
    df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['sma20'] + 1e-9)
    df['is_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)

    df['vwap_5'] = df['mock_volume'] * df['Close'] / df['mock_volume'].rolling(5).sum()
    df['slope_vwap'] = (df['vwap_5'] - df['vwap_5'].shift(3)) / df['vwap_5'].shift(3) * 10000

    df['ma_20'] = df['Close'].rolling(20).mean()
    df['slope_ma20'] = (df['ma_20'] - df['ma_20'].shift(3)) / df['ma_20'].shift(3) * 10000

    df['vol_surge_ratio'] = df['mock_volume'] / (df['mock_volume'].rolling(20).mean() + 1e-9)
    df['price_roc'] = df['Close'].pct_change()
    df['pv_divergence'] = np.where((df['price_roc'] > 0) & (df['vol_surge_ratio'] < 0.8), -1,
                        np.where((df['price_roc'] < 0) & (df['vol_surge_ratio'] < 0.8), 1, 0))

    cols_to_drop = ['vol_price', 'cum_vol_price', 'cum_vol', 'h_l', 'h_pc', 'l_pc', 'tr', 'sma20', 'std20', 'Amount']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    return df.dropna().reset_index(drop=True)

def run_live_simulator():
    print("=" * 60 + "\n🚀 啟動：底層期貨特徵分離版 AI 選擇權即時模擬機 (Event-Driven)\n" + "=" * 60)
    engine = DayTradingDataEngine()

    with open(os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json"), 'r', encoding='utf-8') as f:
        norm_params = json.load(f)

    feature_cols = [c for c in norm_params['feature_cols'] if c in norm_params['mean']]
    mean_v = np.array([norm_params['mean'][c] for c in feature_cols])
    std_v = np.array([norm_params['std'][c] for c in feature_cols])

    ai_model = CompositeDayTradingAI(input_dim=len(feature_cols), d_model=256, nhead=16, num_layers=4)
    model_manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
    ai_model, _, version = model_manager.load_latest_model(ai_model)
    ai_model.eval()

    strategy_engine = StrategyFactory.get_strategy("composite")

    # Initialize Position Manager
    pos_manager = PositionManager(os.path.join(os.path.dirname(__file__), "position_state.db"))

    current_active_code = None

    # 啟動時檢查：若有遺留的未平倉部位，進行跨日清除或當日恢復訂閱
    entry_time_str = pos_manager.get('entry_time')
    if pos_manager.get('position') != 0 and entry_time_str:
        try:
            entry_date = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S").date()
            active_contract_symbol = pos_manager.get('active_contract_symbol')
            if entry_date != datetime.now().date():
                print(f"🧹 發現跨日未平倉部位 ({active_contract_symbol})，自動清除以確保當沖邏輯！")
                pos_manager.clear_position()
            else:
                # 這是今天的部位！必須重新訂閱它的行情才能繼續追蹤平倉
                print(f"🔄 偵測到今日未平倉部位 ({active_contract_symbol})，準備重新訂閱行情...")
                active_contract_obj = None
                for cat in ['TXO', 'TX1', 'TX2', 'TX4', 'TX5', 'TXU', 'TXV', 'TXX']:
                    if hasattr(engine.api.Contracts.Options, cat):
                        for c in getattr(engine.api.Contracts.Options, cat):
                            if c.symbol == active_contract_symbol:
                                active_contract_obj = c
                                break
                    if active_contract_obj:
                        break

                if active_contract_obj:
                    try:
                        engine.api.quote.subscribe(active_contract_obj, quote_type=sj.constant.QuoteType.Tick)
                        engine.api.quote.subscribe(active_contract_obj, quote_type=sj.constant.QuoteType.BidAsk)
                        current_active_code = active_contract_obj.code
                        print(f"✅ 成功重新訂閱 {active_contract_symbol} (Code: {current_active_code}) 行情，繼續監控平倉條件！")
                    except Exception as e:
                        print(f"⚠️ 重新訂閱失敗: {e}，強制清除部位！")
                        pos_manager.clear_position()
                else:
                    print(f"⚠️ 無法找到未平倉部位 ({active_contract_symbol}) 的合約物件，強制清除部位！")
                    pos_manager.clear_position()
        except Exception as e:
            print(f"⚠️ 解析遺留部位發生錯誤 ({e})，強制清除部位！")
            pos_manager.clear_position()

    current_capital = INITIAL_CAPITAL
    last_trade_win = False
    consecutive_failures = {'Call': 0, 'Put': 0}
    consecutive_total_losses = 0
    cooldown_until = None
    last_trade_symbol = None
    last_trade_closed_time = None
    last_trade_pnl = 0.0
    trade_log = []
    eod_report_done = False
    entry_features = {}

    df_chips_daily = load_latest_daily_chips_snapshot()

    print("📥 正在獲取初始 5 天 K 線歷史資料...")
    today = datetime.now()
    start_date = (today - timedelta(days=5)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    txf_contract = engine.api.Contracts.Futures.TXF.TXFR1
    kbars = engine.api.kbars(txf_contract, start=start_date, end=end_date)
    df_raw = pd.DataFrame({**kbars})
    df_raw['ts'] = pd.to_datetime(df_raw['ts'])
    df_raw = df_raw.sort_values('ts').reset_index(drop=True)
    df_raw.rename(columns={'ts': 'date'}, inplace=True)
    print(f"✅ 成功載入 {len(df_raw)} 筆初始 K 線！")

    event_queue = queue.Queue()
    accumulator = KBarAccumulator()
    opt_quotes = {}
    current_txf_price = df_raw.iloc[-1]['Close'] if not df_raw.empty else 0.0

    def on_tick_fop_callback(exchange, tick):
        try:
            symbol = getattr(tick, 'code', None)
            if not symbol: return
            now = datetime.now()
            price = getattr(tick, 'close', None)
            if price is None: return

            if "TXF" in symbol:
                volume = getattr(tick, 'volume', 1)
                event_queue.put(('TXF_TICK', now, float(price), int(volume)))
            else:
                # 這裡不一定有最佳買賣價，我們主要靠 on_bidask_fop_v1_callback 更新
                # 但如果這裡有帶，還是送進去更新 price
                event_queue.put(('OPT_TICK', symbol, now, float(price), 0.0, 0.0))
        except Exception as e:
            print(f"⚠️ on_tick_fop_callback 錯誤: {e}")

    def on_bidask_fop_v1_callback(exchange, bidask):
        try:
            symbol = getattr(bidask, 'code', None)
            if not symbol: return
            now = datetime.now()

            # Shioaji 的 BidAsk 物件有 bid_price / ask_price 陣列 (Decimals)
            bids = getattr(bidask, 'bid_price', [])
            asks = getattr(bidask, 'ask_price', [])

            bid = float(bids[0]) if bids and len(bids) > 0 else 0.0
            ask = float(asks[0]) if asks and len(asks) > 0 else 0.0

            # 發送更新，由於 BidAsk 沒有成交價，price 代 0.0 (主迴圈會忽略 price=0 的更新，只更新 bid/ask)
            event_queue.put(('OPT_TICK', symbol, now, 0.0, float(bid), float(ask)))
        except Exception as e:
            print(f"⚠️ on_bidask_fop_v1_callback 錯誤: {e}")

    def quote_callback(topic, quote):
        try:
            symbol = topic.split('/')[-1]
            now = datetime.now()

            price = getattr(quote, 'close', getattr(quote, 'Close', None))
            bid_prices = getattr(quote, 'BidPrice', getattr(quote, 'bid_price', []))
            ask_prices = getattr(quote, 'AskPrice', getattr(quote, 'ask_price', []))

            # 如果沒有 price 但有買賣價，這可能是一個單純的委託簿更新
            if price is None and not (bid_prices or ask_prices): return

            if "TXF" in symbol:
                if price is not None:
                    volume = getattr(quote, 'volume', getattr(quote, 'Volume', 1))
                    event_queue.put(('TXF_TICK', now, float(price), int(volume)))
            else:
                bid = float(bid_prices[0]) if bid_prices else 0.0
                ask = float(ask_prices[0]) if ask_prices else 0.0
                p = float(price) if price is not None else 0.0
                event_queue.put(('OPT_TICK', symbol, now, p, bid, ask))
        except Exception as e:
            print(f"⚠️ quote_callback 錯誤: {e}")

    try:
        # 兼容不同版本的 Shioaji API
        if hasattr(engine.api.quote, 'set_on_tick_fop_v1_callback'):
            engine.api.quote.set_on_tick_fop_v1_callback(on_tick_fop_callback)

        if hasattr(engine.api.quote, 'set_on_bidask_fop_v1_callback'):
            engine.api.quote.set_on_bidask_fop_v1_callback(on_bidask_fop_v1_callback)

        if hasattr(engine.api.quote, 'set_on_tick_fnc'):
            engine.api.quote.set_on_tick_fnc(quote_callback)
        elif hasattr(engine.api.quote, 'set_quote_callback'):
            engine.api.quote.set_quote_callback(quote_callback)
        elif hasattr(engine.api, 'on_tick'):
            engine.api.on_tick(quote_callback)

        # 確保以 v1 版本訂閱 Tick 與 BidAsk
        if hasattr(sj.constant, 'QuoteVersion'):
            engine.api.quote.subscribe(txf_contract, quote_type=sj.constant.QuoteType.Tick, version=sj.constant.QuoteVersion.v1)
            engine.api.quote.subscribe(txf_contract, quote_type=sj.constant.QuoteType.BidAsk, version=sj.constant.QuoteVersion.v1)
        else:
            engine.api.quote.subscribe(txf_contract, quote_type=sj.constant.QuoteType.Tick)
    except Exception as e:
        print(f"⚠️ 訂閱失敗或 API 版本不相容: {e}")

    print("✅ 事件驅動引擎啟動，進入主迴圈...")
    last_print_time = datetime.now()
    heartbeat_time = datetime.now()
    is_today_settlement = None

    while True:
        now = datetime.now()
        current_time = now.time()
        time_str = now.strftime('%H:%M:%S')

        if not is_market_open(current_time):
            if pos_manager.get('position') != 0:
                pos_manager.clear_position()
            if eod_report_done and current_time < datetime_time(8, 45):
                eod_report_done = False
                trade_log = []
                is_today_settlement = None

        if is_today_settlement is None and is_market_open(current_time):
            try:
                temp_contract = engine.get_best_volume_option_contract(option_type='Call', allocated_capital=100000)
                if temp_contract:
                    is_today_settlement = (get_api_based_dte(temp_contract, now) < 1.0)
                    print(f"[{time_str}] 📅 今日結算狀態: {is_today_settlement} (參考合約交割日: {getattr(temp_contract, 'delivery_date', 'N/A')})")
                else:
                    is_today_settlement = False
            except Exception as e:
                print(f"⚠️ 無法判斷結算日: {e}")
                is_today_settlement = False

        if is_eod_closing_time(now, is_settlement=is_today_settlement) and pos_manager.get('position') == 0 and not eod_report_done:
            generate_eod_report(trade_log, INITIAL_CAPITAL, current_capital)
            
            # --- 新增：匯出盤後資料與視覺化報表 ---
            try:
                from post_market_export import PostMarketExporter
                exporter = PostMarketExporter()
                
                dict_options_history = {}
                # 由於這裡無法直接簡單取得 Option K線，我們傳入空的 DataFrame 或依賴先前的資料
                # 對於實際應用，可以透過 engine.api.kbars(symbol) 來補齊
                # 這裡我們先初始化空的，讓 exporter 處理
                traded_symbols = set([t.get('symbol') for t in trade_log if t.get('symbol')])
                for sym in traded_symbols:
                    dict_options_history[sym] = pd.DataFrame()
                    
                # df_intraday 在這邊不一定被定義，我們使用 df_raw 或是全域維護的特徵 df
                if 'df_intraday' in locals() and df_intraday is not None and not df_intraday.empty:
                    df_to_export = df_intraday
                else:
                    df_to_export = df_raw
                    
                exporter.execute_export(df_to_export, dict_options_history, trade_log)
            except Exception as exp_err:
                print(f"⚠️ 匯出盤後資料失敗: {exp_err}")
            # ----------------------------------------
            
            eod_report_done = True
            time.sleep(300)
            continue

        try:
            q_timeout = 1.0 if is_market_open(current_time) else 5.0
            event = event_queue.get(timeout=q_timeout)
            event_type = event[0]

            if event_type == 'TXF_TICK':
                _, tick_time, price, volume = event
                current_txf_price = price
                bar = accumulator.on_tick(tick_time, price, volume)

                if bar:
                    # --- 1-Minute Bar Completed ---
                    new_row = pd.DataFrame([bar])
                    df_raw = pd.concat([df_raw, new_row], ignore_index=True)

                    df_intraday = calculate_features(df_raw)
                    if df_intraday is None or df_intraday.empty:
                        continue

                    df = engine.integrate_institutional_chips(df_intraday, df_chips_daily) if hasattr(engine, 'integrate_institutional_chips') else df_intraday

                    if df is None or df.empty or len(df) < WINDOW_SIZE:
                        continue

                    for missing_col in feature_cols:
                        if missing_col not in df.columns:
                            df[missing_col] = 0.0

                    df_slice = df[feature_cols].tail(WINDOW_SIZE).copy()
                    for col in ['mock_volume', 'macd_hist', 'vwap_bias']:
                        if col in df_slice.columns:
                            df_slice[col] = np.sign(df_slice[col]) * np.log1p(np.abs(df_slice[col]))

                    feat_tensor = torch.tensor(np.nan_to_num((df_slice.values - mean_v) / np.where(std_v == 0, 1.0, std_v), nan=0.0), dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        probs = torch.softmax(ai_model(feat_tensor), dim=1).squeeze().cpu().numpy()

                    # --- DEBUG AI ---
                    max_idx = int(np.argmax(probs))
                    confidence = probs[max_idx]
                    class_names = {
                        0: "Strong Down (-3)", 1: "Med Down (-2)", 2: "Weak Down (-1)",
                        3: "Hold (0)",
                        4: "Weak Up (1)", 5: "Med Up (2)", 6: "Strong Up (3)"
                    }
                    class_name = class_names.get(max_idx, "Unknown")

                    pos = pos_manager.get('position')
                    if pos != 0:
                        sym = pos_manager.get('active_contract_symbol')
                        ep = pos_manager.get('entry_price')
                        hp = pos_manager.get('highest_price_since_entry')
                        sl = pos_manager.get('hard_sl_price')
                        tp = pos_manager.get('hard_tp_price')

                        quote_data = opt_quotes.get(current_active_code, {}) if current_active_code else {}
                        current_opt_price = quote_data.get('price', 0)
                        if not current_opt_price or current_opt_price <= 0:
                            bid = quote_data.get('bid', 0)
                            ask = quote_data.get('ask', 0)
                            if bid > 0 and ask > 0:
                                current_opt_price = round((bid + ask) / 2, 1)
                            else:
                                current_opt_price = 'N/A'

                        print(f"[{time_str}] 🤖 AI 預測完成: Class={max_idx-3} ({class_name}), Confidence={confidence:.2%} | 📊 持倉: {sym} (現價: {current_opt_price}, 進: {ep}, 高: {hp}, 防: {sl:.1f}, 利: {tp})")
                    else:
                        print(f"[{time_str}] 🤖 AI 預測完成: Class={max_idx-3} ({class_name}), Confidence={confidence:.2%}, Probs={np.round(probs, 3)}")

                    # --- Entry Logic ---
                    if pos_manager.get('position') == 0 and is_market_open(now.time()) and not is_eod_closing_time(now, is_settlement=is_today_settlement):
                        if cooldown_until and now < cooldown_until:
                            continue
                        elif cooldown_until and now >= cooldown_until:
                            cooldown_until = None
                            consecutive_total_losses = 0
                            print(f"[{time_str}] 🟢 冷卻時間結束，恢復交易！")

                        signal = strategy_engine.generate_signal(df, ai_score=probs, last_win=last_trade_win)

                        # 先決定潛在的交易方向，以便提前抓取合約
                        temp_opt_type = 'Call'
                        if signal != 0:
                            temp_opt_type = 'Call' if signal > 0 else 'Put'

                            # 連續虧損 2 次防護 (冷卻一次)
                            if consecutive_failures[temp_opt_type] >= 2:
                                print(f"[{time_str}] ⚠️ {temp_opt_type} 方向已連續虧損 2 次，觸發冷卻機制，本次放棄進場！")
                                consecutive_failures[temp_opt_type] = 0 # 放棄一次後重置
                                continue
                        else:
                            trend_probs = probs.copy()
                            trend_probs[3] = 0
                            max_trend_idx = int(np.argmax(trend_probs))
                            temp_opt_type = 'Call' if max_trend_idx > 3 else 'Put'

                        allocated_capital_limit = min(current_capital * 0.50, MAX_POSITION_CAPITAL)
                        active_contract = engine.get_best_volume_option_contract(option_type=temp_opt_type, allocated_capital=allocated_capital_limit)

                        if not active_contract:
                            continue

                        # 反手交易邏輯 (Reversal Logic)
                        opt_type = temp_opt_type
                        if last_trade_symbol == active_contract.symbol and last_trade_pnl < 0 and last_trade_closed_time:
                            if (now - last_trade_closed_time).total_seconds() <= 240: # 4分鐘內
                                print(f"[{time_str}] 🔄 【反向進場觸發】發現與上次停損合約相同 ({active_contract.symbol}) 且距離平倉小於 4 分鐘，執行反手！")
                                opt_type = 'Put' if temp_opt_type == 'Call' else 'Call'
                                active_contract = engine.get_best_volume_option_contract(option_type=opt_type, allocated_capital=allocated_capital_limit)
                                if not active_contract:
                                    continue
                                # 反轉信號
                                signal = -signal if signal != 0 else (1 if opt_type == 'Call' else -1)

                        # 針對「真正的選擇權合約」計算 DTE
                        dte_days_option = get_api_based_dte(active_contract, now)
                        is_settlement_day = dte_days_option < 1.0

                        # 計算動態持有時間 (針對結算日縮短預期，加速獲利了結或停損)
                        expected_hold_time = 2.0
                        if is_settlement_day:
                            if current_time >= datetime_time(13, 15):
                                expected_hold_time = 0.25
                            elif current_time >= datetime_time(12, 30):
                                expected_hold_time = 0.5
                            else:
                                expected_hold_time = 1.0

                        # 降低門檻機制：即使 AI 預測最強類別是 Hold (0)，但若有其他趨勢類別達到指定信心度，則強制進場
                        if signal == 0:
                            trend_probs = probs.copy()
                            trend_probs[3] = 0 # 排除 Hold 類別
                            max_trend_idx = int(np.argmax(trend_probs))

                            # 針對不同 Level 設定不同的激進門檻以增加短線與波段的出手次數
                            abs_level = abs(max_trend_idx - 3)
                            if abs_level == 3:
                                threshold = 0.37 # Level 3 極強勢維持較高標準
                            elif abs_level == 2:
                                threshold = 0.27 # Level 2 標準波段降低至 25%
                            else:
                                threshold = 0.22 # Level 1 短線游擊降低至 20% (從0.22微降)

                            # 依據預期持倉時間動態提高門檻 (時間越短，容錯率越低，要求更高的爆發力)
                            if expected_hold_time <= 0.25:
                                threshold += 0.12 # 剩餘 15 分鐘，門檻大幅提高 15%
                            elif expected_hold_time <= 0.5:
                                threshold += 0.07 # 剩餘 30 分鐘，門檻提高 10%
                            elif expected_hold_time <= 1.0:
                                threshold += 0.03 # 剩餘 1 小時，門檻提高 5%

                            if trend_probs[max_trend_idx] > threshold:
                                mapping = {0:-3, 1:-2, 2:-1, 3:0, 4:1, 5:2, 6:3}
                                signal = mapping.get(max_trend_idx, 0)
                                if signal != 0:
                                    print(f"[{time_str}] ⚠️ 觸發激進短線門檻：偵測到 Level {abs_level} 趨勢 (Class {max_trend_idx-3}), Prob={trend_probs[max_trend_idx]:.2%} > 動態門檻 {threshold:.2%} (預期持倉 {expected_hold_time}h)")
                            elif trend_probs[max_trend_idx] > 0.15:
                                # 只印出較有機會但不夠門檻的，避免洗版
                                print(f"[{time_str}] ℹ️ 趨勢訊號不足門檻：Level {abs_level} (Class {max_trend_idx-3}), Prob={trend_probs[max_trend_idx]:.2%} <= 門檻 {threshold:.2%} (預期持倉 {expected_hold_time}h)")

                        # 價格反曲點突破追加邏輯 (Inflection Breakout)
                        if 'inflection_breakout' in df.columns:
                            inf_break = df['inflection_breakout'].iloc[-1]
                            recent_peak = df['recent_peak_val'].iloc[-1]
                            recent_trough = df['recent_trough_val'].iloc[-1]
                            if inf_break == 1 and signal <= 0:
                                print(f"[{time_str}] 📈 價格突破前次反曲高點 ({recent_peak:.1f})，確認多頭趨勢，產生順勢做多信號！")
                                signal = 1
                            elif inf_break == -1 and signal >= 0:
                                print(f"[{time_str}] 📉 價格跌破前次反曲低點 ({recent_trough:.1f})，確認空頭趨勢，產生順勢做空信號！")
                                signal = -1

                        if signal != 0:
                            opt_type = 'Call' if signal > 0 else 'Put'
                            try:
                                snap = engine.api.snapshots([active_contract])[0]

                                # 吃單成本修正：進場買入 (Long) 時應支付 Best Ask (賣價)
                                # 這裡假設 best_ask 存在於快照中
                                best_bid = getattr(snap, 'buy_price', 0)
                                if not best_bid and hasattr(snap, 'bids') and snap.bids:
                                    best_bid = snap.bids[0].price
                                best_ask = getattr(snap, 'sell_price', 0)
                                if not best_ask and hasattr(snap, 'asks') and snap.asks:
                                    best_ask = snap.asks[0].price

                                if best_ask < 5:
                                    print(f"⚠️ {active_contract.symbol} 權利金過低 (Best Ask={best_ask} < 5)，放棄進場！")
                                    continue

                                entry_price = best_ask # 實際成交在賣價

                                # 獲取當前波動率 (ATR) 以調整價差容忍度
                                current_atr = df_intraday['atr'].iloc[-1]
                                # 假設一般 ATR 約為 15-20。如果大於 25，視為高波動；大於 35 視為極端波動
                                vol_multiplier = 1.0
                                if current_atr > 35:
                                    vol_multiplier = 2.0
                                elif current_atr > 25:
                                    vol_multiplier = 1.5

                                # 導入階梯式百分比價差過濾機制，拒絕惡意價差與瞬間抽單
                                if best_ask < 50:
                                    # 低價位：允許最大 2.0 點 或 5% (最高 2.5點)
                                    dynamic_spread_threshold = max(2.0, best_ask * 0.05) * vol_multiplier
                                elif best_ask < 100:
                                    # 中價位：允許最大 3.0 點 或 4% (最高 4.0點)
                                    dynamic_spread_threshold = max(3.0, best_ask * 0.04) * vol_multiplier
                                else:
                                    # 高價位 (破百點)：嚴格限制在 3% 或最低 5.0 點，避免抽單瞬間蒸發
                                    # 加上波動率調整，極端殺盤時最高容忍可達 10 點以上
                                    dynamic_spread_threshold = max(5.0, best_ask * 0.03) * vol_multiplier

                                if best_ask > 0 and best_bid > 0 and (best_ask - best_bid) > dynamic_spread_threshold:
                                    print(f"⚠️ {active_contract.symbol} 買賣價差過大 ({best_ask} - {best_bid} = {best_ask-best_bid:.1f} > {dynamic_spread_threshold:.1f}，ATR={current_atr:.1f})，放棄進場！")
                                    continue

                            except Exception as e:
                                print(f"獲取快照失敗: {e}")
                                continue

                            abs_sig = abs(signal)  #停利/停損乘數
                            if abs_sig == 10:
                                tp_mult, sl_mult = 1.5, 1.9
                                strategy_label = f"⚡ Level 10 V轉極速剝頭皮 (Buy {opt_type})"
                            elif abs_sig == 3:
                                tp_mult, sl_mult = 4.0, 5.0
                                strategy_label = f"🚀 Level 3 極強勢波段 (Buy {opt_type})"
                            elif abs_sig == 2:
                                tp_mult, sl_mult = 2.2, 3.2
                                strategy_label = f"📈 Level 2 標準波段 (Buy {opt_type})"
                            else:
                                tp_mult, sl_mult = 1.6, 2.2
                                strategy_label = f"⚡ Level 1 短線游擊 (Buy {opt_type})"

                            current_atr = df_intraday['atr'].iloc[-1]
                            try:
                                strike_p = float(active_contract.strike_price) if hasattr(active_contract, 'strike_price') else current_txf_price
                            except:
                                strike_p = current_txf_price

                            # 改用 API 真實抓取的剩餘天數
                            days_to_expiry = get_api_based_dte(active_contract, now) / 365.0
                            current_iv = 0.22

                            # 將手續費換算成點數 (每口 100 元，大台選擇權 1 點 50 元)
                            fee_points = FEE_SLIPPAGE_PER_CONTRACT / float(CONTRACT_MULTIPLIER)

                            hard_tp_price, hard_sl_price, d, g, t_decay = get_dynamic_bsm_bounds(
                                S=current_txf_price,
                                K=strike_p,
                                T=days_to_expiry,
                                r=0.015,
                                iv=current_iv,
                                atr=current_atr,
                                tp_mult=tp_mult,
                                sl_mult=sl_mult,
                                expected_hold_hours=expected_hold_time,
                                option_type=opt_type,
                                actual_entry_price=entry_price,
                                fee_points=fee_points
                            )

                            expected_profit_points = hard_tp_price - entry_price
                            if expected_profit_points < 3.5:
                                print(f"[{time_str}] ⚠️ 預期權利金獲利空間太小 ({expected_profit_points:.1f} 點 < 3.5 點)，放棄進場！")
                                continue

                            num_contracts = max(1, int(allocated_capital_limit // (entry_price * CONTRACT_MULTIPLIER))) if entry_price > 0 else 1
                            trade_capital_used = num_contracts * entry_price * CONTRACT_MULTIPLIER

                            pos_manager.update(
                                position=1 if signal > 0 else -1,
                                entry_price=entry_price,
                                num_contracts=num_contracts,
                                highest_price_since_entry=entry_price,
                                active_contract_symbol=active_contract.symbol,
                                entry_time=now.strftime("%Y-%m-%d %H:%M:%S"),
                                trade_capital_used=trade_capital_used,
                                hard_tp_price=hard_tp_price,
                                hard_sl_price=hard_sl_price,
                                strategy_label=strategy_label
                            )
                            current_active_code = active_contract.code
                            entry_features = {f"feat_{k}": v for k, v in df.iloc[-1].to_dict().items()}

                            try:
                                engine.api.quote.subscribe(active_contract, quote_type=sj.constant.QuoteType.Tick)
                                engine.api.quote.subscribe(active_contract, quote_type=sj.constant.QuoteType.BidAsk)
                            except:
                                pass

                            delivery_date_str = getattr(active_contract, 'delivery_date', 'N/A')
                            print(f"\n{'='*40}\n🔥 【盤中進場】: {strategy_label}\n當前本金: NT$ {current_capital:,.0f} | 合約: {active_contract.symbol} (交割日: {delivery_date_str})\n進場價: {entry_price:.2f} | {num_contracts} 口")
                            print(f"📊 [BSM風控對齊] Delta: {d:.3f} | Gamma: {g:.5f} | Theta 損耗: {t_decay:.2f} 點")
                            print(f"🎯 動態停利點: {hard_tp_price} | 動態停損點: {hard_sl_price}\n{'='*40}\n")

            elif event_type == 'OPT_TICK':
                _, symbol, tick_time, price, bid, ask = event

                if symbol not in opt_quotes:
                    opt_quotes[symbol] = {'price': 0.0, 'bid': 0.0, 'ask': 0.0}

                if price > 0:
                    opt_quotes[symbol]['price'] = price
                if bid > 0:
                    opt_quotes[symbol]['bid'] = bid
                if ask > 0:
                    opt_quotes[symbol]['ask'] = ask

                position = pos_manager.get('position')
                if position != 0 and current_active_code == symbol:
                    # 離場成本修正：出場平倉 (賣出) 時應對齊 Best Bid (買價)
                    best_bid = opt_quotes[symbol]['bid']
                    last_price = opt_quotes[symbol]['price']

                    if best_bid <= 0:
                        # 若無委買價，退而求其次使用最新成交價 (price) 來進行出場判斷，避免永遠卡單無法平倉
                        best_bid = last_price

                    exec_price = best_bid
                    entry_price = pos_manager.get('entry_price')
                    hard_tp_price = pos_manager.get('hard_tp_price')
                    hard_sl_price = pos_manager.get('hard_sl_price')

                    # 修正「最高價幻覺」：使用實際可出場的 best_bid 來更新最高價，而非被瞬間拉高卻無法成交的 last_price (外盤價)
                    highest_price = max(pos_manager.get('highest_price_since_entry'), best_bid)

                    # 計算動態高點回檔停利 (Trailing Stop)
                    # 邏輯：如果最高獲利超過 15 點，則從最高點回檔 22% 或是固定回檔 13 點就停利出場，確保獲利落袋
                    profit_points = highest_price - entry_price
                    trailing_sl = hard_sl_price
                    # 只要帳面獲利超過 8 點，就啟動追蹤與保本機制
                    if profit_points >= 15:
                        if profit_points >= 35:
                            # 獲利超過 35 點後：容許較大回檔 (例如 12 點 或 獲利的 24%)
                            pullback = max(12, profit_points * 0.24)
                        else:
                            # 獲利 15~34.9 點階段：容許小回檔 (例如 7 點 或 獲利的 19%)
                            pullback = max(7, profit_points * 0.19)

                        # 動態上移防守線
                        trailing_sl = max(hard_sl_price, highest_price - pullback)

                        # 強制保本鎖：一旦獲利超過 15 點，防守線「絕對不能」低於進場價 + 4.0 點 (手續費滑價)
                        trailing_sl = max(trailing_sl, entry_price + 4.0)

                    if highest_price > pos_manager.get('highest_price_since_entry'):
                        pos_manager.update(highest_price_since_entry=highest_price)

                    exit_reason = None
                    
                    # 價格反曲點突破反轉平倉邏輯 (Inflection Breakout Reversal)
                    inf_break = df['inflection_breakout'].iloc[-1] if 'df' in locals() and not df.empty and 'inflection_breakout' in df.columns else 0
                    
                    if is_eod_closing_time(now, is_settlement=is_today_settlement):
                        exit_reason = "尾盤強制平倉 (EOD)"
                    elif position == 1 and inf_break == -1:
                        exit_reason = f"📉 跌破前次反曲低點 ({df['recent_trough_val'].iloc[-1]:.1f})，多單趨勢反轉出場"
                    elif position == -1 and inf_break == 1:
                        exit_reason = f"📈 突破前次反曲高點 ({df['recent_peak_val'].iloc[-1]:.1f})，空單趨勢反轉出場"
                    elif exec_price <= trailing_sl and trailing_sl > hard_sl_price:
                        exit_reason = f"觸及高點回檔動態停利 ({trailing_sl:.1f})"
                    elif exec_price <= hard_sl_price:
                        exit_reason = f"觸及原始動態停損 ({hard_sl_price})"
                    elif exec_price >= hard_tp_price:
                        exit_reason = f"觸及動態停利 ({hard_tp_price})"

                    if exit_reason:
                        last_trade_closed_time = now
                        last_trade_symbol = current_active_code

                        num_contracts = pos_manager.get('num_contracts')
                        trade_capital_used = pos_manager.get('trade_capital_used')
                        points_gained = exec_price - entry_price
                        current_pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - (num_contracts * FEE_SLIPPAGE_PER_CONTRACT)
                        current_ret = current_pnl / trade_capital_used if trade_capital_used > 0 else 0

                        last_trade_pnl = current_pnl
                        current_capital += current_pnl
                        last_trade_win = current_pnl > 0
                        trade_direction = 'Call' if position == 1 else 'Put'

                        if current_pnl < 0:
                            consecutive_failures[trade_direction] += 1
                            consecutive_total_losses += 1
                            if consecutive_total_losses >= 4:
                                cooldown_until = now + timedelta(minutes=15)
                                print(f"\n{'='*40}\n❄️ 【系統冷卻】連續 4 次虧損，暫停交易 15 分鐘！預計於 {cooldown_until.strftime('%H:%M:%S')} 恢復。\n{'='*40}\n")
                        else:
                            consecutive_failures[trade_direction] = 0
                            consecutive_total_losses = 0

                        trade_record = {
                            'entry_time': pos_manager.get('entry_time'),
                            'exit_time': now.strftime("%Y-%m-%d %H:%M:%S"),
                            'symbol': symbol,
                            'direction': trade_direction,
                            'entry_price': entry_price,
                            'exit_price': exec_price,
                            'pnl': current_pnl,
                            'ret': current_ret,
                            'reason': exit_reason
                        }
                        if entry_features:
                            trade_record.update(entry_features)

                        trade_log.append(trade_record)

                        # 即時備份交易紀錄，防止盤中當機遺失
                        try:
                            rt_dir = "data_learn"
                            if not os.path.exists(rt_dir): os.makedirs(rt_dir)
                            rt_csv_path = os.path.join(rt_dir, f"realtime_trade_report_{now.strftime('%Y%m%d')}.csv")
                            pd.DataFrame(trade_log).to_csv(rt_csv_path, index=False, encoding="utf-8-sig")
                        except Exception as save_err:
                            print(f"⚠️ 即時存檔交易紀錄失敗: {save_err}")

                        print(f"\n{'='*40}\n🔔 【平倉】: {exit_reason}\n實際盈虧: NT$ {current_pnl:,.0f} ({current_ret * 100:.2f}%)\n最新資金: NT$ {current_capital:,.0f}\n{'='*40}\n")
                        pos_manager.clear_position()
                        current_active_code = None
                        entry_features = {}

        except queue.Empty:
            position = pos_manager.get('position')
            if position != 0:
                if (now - last_print_time).total_seconds() >= 10:
                    last_print_time = now
                    sym = pos_manager.get('active_contract_symbol')
                    quote_data = opt_quotes.get(current_active_code, {}) if current_active_code else {}
                    price = quote_data.get('price', 0)

                    if price > 0:
                        entry_price = pos_manager.get('entry_price')
                        num_contracts = pos_manager.get('num_contracts')
                        hard_sl = pos_manager.get('hard_sl_price')
                        highest = pos_manager.get('highest_price_since_entry')

                        profit_points = highest - entry_price
                        trailing_sl = hard_sl
                        if profit_points >= 15:
                            pullback = max(10, profit_points * 0.3)
                            trailing_sl = max(hard_sl, highest - pullback)

                        points_gained = price - entry_price
                        current_pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - (num_contracts * FEE_SLIPPAGE_PER_CONTRACT)
                        print(f"[{time_str}] 持倉: {sym} | 買入價: {entry_price} | 現價: {price} | 最高: {highest} | 防守價(SL/TSL): {trailing_sl:.1f} | 停利: {pos_manager.get('hard_tp_price')} | 帳面損益: {current_pnl:,.0f}")
            else:
                if (now - heartbeat_time).total_seconds() >= 60:
                    heartbeat_time = now
                    if is_market_open(current_time):
                        print(f"[{time_str}] ⏳ 系統監控中... 等待交易訊號 (目前台指期現價: {current_txf_price})")
                    else:
                        print(f"[{time_str}] 💤 非交易時段，等待日盤開盤 (08:45)...")
        except Exception as e:
            print(f"[{time_str}] ❌ 主迴圈異常錯誤: {e}")

if __name__ == "__main__":
    run_live_simulator()
