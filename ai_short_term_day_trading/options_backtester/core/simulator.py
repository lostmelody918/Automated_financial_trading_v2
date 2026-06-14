import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import torch
import traceback
from dotenv import load_dotenv

# Append paths
core_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(core_dir) # options_backtester
ai_dir = os.path.dirname(base_dir) # ai_short_term_day_trading
project_root = os.path.dirname(ai_dir)

sys.path.append(base_dir)
sys.path.append(ai_dir)

# Load .env variables
load_dotenv(os.path.join(project_root, '.env'))

from database.timescale_client import TimescaleDBClient

# Import AI modules
try:
    from data_engine import DayTradingDataEngine
    from composite_ai import CompositeDayTradingAI
    from model_manager import TradingModelManager
    from strategy_factory import StrategyFactory
    from delta_gamma_theta import get_dynamic_bsm_bounds, calculate_bs_greeks
except ImportError as e:
    print(f"Error importing AI modules: {e}")
    traceback.print_exc()

# Import C++ or Fallback Engine
try:
    build_dir = os.path.join(base_dir, 'cpp_engine', 'build')
    sys.path.append(build_dir)
    import options_replay
    print("✅ Successfully loaded high-performance C++ Order Book Replay Engine.")
except ImportError:
    print("⚠️ Warning: options_replay C++ module not found. Falling back to Python mock engine.")
    import options_replay_fallback as options_replay

class OptionsSimulator:
    def __init__(self, target_date: str):
        self.target_date = target_date
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            print("Warning: DATABASE_URL not found in .env, falling back to default.")
            db_url = "postgresql://postgres:postgres@localhost:5432/finance_db"
        self.db = TimescaleDBClient(db_url)
        self.engine = options_replay.SimulationEngine() if options_replay else None
        
        self._init_ai()
        
    def _init_ai(self):
        print("🤖 Initializing AI Model and Strategy Factory...")
        # Load Norm Params
        norm_path = os.path.join(ai_dir, "saved_models", "norm_params.json")
        if os.path.exists(norm_path):
            with open(norm_path, 'r', encoding='utf-8') as f:
                self.norm_params = json.load(f)
        else:
            print("❌ norm_params.json not found!")
            self.norm_params = None

        if self.norm_params:
            self.feature_cols = [c for c in self.norm_params['feature_cols'] if c in self.norm_params['mean']]
            self.mean_v = np.array([self.norm_params['mean'][c] for c in self.feature_cols])
            self.std_v = np.array([self.norm_params['std'][c] for c in self.feature_cols])
        
            self.ai_model = CompositeDayTradingAI(input_dim=len(self.feature_cols), d_model=256, nhead=16, num_layers=4)
            model_manager = TradingModelManager(model_dir=os.path.join(ai_dir, "saved_models"))
            self.ai_model, _, _ = model_manager.load_latest_model(self.ai_model)
            self.ai_model.eval()
        
        self.strategy_engine = StrategyFactory.get_strategy("composite")
        
        print("📡 Fetching historical features from Shioaji Data Engine...")
        data_engine = DayTradingDataEngine()
        df_all = data_engine.fetch_intraday_data(days=15)
        
        # Filter for the target date
        if not df_all.empty:
            df_all['date_str'] = df_all['date'].dt.strftime('%Y-%m-%d')
            self.df_intraday = df_all[df_all['date_str'] == self.target_date].reset_index(drop=True)
            print(f"📊 Extracted {len(self.df_intraday)} K-bars for {self.target_date}.")
        else:
            self.df_intraday = pd.DataFrame()
            print("⚠️ Data engine returned empty dataframe.")

    def find_top_volume_contracts(self) -> dict:
        start_time = f"{self.target_date} 08:45:00"
        end_time = f"{self.target_date} 13:45:00"
        
        try:
            if not self.db.conn:
                self.db.connect()
            query = """
                SELECT symbol, SUM(volume) as total_vol 
                FROM options_ticks 
                WHERE time >= %s AND time <= %s 
                GROUP BY symbol 
                ORDER BY total_vol DESC
            """
            df_symbols = pd.read_sql_query(query, self.db.conn, params=[start_time, end_time])
            
            if not df_symbols.empty:
                calls = df_symbols[df_symbols['symbol'].str.contains('C|c')]
                puts = df_symbols[df_symbols['symbol'].str.contains('P|p')]
                
                call_list = [{'symbol': row['symbol'], 'volume': int(row['total_vol'])} for _, row in calls.head(4).iterrows()]
                put_list = [{'symbol': row['symbol'], 'volume': int(row['total_vol'])} for _, row in puts.head(4).iterrows()]
                
                return {
                    'calls': call_list,
                    'puts': put_list
                }
        except Exception as e:
            print(f"Failed to query distinct symbols: {e}")
            
        # Fallback: dynamically estimate ATM contracts based on intraday features
        base_strike = 15000
        if hasattr(self, 'df_intraday') and not self.df_intraday.empty:
            open_price = self.df_intraday.iloc[0]['Close']
            # Round to nearest 100
            base_strike = int(round(open_price / 100.0) * 100)
            
        c_strikes = [base_strike - 100, base_strike, base_strike + 100, base_strike + 200]
        p_strikes = [base_strike + 100, base_strike, base_strike - 100, base_strike - 200]
            
        # Mock volumes for fallback
        vols = [15000, 12000, 8000, 4000]
        return {
            'calls': [{'symbol': f'TXO{s}C', 'volume': v} for s, v in zip(c_strikes, vols)],
            'puts': [{'symbol': f'TXO{s}P', 'volume': v} for s, v in zip(p_strikes, vols)]
        }

    def load_ticks_to_engine(self, symbols: list):
        start_time = f"{self.target_date} 08:45:00"
        end_time = f"{self.target_date} 13:45:00"
        
        try:
            df = self.db.fetch_ticks(start_time, end_time, symbols)
        except Exception as e:
            print(f"Database fetch failed: {e}. Generating mock data for testing.")
            times = pd.date_range(start=start_time, end=end_time, freq="1s")
            dfs = []
            for sym in symbols:
                # 簡單 mock: 選擇權價格 100 左右
                sym_df = pd.DataFrame({
                    'time': times,
                    'symbol': sym,
                    'price': np.random.normal(0, 0.1, len(times)).cumsum() + 100,
                    'volume': np.random.randint(1, 5, len(times)),
                })
                sym_df['bid_price'] = sym_df['price'] - 1.0
                sym_df['ask_price'] = sym_df['price'] + 1.0
                sym_df['bid_volume'] = 10
                sym_df['ask_volume'] = 10
                dfs.append(sym_df)
            df = pd.concat(dfs, ignore_index=True)
        
        if df.empty or self.engine is None:
            return df
            
        for symbol in symbols:
            sym_df = df[df['symbol'] == symbol]
            ticks = []
            for _, row in sym_df.iterrows():
                t = options_replay.Tick()
                t.timestamp_ms = int(row['time'].timestamp() * 1000)
                t.price = row['price']
                t.volume = row['volume']
                t.bid_price = row['bid_price']
                t.bid_volume = row['bid_volume']
                t.ask_price = row['ask_price']
                t.ask_volume = row['ask_volume']
                ticks.append(t)
            self.engine.feed_ticks(symbol, ticks)
            
        return df

    def extract_strike_from_symbol(self, symbol: str):
        import re
        match = re.search(r'\d+', symbol)
        if match:
            return float(match.group())
        return 15000.0

    def run_simulation(self):
        if not self.engine or self.df_intraday.empty:
            print("Cannot run simulation. Missing C++ engine or intraday features.")
            return [], pd.DataFrame(), {}, pd.DataFrame()
            
        top_contracts = self.find_top_volume_contracts()
        all_symbols = [c['symbol'] for c in top_contracts['calls']] + [p['symbol'] for p in top_contracts['puts']]
        if not all_symbols:
            print("No contracts found to simulate.")
            return [], self.df_intraday, {}, pd.DataFrame()
            
        df_ticks = self.load_ticks_to_engine(all_symbols)
        
        start_ts = int(pd.to_datetime(f"{self.target_date} 08:45:00").timestamp() * 1000)
        end_ts = int(pd.to_datetime(f"{self.target_date} 13:45:00").timestamp() * 1000)
        
        print(f"\n🚀 Starting Live-Logic Options Backtest from {start_ts} to {end_ts}")
        
        current_ts = start_ts
        
        # Position State
        position = 0
        active_symbol = None
        entry_price = 0.0
        hard_sl_price = 0.0
        hard_tp_price = 0.0
        pnl = 0.0
        trade_log = []
        last_trade_win = False
        
        # Time Management
        # Pre-process the intraday df so we can quickly lookup features by minute
        self.df_intraday['time_ms'] = self.df_intraday['date'].apply(lambda x: int(x.timestamp() * 1000))
        
        while current_ts <= end_ts:
            self.engine.advance_to(current_ts)
            current_dt = pd.to_datetime(current_ts, unit='ms')
            
            # Check minute boundary to evaluate AI
            if current_ts % 60000 == 0:
                # Get features up to current minute
                df_slice = self.df_intraday[self.df_intraday['time_ms'] <= current_ts]
                
                if len(df_slice) >= 40: # WINDOW_SIZE = 40
                    df_window = df_slice.tail(40).copy()
                    
                    for missing_col in self.feature_cols:
                        if missing_col not in df_window.columns:
                            df_window[missing_col] = 0.0

                    df_window = df_window[self.feature_cols].copy()
                    
                    # Normalize (Same as live_option_simulator_v2.py)
                    for col in ['mock_volume', 'macd_hist', 'vwap_bias']:
                        if col in df_window.columns:
                            df_window[col] = np.sign(df_window[col]) * np.log1p(np.abs(df_window[col]))

                    feat_tensor = torch.tensor(np.nan_to_num((df_window.values - self.mean_v) / np.where(self.std_v == 0, 1.0, self.std_v), nan=0.0), dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        probs = torch.softmax(self.ai_model(feat_tensor), dim=1).squeeze().cpu().numpy()
                    
                    # Strategy Evaluation
                    signal = self.strategy_engine.generate_signal(df_slice, ai_score=probs, last_win=last_trade_win)
                    
                    # Check Exit (EOD or SL/TP)
                    if position != 0:
                        current_bid = self.engine.get_best_bid(active_symbol)
                        current_ask = self.engine.get_best_ask(active_symbol)
                        current_opt_price = current_bid if position > 0 else current_ask
                        
                        exit_reason = None
                        if current_dt.time() >= datetime.strptime("13:30:00", "%H:%M:%S").time():
                            exit_reason = "EOD Force Close"
                        elif current_opt_price <= hard_sl_price:
                            exit_reason = f"SL Triggered ({hard_sl_price})"
                        elif current_opt_price >= hard_tp_price:
                            exit_reason = f"TP Triggered ({hard_tp_price})"
                            
                        if exit_reason:
                            exit_price = current_opt_price
                            points = exit_price - entry_price if position > 0 else entry_price - exit_price
                            trade_pnl = points * 50
                            pnl += trade_pnl
                            last_trade_win = trade_pnl > 0
                            time_str = current_dt.strftime('%H:%M:%S')
                            print(f"[{time_str}] 🔴 EXIT {active_symbol}: Sold at {exit_price} | Reason: {exit_reason} | PnL: NT$ {trade_pnl:,.0f}")
                            
                            trade_log.append({
                                'entry_time': entry_time,
                                'exit_time': current_dt.strftime('%H:%M:%S'),
                                'symbol': active_symbol,
                                'type': 'Call' if 'C' in active_symbol else 'Put',
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl': trade_pnl
                            })
                            position = 0
                            active_symbol = None
                    
                    # Check Entry
                    if position == 0 and signal != 0 and current_dt.time() < datetime.strptime("13:25:00", "%H:%M:%S").time():
                        opt_type = 'Call' if signal > 0 else 'Put'
                        symbol_list = top_contracts['calls'] if opt_type == 'Call' else top_contracts['puts']
                        active_symbol = symbol_list[0] if symbol_list else None
                        
                        if active_symbol:
                            ask_price = self.engine.get_best_ask(active_symbol)
                            if ask_price > 0:
                                position = 1 if signal > 0 else -1
                                entry_price = ask_price
                                entry_time = current_dt.strftime('%H:%M:%S')
                                
                                # Use Dynamic BSM for Bounds
                                S = df_slice.iloc[-1]['Close']
                                K = self.extract_strike_from_symbol(active_symbol)
                                T = 2.0 / 365.0 # Mock DTE
                                r = 0.015
                                iv = 0.20
                                atr = df_slice.iloc[-1].get('atr', 20.0)
                                tp_mult, sl_mult = 2.0, 1.0
                                
                                bounds = get_dynamic_bsm_bounds(S, K, T, r, iv, atr, tp_mult, sl_mult, 2.0, option_type=opt_type, actual_entry_price=entry_price)
                                hard_tp_price, hard_sl_price, _, _, _ = bounds
                                
                                print(f"[{entry_time}] 🟢 ENTER {opt_type}: Bought 1 {active_symbol} at {entry_price} | SL: {hard_sl_price:.1f}, TP: {hard_tp_price:.1f}")
                                
            current_ts += 1000 # Step 1 second
            
        print("\n" + "="*50)
        print("📊 Backtest Simulation Complete")
        print(f"💰 Total PnL: NT$ {pnl:,.0f}")
        print("="*50 + "\n")
        
        return trade_log, self.df_intraday, top_contracts, df_ticks

if __name__ == "__main__":
    sim = OptionsSimulator("2026-06-11")
    sim.run_simulation()
