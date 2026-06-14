import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

class PostMarketExporter:
    """
    負責日盤結束後的資料匯出、視覺化與狀態確認機制。
    符合嚴格品質與測試要求：極端防呆、時間序列排序、動態縮放繪圖。
    提供雙版本輸出：Human (Markdown, PNG) 與 CLI (JSONL, CSV).
    """
    def __init__(self, export_dir="reports_output/post_market"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)
        
    def _safe_timestamp(self, ts):
        if pd.isna(ts): return None
        if isinstance(ts, pd.Timestamp): return ts.isoformat()
        if isinstance(ts, datetime): return ts.isoformat()
        return str(ts)

    def _sanitize_dataframe(self, df, time_col='date'):
        """極端防呆機制：處理 NaN、排序時間、處理空資料"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        df_clean = df.copy()
        if time_col in df_clean.columns:
            # 確保嚴格的時間序列排序 (Strict Time Sequence)
            df_clean[time_col] = pd.to_datetime(df_clean[time_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[time_col]) # Drop invalid times
            df_clean = df_clean.sort_values(time_col).reset_index(drop=True)
        
        # 處理數值 NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(0)
        return df_clean

    def export_cli_version(self, df_market, dict_options_history, trade_log, date_str):
        """
        給 CLI 看的版本（自動化與極致結構化）
        1. 純文字格式（標準 JSONL）
        2. 高效讀取、時間戳排序、無 UI 字元
        """
        df_market = self._sanitize_dataframe(df_market)
        
        # 4.1 當天日盤市況資料（現貨與期權 - 主市場特徵）
        market_file = os.path.join(self.export_dir, f"cli_market_data_{date_str}.jsonl")
        if not df_market.empty:
            if 'date' in df_market.columns:
                df_market['timestamp'] = df_market['date'].apply(self._safe_timestamp)
                df_market['date'] = df_market['date'].astype(str) # 解決 Timestamp serializable 問題
            records = df_market.to_dict(orient='records')
            with open(market_file, 'w', encoding='utf-8') as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')

        # 4.2 實際交易之選擇權合約資料
        options_file = os.path.join(self.export_dir, f"cli_options_data_{date_str}.jsonl")
        with open(options_file, 'w', encoding='utf-8') as f:
            for sym, df_opt in dict_options_history.items():
                df_o = self._sanitize_dataframe(df_opt)
                if not df_o.empty:
                    if 'date' in df_o.columns:
                        df_o['timestamp'] = df_o['date'].apply(self._safe_timestamp)
                        df_o['date'] = df_o['date'].astype(str)
                    df_o['symbol'] = sym
                    for r in df_o.to_dict(orient='records'):
                        f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
                        
        # 4.3 單筆交易區間詳細紀錄
        # 包含：AI Strategy, TP/SL, Entry Time, Exit Time, All Feature Values
        trades_file = os.path.join(self.export_dir, f"cli_trade_log_{date_str}.jsonl")
        with open(trades_file, 'w', encoding='utf-8') as f:
            for t in trade_log:
                f.write(json.dumps(t, ensure_ascii=False, default=str) + '\n')

    def export_human_version(self, df_market, dict_options_history, trade_log, date_str):
        """
        給人看的版本（高精準視覺化與易讀性）
        1. 視覺化繪圖 (Matplotlib)
        2. Markdown 報表
        """
        df_market = self._sanitize_dataframe(df_market)
        
        # Markdown 報表
        md_file = os.path.join(self.export_dir, f"human_report_{date_str}.md")
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# 日盤當沖交易結算報告 - {date_str}\n\n")
            
            f.write("## 1. 交易紀錄彙總\n")
            if not trade_log:
                f.write("本日無交易紀錄。\n")
            else:
                total_pnl = sum([t.get('pnl', 0) for t in trade_log])
                f.write(f"**本日總損益 (PnL):** {total_pnl:,.0f} 元\n\n")
                for i, t in enumerate(trade_log):
                    f.write(f"### Trade {i+1}: {t.get('symbol', 'Unknown')}\n")
                    f.write(f"- **策略 (AI Strategy):** {t.get('strategy_label', 'N/A')}\n")
                    f.write(f"- **方向 (Direction):** {'做多 (Long)' if t.get('direction') == 1 else '做空 (Short)'}\n")
                    f.write(f"- **進場時間 (Entry Time):** {t.get('entry_time', 'N/A')} @ {t.get('entry_price', 0)}\n")
                    f.write(f"- **出場時間 (Exit Time):** {t.get('exit_time', 'N/A')} @ {t.get('exit_price', 0)}\n")
                    f.write(f"- **停利點 (Take-Profit):** {t.get('hard_tp_price', 'N/A')}\n")
                    f.write(f"- **停損點 (Stop-Loss):** {t.get('hard_sl_price', 'N/A')}\n")
                    f.write(f"- **最終損益 (PnL):** {t.get('pnl', 0)}\n\n")

            f.write("## 2. 歷史行情與特徵概況\n")
            f.write(f"- **主市場資料筆數:** {len(df_market)}\n")
            f.write(f"- **涉及選擇權合約數:** {len(dict_options_history)}\n")

        # 視覺化繪圖：圖表精準與動態縮放
        if not df_market.empty and 'date' in df_market.columns:
            try:
                fig, ax = plt.subplots(figsize=(14, 7))
                
                ax.plot(df_market['date'], df_market['Close'], label='Market Close Price', color='#1f77b4', linewidth=1.5)
                
                # 標示進出場與停利停損對齊線
                y_min, y_max = df_market['Close'].min(), df_market['Close'].max()
                
                for t in trade_log:
                    entry_time = pd.to_datetime(t.get('entry_time', None), errors='coerce')
                    exit_time = pd.to_datetime(t.get('exit_time', None), errors='coerce')
                    
                    if pd.notna(entry_time):
                        entry_price = t.get('entry_price')
                        direction = t.get('direction', 1)
                        tp_price = t.get('hard_tp_price')
                        sl_price = t.get('hard_sl_price')
                        
                        # 標示進場
                        color = 'green' if direction == 1 else 'red'
                        marker = '^' if direction == 1 else 'v'
                        ax.scatter(entry_time, entry_price, color=color, marker=marker, s=150, zorder=5, label='Entry' if 'Entry' not in ax.get_legend_handles_labels()[1] else "")
                        
                        if pd.notna(exit_time):
                            exit_price = t.get('exit_price')
                            ax.scatter(exit_time, exit_price, color='blue', marker='X', s=150, zorder=5, label='Exit' if 'Exit' not in ax.get_legend_handles_labels()[1] else "")
                            # 繪製進出場之間的虛線
                            ax.plot([entry_time, exit_time], [entry_price, exit_price], linestyle='--', color='gray', alpha=0.7)
                        
                        # 動態縮放 Y 軸，包容 TP / SL (如果存在)
                        if tp_price and tp_price > 0:
                            y_max = max(y_max, tp_price)
                            y_min = min(y_min, tp_price)
                            ax.hlines(y=tp_price, xmin=entry_time, xmax=exit_time if pd.notna(exit_time) else df_market['date'].iloc[-1], colors='green', linestyles='dotted', alpha=0.5)
                        if sl_price and sl_price > 0:
                            y_max = max(y_max, sl_price)
                            y_min = min(y_min, sl_price)
                            ax.hlines(y=sl_price, xmin=entry_time, xmax=exit_time if pd.notna(exit_time) else df_market['date'].iloc[-1], colors='red', linestyles='dotted', alpha=0.5)
                            
                # 動態縮放與防呆
                if y_min == y_max:
                    y_min, y_max = y_min * 0.99, y_max * 1.01
                margin = (y_max - y_min) * 0.05
                ax.set_ylim(y_min - margin, y_max + margin)
                
                # 時間軸格式
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                fig.autofmt_xdate()
                
                ax.set_title(f"Market Price and Trading Execution - {date_str}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Price")
                ax.legend(loc='best')
                ax.grid(True, linestyle='--', alpha=0.6)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.export_dir, f"human_chart_{date_str}.png"), dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"⚠️ 圖表繪製失敗: {e}")

    def execute_export(self, df_market, dict_options_history, trade_log):
        """主入口"""
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"🚀 開始匯出盤後資料... (批次 {date_str})")
        self.export_cli_version(df_market, dict_options_history, trade_log, date_str)
        self.export_human_version(df_market, dict_options_history, trade_log, date_str)
        print(f"✅ 匯出完成，檔案儲存於 {self.export_dir}")

