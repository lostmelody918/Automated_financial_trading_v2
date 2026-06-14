import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# 解決 matplotlib 中文字體顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] # 優先使用微軟正黑體
plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號
import torch
from data_engine import DayTradingDataEngine
from strategy_factory import StrategyFactory
from composite_ai import CompositeDayTradingAI
from model_manager import TradingModelManager
from delta_gamma_theta import calculate_bs_greeks, get_dynamic_bsm_bounds

def save_trade_plot(df, entry_idx, exit_idx, trade_type, ret, trade_id, entry_features=None, trade_capital=0, position_dir=1):
    """助手函數：繪製單筆交易的波段圖，並標註特徵與儲存資料"""
    try:
        # 動態調整 X 軸區間 (前後多抓一些 K 線)
        duration = exit_idx - entry_idx
        padding = max(10, int(duration * 0.5))
        start_plot_idx = max(0, entry_idx - padding)
        end_plot_idx = min(len(df)-1, exit_idx + padding)
        plot_df = df.iloc[start_plot_idx:end_plot_idx+1].copy()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_df['date'], plot_df['Close'], color='gray', alpha=0.5, label='Price')

        # 動態調整 Y 軸區間，避免變成直線
        y_min = plot_df['Low'].min()
        y_max = plot_df['High'].max()
        y_padding = (y_max - y_min) * 0.1
        if y_padding == 0:
            y_padding = y_max * 0.05
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        entry_row = df.iloc[entry_idx]
        exit_row = df.iloc[exit_idx]

        ax.scatter(entry_row['date'], entry_row['Close'], color='blue', marker='^', s=100, label='Entry')
        ax.scatter(exit_row['date'], exit_row['Close'], color='red', marker='v', s=100, label='Exit')

        ax.plot([entry_row['date'], exit_row['date']], [entry_row['Close'], exit_row['Close']],
                 color='green' if ret > 0 else 'red', linestyle='--', alpha=0.6)

        pnl_amount = trade_capital * ret
        dir_str = "LONG" if position_dir == 1 else "SHORT"
        title_str = f"Trade #{trade_id} | {dir_str} | {trade_type} | Ret: {ret*100:.2f}% | Cap: NT${trade_capital:,.0f} | PnL: NT${pnl_amount:,.0f}"
        ax.set_title(title_str)

        # 新增明顯的圖表內標示：做多/做空、交易本金、獲利/虧損金額
        info_text = f"方向 (Direction): {dir_str}\n本金 (Capital): NT${trade_capital:,.0f}\n損益 (PnL): NT${pnl_amount:,.0f}"
        props = dict(boxstyle='round', facecolor='white' if ret > 0 else 'mistyrose', alpha=0.9, edgecolor='gray')
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, color='green' if ret > 0 else 'red', weight='bold')

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plots_dir = os.path.join(os.path.dirname(__file__), "data_learn", "trade_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 在圖表上加上特徵文字
        if entry_features:
            feature_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in entry_features.items() if k != 'entry_date'])
            plt.gcf().text(0.02, 0.5, feature_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

        filename_base = f"trade_{trade_id:03d}_{trade_type}_{'WIN' if ret > 0 else 'LOSS'}"
        plt.tight_layout(rect=[0.15, 0, 1, 1]) # 留空間給左側文字
        plt.savefig(os.path.join(plots_dir, f"{filename_base}.png"))
        plt.close()

        # 儲存詳細資料成 txt
        if entry_features:
            with open(os.path.join(plots_dir, f"{filename_base}.txt"), 'w', encoding='utf-8') as f:
                f.write(f"Trade ID: {trade_id}\n")
                f.write(f"Type: {trade_type}\n")
                f.write(f"Return: {ret*100:.2f}%\n")
                f.write("-" * 20 + "\n")
                for k, v in entry_features.items():
                    f.write(f"{k}: {v}\n")
    except Exception as e:
        print(f"Plot saving failed: {e}")

def run_advanced_simulator(initial_capital=100000, days=5):
    engine = DayTradingDataEngine()
    df_raw = engine.fetch_intraday_data(days=days)
    df_chips = engine.fetch_real_historical_chips(days=days + 15)

    if df_raw.empty or df_chips.empty:
        print("沒有數據。")
        return

    df = engine.integrate_institutional_chips(df_raw, df_chips)

    if df.empty:
        print("融合後沒有數據。")
        return

    norm_path = os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json")
    if os.path.exists(norm_path):
        import json
        with open(norm_path, 'r') as f:
            norm_params = json.load(f)

        # 過濾掉無法正規化的欄位 (如 date_x, date_y)
        valid_cols = [c for c in norm_params['feature_cols'] if c in norm_params['mean'] and c in df.columns]
        norm_params['feature_cols'] = valid_cols

        mean_v = np.array([norm_params['mean'][c] for c in valid_cols])
        std_v = np.array([norm_params['std'][c] for c in valid_cols])
        input_dim = len(valid_cols)
    else:
        mean_v, std_v = 0, 1
        feature_cols = [c for c in df.columns if c not in ['date', 'time', 'date_only', 'day_of_week', 'date_x', 'date_y']]
        input_dim = len(feature_cols)

    # 必須與 train_model.py 的參數一致
    ai_model = CompositeDayTradingAI(input_dim=input_dim, d_model=256, nhead=16, num_layers=4)
    optimizer = torch.optim.Adam(ai_model.parameters(), lr=0.001)
    model_manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))

    ai_model, optimizer, current_version = model_manager.load_latest_model(ai_model, optimizer)
    ai_model.eval()

    strategy_engine = StrategyFactory.get_strategy("composite")

    # 選擇權實務模式：無額外槓桿，依照選擇權真實點數與乘數計算損益
    LEVERAGE = 1
    # 選擇權一口交易成本約 100 元 (手續費 + 滑價約 1 tick)
    CONTRACT_MULTIPLIER = 50
    FEE_SLIPPAGE_PER_CONTRACT = 100
    MAX_POSITION_CAPITAL = 4000000

    def get_dynamic_cost(num_contracts):
        return num_contracts * FEE_SLIPPAGE_PER_CONTRACT

    print(f"\n📊 啟動【選擇權實務模式】AI 突破推理 & 交易波段自動繪圖模擬器")
    print(f"💵 初始本金: NT$ {initial_capital:,} | 選擇權乘數: {CONTRACT_MULTIPLIER}")

    # 控制是否要儲存交易圖表，設為 True 會大幅拖慢回測速度 (若交易次數破千)
    SAVE_PLOTS = False
    if not SAVE_PLOTS:
        print("⚡ 已關閉單筆交易波段圖繪製 (SAVE_PLOTS=False) 以加速回測運行。")

    trade_log = []
    position = 0
    entry_price = 0
    entry_idx = 0
    current_entry_features = {}
    current_capital = initial_capital
    capital_curve = [initial_capital]
    trade_capital_used = 0
    num_contracts = 0
    WINDOW_SIZE = 40

    # 連續波段與追蹤停損變數
    last_trade_win = False
    is_scalp = False
    highest_price_since_entry = 0.0
    hard_tp_price = 0.0
    hard_sl_price = 0.0

    total_bars = len(df)
    for i in range(WINDOW_SIZE, total_bars-1):
        if i % 2000 == 0:
            print(f"⏳ 回測進度: {i} / {total_bars} ({i/total_bars*100:.1f}%) | 當前本金: {current_capital:,.0f}")

        curr_slice = df.iloc[:i+1]
        last_row = curr_slice.iloc[-1]
        next_row = df.iloc[i+1]
        curr_time = last_row['date'].time()

        # AI 推理
        feat_data = curr_slice[norm_params['feature_cols']].tail(WINDOW_SIZE).values
        feat_normalized = (feat_data - mean_v) / std_v
        feat_tensor = torch.tensor(feat_normalized, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = ai_model(feat_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # 1. 平倉邏輯
        if position != 0:
            S_high = next_row['High']
            S_low = next_row['Low']
            S_close = next_row['Close']

            K = current_entry_features['K']
            T = current_entry_features['T']
            opt_type = current_entry_features['opt_type']

            # 使用 BSM 轉換最高最低指數價為權利金 (這是一種粗估，實務上還要考慮動態時間與波動率)
            _, _, _, opt_price_at_high = calculate_bs_greeks(S_high, K, T, 0.015, 0.22, opt_type)
            _, _, _, opt_price_at_low = calculate_bs_greeks(S_low, K, T, 0.015, 0.22, opt_type)
            _, _, _, opt_price_close = calculate_bs_greeks(S_close, K, T, 0.015, 0.22, opt_type)

            # 對 Call 而言，指數越高選擇權越高；對 Put 而言，指數越低選擇權越高
            if position == 1:
                opt_high = opt_price_at_high
                opt_low = opt_price_at_low
            else:
                opt_high = opt_price_at_low
                opt_low = opt_price_at_high

            # 追蹤歷史最高權利金以啟動 Trailing Stop
            if opt_high > highest_price_since_entry:
                highest_price_since_entry = opt_high

            exit_reason = None
            exec_price = 0.0

            # === 極短線 (Scalping) 出場邏輯 ===
            if is_scalp:
                bars_held = (i + 1) - entry_idx
                is_reverse_k = (next_row['Close'] < next_row['Open']) if position == 1 else (next_row['Close'] > next_row['Open'])
                rsi_exit_long = position == 1 and last_row['rsi_fast'] >= 80 and is_reverse_k
                rsi_exit_short = position == -1 and last_row['rsi_fast'] <= 20 and is_reverse_k
                time_exit = bars_held >= 8

                if rsi_exit_long or rsi_exit_short or time_exit:
                    exit_reason = 'Scalp_RSI_Exit' if (rsi_exit_long or rsi_exit_short) else 'Scalp_Time_Exit'
                    exec_price = opt_price_close
                    is_scalp = False

            # === 波段/常規出場邏輯 ===
            if not exit_reason:
                is_next_day = next_row['date_only'] > current_entry_features['entry_date'].date()
                if is_next_day and curr_time.hour == 13 and curr_time.minute >= 25:
                    exit_reason = 'Close_Next_Day_EOD'
                    exec_price = opt_price_close

            if not exit_reason:
                # Trailing Stop 檢查
                profit_points = highest_price_since_entry - entry_price
                trailing_sl = hard_sl_price
                if profit_points >= 9:
                    pullback = max(13, profit_points * 0.22) if profit_points >= 20 else max(5, profit_points * 0.16)
                    trailing_sl = max(hard_sl_price, highest_price_since_entry - pullback)
                    trailing_sl = max(trailing_sl, entry_price + 1.0) # 強制保本

                if opt_low <= trailing_sl and trailing_sl > hard_sl_price:
                    exit_reason = 'Trailing_Stop'
                    exec_price = trailing_sl
                elif opt_low <= hard_sl_price:
                    exit_reason = 'Stop_Loss'
                    exec_price = hard_sl_price
                elif opt_high >= hard_tp_price:
                    exit_reason = 'Take_Profit'
                    exec_price = hard_tp_price

            if exit_reason:
                points_gained = exec_price - entry_price # 選擇權不分多空，獲利就是賣價減買價
                pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - get_dynamic_cost(num_contracts)
                net_ret = pnl / trade_capital_used if trade_capital_used > 0 else 0
                current_capital += pnl
                last_trade_win = pnl > 0

                trade_log.append({'date': next_row['date'], 'type': exit_reason, 'ret': net_ret, 'capital': current_capital, 'entry_features': current_entry_features})
                save_trade_plot(df, entry_idx, i+1, exit_reason, net_ret, len(trade_log), current_entry_features, trade_capital_used, position)

                position = 0
                is_scalp = False
                capital_curve.append(current_capital)
                continue

        capital_curve.append(current_capital)

        # 2. 進場邏輯
        signal = strategy_engine.generate_signal(curr_slice, ai_score=probs, last_win=last_trade_win)
        
        # 僅限日盤交易 (08:45 ~ 13:45)
        import datetime
        is_day_session = (datetime.time(8, 45) <= curr_time <= datetime.time(13, 45))
        
        if signal != 0 and position == 0 and is_day_session:
            is_scalp = (abs(signal) == 10)
            position = 1 if signal > 0 else -1

            # --- 使用 BSM 模擬真實選擇權權利金與風控點 ---
            S = next_row['Open']
            K = round(S / 50) * 50 # 取最接近的價平履約價
            T = 7 / 365.0 # 假設為近週選
            iv = 0.22
            opt_type = 'Call' if position == 1 else 'Put'

            _, _, _, simulated_entry_price = calculate_bs_greeks(S, K, T, 0.015, iv, opt_type)

            # 若計算出權利金小於 5 點，代表太過價外或模型偏差，強制棄單
            if simulated_entry_price < 5:
                position = 0
                continue

            entry_price = simulated_entry_price
            entry_idx = i + 1
            highest_price_since_entry = entry_price

            # 設定動態風控區間
            abs_sig = abs(signal)
            if abs_sig == 3:
                tp_mult, sl_mult = 5.0, 1.5
                expected_hold = 2.0
            elif abs_sig == 2:
                tp_mult, sl_mult = 3.0, 1.0
                expected_hold = 1.0
            else:
                tp_mult, sl_mult = 1.5, 0.5
                expected_hold = 0.5

            hard_tp_price, hard_sl_price, _, _, _ = get_dynamic_bsm_bounds(
                S=S, K=K, T=T, r=0.015, iv=iv, atr=last_row['atr'],
                tp_mult=tp_mult, sl_mult=sl_mult, expected_hold_hours=expected_hold,
                option_type=opt_type, actual_entry_price=entry_price
            )

            # 資金管理
            pos_size_pct = 0.50
            allocated_capital = min(current_capital * pos_size_pct, MAX_POSITION_CAPITAL)
            contract_cost = entry_price * CONTRACT_MULTIPLIER
            if contract_cost > 0:
                num_contracts = int(allocated_capital // contract_cost)
            else:
                num_contracts = 0

            if num_contracts < 1:
                num_contracts = 1

            trade_capital_used = num_contracts * contract_cost

            current_entry_features = {
                'entry_date': next_row['date'],
                'signal': signal,
                'opt_type': opt_type,
                'K': K,
                'T': T,
                'prob_down': probs[0],
                'prob_neutral': probs[1],
                'prob_up': probs[2]
            }
            # 儲存 AI 所看到的所有特徵
            for col in norm_params['feature_cols']:
                current_entry_features[col] = last_row[col]

    # 結算與報告
    df_trades = pd.DataFrame(trade_log)
    if df_trades.empty:
        print("沒有交易次數")
        return

    features_log = []
    for t in trade_log:
        if 'entry_features' in t:
            feat = t['entry_features'].copy()
            feat['exit_date'] = t['date']
            feat['trade_type'] = t['type']
            feat['ret'] = t['ret']
            features_log.append(feat)

    if features_log:
        df_features = pd.DataFrame(features_log)
        out_dir = os.path.join(os.path.dirname(__file__), "data_learn")
        try:
            df_features.to_csv(os.path.join(out_dir, "trade_features_log.csv"), index=False, encoding="utf-8-sig")
            print(f"💾 已儲存交易特徵日誌至 data_learn/trade_features_log.csv (共 {len(df_features)} 筆)")
        except PermissionError:
            print("⚠️ 無法儲存 trade_features_log.csv，檔案可能正被 Excel 開啟。")
        
        # --- 新增: 特徵最佳化分析報告 ---
        print("\n📊 --- 特徵最佳化分析報告 ---")
        win_trades = df_features[df_features['ret'] > 0]
        loss_trades = df_features[df_features['ret'] <= 0]
        if not win_trades.empty and not loss_trades.empty:
            analysis_lines = []
            for col in norm_params['feature_cols']:
                if col in df_features.columns:
                    win_mean = win_trades[col].mean()
                    loss_mean = loss_trades[col].mean()
                    diff_pct = (win_mean - loss_mean) / (abs(loss_mean) + 1e-9) * 100
                    if abs(diff_pct) > 10: # 只列出差異超過 10% 的特徵
                        analysis_lines.append(f"  - {col}: 獲利均值 {win_mean:.4f} | 虧損均值 {loss_mean:.4f} (差異 {diff_pct:+.1f}%)")
            
            if analysis_lines:
                print("💡 發現獲利與虧損交易在以下特徵有顯著差異 (可用於後續優化):")
                for line in analysis_lines:
                    print(line)
                
                with open(os.path.join(out_dir, "feature_optimization_report.txt"), 'w', encoding='utf-8') as f:
                    f.write("發現獲利與虧損交易在以下特徵有顯著差異 (可用於後續優化):\n")
                    f.write("\n".join(analysis_lines))
            else:
                print("💡 目前獲利與虧損交易在各特徵上的平均差異皆不明顯。")

    # 修正 weekly summary: 加入年份避免跨年週次重疊
    df_trades['year_week'] = df_trades['date'].dt.strftime('%Y-W%W')
    weekly_true_ret = {}
    weekly_log = []
    last_week_capital = initial_capital

    # 獲取所有測試天數內的週次 (確保即便沒交易的週也會顯示)
    all_weeks = sorted(df['date'].dt.strftime('%Y-W%W').unique())

    for week in all_weeks:
        group = df_trades[df_trades['year_week'] == week]

        if not group.empty:
            week_end_capital = group['capital'].iloc[-1]
            ret = (week_end_capital - last_week_capital) / last_week_capital

            # 每週最佳與最差出手
            best_trade = group.loc[group['ret'].idxmax()]
            worst_trade = group.loc[group['ret'].idxmin()]

            best_feat = best_trade['entry_features']
            worst_feat = worst_trade['entry_features']

            weekly_log.append({
                'Year_Week': week,
                'Week_End_Total_Capital': week_end_capital,
                'Weekly_Net_Return_%': ret * 100,
                'Weekly_Profit': week_end_capital - last_week_capital,
                'Trade_Count': len(group),
                'Best_Trade_Ret_%': best_trade['ret'] * 100,
                'Best_Trade_Time': best_feat['entry_date'],
                'Best_Trade_Type': best_trade['type'],
                'Best_Trade_Signal': "LONG" if best_feat['signal'] == 1 else "SHORT",
                'Worst_Trade_Ret_%': worst_trade['ret'] * 100,
                'Worst_Trade_Time': worst_feat['entry_date'],
                'Worst_Trade_Type': worst_trade['type'],
                'Worst_Trade_Signal': "LONG" if worst_feat['signal'] == 1 else "SHORT"
            })
            weekly_true_ret[week] = ret
            last_week_capital = week_end_capital
        else:
            # 沒交易的週
            weekly_log.append({
                'Year_Week': week,
                'Week_End_Total_Capital': last_week_capital,
                'Weekly_Net_Return_%': 0.0,
                'Weekly_Profit': 0.0,
                'Trade_Count': 0,
                'Best_Trade_Ret_%': 0.0,
                'Best_Trade_Time': "N/A",
                'Best_Trade_Type': "N/A",
                'Best_Trade_Signal': "N/A",
                'Worst_Trade_Ret_%': 0.0,
                'Worst_Trade_Time': "N/A",
                'Worst_Trade_Type': "N/A",
                'Worst_Trade_Signal': "N/A"
            })
            weekly_true_ret[week] = 0.0

    if weekly_log:
        df_weekly = pd.DataFrame(weekly_log)
        df_weekly.to_csv(os.path.join(out_dir, "weekly_summary.csv"), index=False, encoding="utf-8-sig")
        print(f"💾 已儲存詳細每週報酬報告至 data_learn/weekly_summary.csv (共 {len(df_weekly)} 週)")

    avg_weekly_ret = pd.Series(weekly_true_ret).mean()
    total_ret = (current_capital - initial_capital) / initial_capital

    print("\n" + "="*50)
    print(f"🚀 --- 終極期權當沖模擬器結算報告 ---")
    print(f"累積真實淨報酬率: {total_ret*100:.2f}%")
    print(f"真實平均每週報酬: {avg_weekly_ret*100:.2f}%")
    win_rate = (df_trades['ret']>0).mean() * 100
    print(f"總交易次數: {len(df_trades)} | 勝率: {win_rate:.2f}%")
    print("="*50)
    print(f"\n✅ 回測完成！交易波段圖已儲存至 data_learn/trade_plots/")

    if avg_weekly_ret > 0:
        model_manager.save_model(ai_model, optimizer, {"avg_weekly_ret": avg_weekly_ret}, {"leverage": LEVERAGE})

    plt.figure(figsize=(12, 6))
    plt.plot(capital_curve)
    plt.savefig(os.path.join(os.path.dirname(__file__), "data_learn", "equity_curve.png"))

if __name__ == "__main__":
    run_advanced_simulator(initial_capital=120000, days=120)
