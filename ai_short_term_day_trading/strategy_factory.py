import numpy as np

class StrategyFactory:
    """模組化策略工廠：直接使用 AI 決策"""
    @staticmethod
    def get_strategy(strategy_name="composite"):
        return CompositeOptionsStrategy()

class CompositeOptionsStrategy:
    def __init__(self):
        self.name = "AI Direct Output"

    def generate_signal(self, df_slice, ai_score=None, last_win=False):
        if ai_score is None or len(ai_score) != 7: return 0

        # ai_score indices:
        # 0: Strong Down (-3), 1: Med Down (-2), 2: Weak Down (-1)
        # 3: Hold (0)
        # 4: Weak Up (1), 5: Med Up (2), 6: Strong Up (3)

        pred_class = np.argmax(ai_score)

        # Map class to signal strength
        mapping = {
            0: -3,
            1: -2,
            2: -1,
            3: 0,
            4: 1,
            5: 2,
            6: 3
        }

        base_signal = mapping.get(pred_class, 0)

        # --- 動能耗竭防護網 (Momentum Exhaustion Safety Net) 增強版 ---
        if base_signal != 0 and not df_slice.empty:
            last_row = df_slice.iloc[-1]

            # 獲取安全網所需特徵
            vwap_bias = last_row.get('vwap_bias', 0.0)
            rsi_fast = last_row.get('rsi_fast', 50.0)
            spot_proxy = last_row.get('spot_futures_proxy', 0.0)
            slope_ma20 = last_row.get('slope_ma20', 0.0)

            # 大局觀：針對單邊趨勢放寬乖離容忍度 (2.5 倍)
            long_bias_warn = 0.0020
            long_bias_block = 0.0040
            short_bias_warn = -0.0020
            short_bias_block = -0.0040

            if slope_ma20 > 2.0: # 多頭趨勢，軋空或緩漲
                long_bias_warn *= 2.5
                long_bias_block *= 2.5
            elif slope_ma20 < -2.0: # 空頭趨勢，殺多或緩跌
                short_bias_warn *= 2.5
                short_bias_block *= 2.5

            # 做多過熱防護
            if base_signal > 0:
                if vwap_bias > long_bias_warn or rsi_fast > 80:
                    # 如果極度過熱，嚴格 AND 條件才大降級
                    if vwap_bias > long_bias_block and rsi_fast > 90:
                        base_signal = 1 # 降級為 Level 1 而非沒收
                        print(f"🛡️ [安全網] 極端追高降級至 Level 1: vwap_bias={vwap_bias:.5f}, rsi_fast={rsi_fast:.1f}, slope20={slope_ma20:.2f}")
                    # 否則稍微降級
                    else:
                        base_signal = max(1, base_signal - 1)

            # 做空過熱防護
            elif base_signal < 0:
                if vwap_bias < short_bias_warn or rsi_fast < 20:
                    # 如果極端殺低，嚴格 AND 條件才大降級
                    if vwap_bias < short_bias_block and rsi_fast < 10:
                        base_signal = -1 # 降級為 Level 1 而非沒收
                        print(f"🛡️ [安全網] 極端殺低降級至 Level 1: vwap_bias={vwap_bias:.5f}, rsi_fast={rsi_fast:.1f}, slope20={slope_ma20:.2f}")
                    else:
                        base_signal = min(-1, base_signal + 1)

        # --- 淺回檔順勢承接 (Micro-Pullback) & V轉捕捉 ---
        if base_signal == 0 and not df_slice.empty:
            last_row = df_slice.iloc[-1]

            # 確保所需的特徵存在
            required_cols = ['rsi_fast', 'vwap_bias', 'Close', 'Open', 'momentum_explosion', 'slope_ma20']
            if all(col in df_slice.columns for col in required_cols):
                rsi_fast = last_row['rsi_fast']
                vwap_bias = last_row['vwap_bias']
                slope_ma20 = last_row['slope_ma20']

                # 淺回檔做多 (多頭趨勢中的微幅拉回)
                if slope_ma20 > 2.2 and -0.0005 < vwap_bias < 0.0010 and rsi_fast < 35:
                    print(f"🎯 [微觀回檔] 觸發順勢做多: slope20={slope_ma20:.2f}, vwap_bias={vwap_bias:.5f}, rsi_fast={rsi_fast:.1f}")
                    return 1  # 賦予標準 Level 1 訊號

                # 淺回檔做空 (空頭趨勢中的微幅反彈)
                if slope_ma20 < -2.2 and -0.0010 < vwap_bias < 0.0005 and rsi_fast > 68:
                    print(f"🎯 [微觀回檔] 觸發順勢做空: slope20={slope_ma20:.2f}, vwap_bias={vwap_bias:.5f}, rsi_fast={rsi_fast:.1f}")
                    return -1 # 賦予標準 Level 1 訊號

                # V轉做多 (瞬間超賣 + 遠離 VWAP 下方 + 動能爆發 + 出現紅K)
                if (rsi_fast < 25 and
                    vwap_bias < -0.003 and
                    last_row['momentum_explosion'] == 1 and
                    last_row['Close'] > last_row['Open']):
                    return 10  # 10 代表 V 轉做多剝頭皮 (Scalping)

                # V轉做空 (瞬間超買 + 遠離 VWAP 上方 + 動能爆發 + 出現黑K)
                if (rsi_fast > 75 and
                    vwap_bias > 0.003 and
                    last_row['momentum_explosion'] == 1 and
                    last_row['Close'] < last_row['Open']):
                    return -10 # -10 代表 V 轉做空剝頭皮 (Scalping)

        return base_signal