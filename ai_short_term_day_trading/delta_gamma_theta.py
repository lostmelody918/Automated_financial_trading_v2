import numpy as np
from scipy.stats import norm
from datetime import datetime

def get_api_based_dte(active_contract, current_time):
    """
    🚀 利用 Shioaji API 合約屬性精準計算 DTE (完全免疫國定假日與特殊契約)
    """
    try:
        # 1. 從合約物件安全提取交割日字串
        delivery_date_str = getattr(active_contract, 'delivery_date', None)

        if not delivery_date_str:
            raise ValueError("合約物件缺乏 delivery_date 屬性")

        # 2. 統一格式化 (消除斜線，統一變為 YYYYMMDD)
        clean_date_str = delivery_date_str.replace('/', '')

        # 3. 綁定台指期/選擇權的法定結算時間 (13:30:00)
        settlement_time = datetime.strptime(f"{clean_date_str} 13:30:00", "%Y%m%d %H:%M:%S")

        # 4. 計算精確剩餘秒數並轉為天數
        delta = settlement_time - current_time
        dte_days = delta.total_seconds() / 86400.0

        # 5. 防禦機制：如果已經超過結算時間，給予極小值避免 BSM 分母除以 0
        return max(dte_days, 0.001)

    except Exception as e:
        print(f"⚠️ 解析 API 交割日失敗 ({e})，啟動降級防禦 (預設 1 天)")
        return 1.0  # 降級保護

def calculate_bs_greeks(S, K, T, r, iv, option_type="Call"):
    """
    計算 Black-Scholes 選擇權理論價與 Greeks
    T: 年化剩餘時間 (Days / 365.0)
    iv: 隱含波動率 (例如 0.22)
    """
    if T <= 0 or iv <= 0:
        return 0.5 if option_type == "Call" else -0.5, 0.0, 0.0, S if option_type == "Call" else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)

    # 概率密度與累積分布
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)

    if option_type == "Call":
        delta = cdf_d1
        # 年化 Theta 轉日化 Theta (除以 365)
        theta = (- (S * pdf_d1 * iv) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = cdf_d1 - 1
        theta = (- (S * pdf_d1 * iv) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0

    gamma = pdf_d1 / (S * iv * np.sqrt(T))
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if option_type == "Call" else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return delta, gamma, theta, price

def get_dynamic_bsm_bounds(S, K, T, r, iv, atr, tp_mult, sl_mult, expected_hold_hours, option_type="Call", actual_entry_price=None, fee_points=0.0):
    """
    依據二階泰勒展開式，精確計算選擇權權利金的停損與停利絕對價格
    expected_hold_hours: 預期此筆短線單最大持有時間(小時)，用以預扣時間價值
    fee_points: 來回手續費與滑價所折合的點數，將墊高停利目標並收緊停損防線
    """
    delta, gamma, theta, theoretical_price = calculate_bs_greeks(S, K, T, r, iv, option_type)
    
    # 若有提供實際進場價，則以實際進場價為基準，否則使用理論價
    base_price = actual_entry_price if actual_entry_price is not None else theoretical_price

    # 計算剩餘時間價值 (Extrinsic Value) 作為 Theta 衰減的上限
    intrinsic_value = max(0, S - K) if option_type == "Call" else max(0, K - S)
    extrinsic_value = max(0, base_price - intrinsic_value)

    # 轉化持有時間為天數單位，計算預期 Theta 衰減
    dt_days = expected_hold_hours / 24.0
    raw_theta_decay = theta * dt_days
    
    # 防止末日選擇權的瞬間 Theta 過大導致預估衰減超出實際時間價值
    # 限制最大衰減不超過當前時間價值的 80%
    theta_decay = -min(abs(raw_theta_decay), extrinsic_value * 0.8)

    # 1. 停利情境下的標的資產變動 (看對方向)
    delta_S_tp = (atr * tp_mult) if option_type == "Call" else -(atr * tp_mult)
    opt_change_tp = (delta * delta_S_tp) + (0.5 * gamma * (delta_S_tp ** 2)) + theta_decay
    
    # 確保停利點至少高於進場價 (覆蓋手續費)
    # 將手續費成本 fee_points 納入，墊高動態停利點
    hard_tp_price = max(base_price + fee_points + 1.0, base_price + opt_change_tp + fee_points)

    # 2. 停損情境下的標的資產變動 (看錯方向)
    delta_S_sl = -(atr * sl_mult) if option_type == "Call" else (atr * sl_mult)
    
    # 修正停損邏輯：我們「不」把時間衰減(負值)加進停損中！
    # 時間流逝造成的虧損也應該被算在停損額度內，因此不該放寬停損線。
    opt_change_sl = (delta * delta_S_sl) + (0.5 * gamma * (delta_S_sl ** 2))
    # 將手續費成本 fee_points 納入，收緊停損點 (停損點往進場價靠近)
    hard_sl_price = base_price + opt_change_sl + fee_points

    # 安全邊界防禦：確保停損價不為負值，且維持最低點位
    hard_sl_price = max(0.5, hard_sl_price)
    
    # 如果算出來的停損太接近進場價 (例如小於 3 點 + 成本)，給予基本的防禦空間
    if (base_price - hard_sl_price) < (3.0 + fee_points):
        hard_sl_price = max(0.5, base_price - (3.0 + fee_points))

    # 用戶新增規則：動態停損點最多只能損 33% (也就是停損價最低不能低於進場價的 0.67 倍)
    max_loss_price = base_price * 0.67
    hard_sl_price = max(hard_sl_price, max_loss_price)

    return round(hard_tp_price, 1), round(hard_sl_price, 1), delta, gamma, theta_decay

"""
# [🎯 盤中進場：引入 BSM 希臘字母與時間衰減優化版]
# 假定相關即時參數已透過 API 取得或由歷史常數定義
underlying_price = snap.close  # 目前台指期貨現價
strike_price = active_contract.strike_price
days_to_expiry = float(norm_params.get("current_dte", 5)) / 365.0 # 剩餘年化時間
current_iv = 0.22 # 可透過即時報價倒推或使用昨收盤歷史波動率
current_atr = df['atr'].iloc[-1]

# 設定預期持倉時間 (例如：當沖短線游擊預期持有 2 小時)
expected_hold_time = 2.0

hard_tp_price, hard_sl_price, d, g, t_decay = get_dynamic_bsm_bounds(
    S=underlying_price,
    K=strike_price,
    T=days_to_expiry,
    r=0.015, # 無風險利率
    iv=current_iv,
    atr=current_atr,
    tp_mult=tp_mult,  # 依據 Level 1, 2, 3 定義的乘數
    sl_mult=sl_mult,
    expected_hold_hours=expected_hold_time,
    option_type=opt_type
)

print(f"📊 [BSM風控對齊] Delta: {d:.3f} | Gamma: {g:.5f} | 預估持有 {expected_hold_time} 小時 Theta 損耗: {t_decay:.2f} 點")
print(f"🎯 精確風控目標 -> 買入權利金: {entry_price} | 動態停利點: {hard_tp_price} | 動態停損點: {hard_sl_price}")
"""