import os
import json
from data_engine import DayTradingDataEngine

def main():
    print("🚀 啟動盤前/盤後籌碼抓取程序...")
    engine = DayTradingDataEngine()
    df_chips = engine.fetch_real_historical_chips(days=7)
    
    if df_chips is not None and not df_chips.empty:
        # Save to chips_cache.json
        cache_path = os.path.join(os.path.dirname(__file__), "chips_cache.json")
        df_chips['date'] = df_chips['date'].astype(str)
        data = df_chips.to_dict(orient='records')
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ 籌碼快照已成功儲存至 {cache_path}")
    else:
        print("❌ 抓取籌碼失敗，快照未更新。")

if __name__ == "__main__":
    main()
