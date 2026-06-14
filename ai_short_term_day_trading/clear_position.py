import sqlite3
import os

def force_clear_position():
    # 確保無論在哪裡執行這個腳本，都能正確找到資料庫位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "position_state.db")

    if not os.path.exists(db_path):
        print(f"找不到 {db_path}，目前為乾淨狀態。")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            # 這是對應你 PositionManager 的 clear_position 邏輯
            conn.execute("""
                UPDATE position_state
                SET position=0,
                    entry_price=0.0,
                    num_contracts=0,
                    highest_price_since_entry=0.0,
                    active_contract_symbol=NULL,
                    entry_time=NULL,
                    trade_capital_used=0.0,
                    hard_tp_price=0.0,
                    hard_sl_price=0.0,
                    strategy_label=NULL
                WHERE id=1
            """)
            print(f"✅ 資料庫未平倉狀態已成功強制清空！({db_path})")
    except Exception as e:
        print(f"❌ 清除失敗: {e}")

if __name__ == "__main__":
    force_clear_position()