import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from urllib.parse import urlparse

# 設定專案根目錄路徑，確保可以找到模組與 .env 檔
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from ai_short_term_day_trading.options_backtester.database.timescale_client import TimescaleDBClient

# 載入 .env 變數
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

# 取得連線字串
db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("錯誤: 找不到 DATABASE_URL，請確認 .env 檔案設定。")
    exit(1)

# 解析 URL 以便提取資料庫名稱與基本連線資訊
parsed_url = urlparse(db_url)
db_name = parsed_url.path.lstrip('/')
base_url = f"{parsed_url.scheme}://{parsed_url.username}:{parsed_url.password}@{parsed_url.hostname}:{parsed_url.port}/postgres"

# 第一步：確保資料庫存在
try:
    print("檢查並建立資料庫（如果尚未存在）...")
    conn = psycopg2.connect(base_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # 檢查資料庫是否存在
    cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
    exists = cur.fetchone()
    
    if not exists:
        print(f"資料庫 {db_name} 不存在，正在建立...")
        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"✅ 資料庫 {db_name} 建立完成！")
    else:
        print(f"✅ 資料庫 {db_name} 已存在。")
        
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ 建立資料庫時發生錯誤: {e}")
    print("嘗試直接連線進行初始化...")

# 第二步：連線至指定資料庫並初始化 Schema
print(f"準備連線至資料庫 {db_name} 並初始化 Schema...")
client = TimescaleDBClient(db_url)

try:
    client.initialize_schema()
    print("✅ TimescaleDB Schema 初始化完成！資料表與 Hypertables 已成功建立。")
except Exception as e:
    print(f"❌ 初始化失敗: {e}")
finally:
    client.disconnect()
