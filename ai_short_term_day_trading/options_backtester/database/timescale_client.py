import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from sqlalchemy import create_engine
from typing import List, Dict, Any

class TimescaleDBClient:
    def __init__(self, connection_string: str):
        """
        Initialize connection to TimescaleDB.
        Example connection_string: 'postgresql://user:password@localhost:5432/finance_db'
        """
        self.conn_str = connection_string
        self.conn = None
        # Replace 'postgresql://' with 'postgresql+psycopg2://' for SQLAlchemy compatibility if needed
        sa_url = connection_string
        if sa_url.startswith('postgresql://'):
            sa_url = sa_url.replace('postgresql://', 'postgresql+psycopg2://', 1)
        self.sa_engine = create_engine(sa_url)

    def connect(self):
        self.conn = psycopg2.connect(self.conn_str)
        self.conn.autocommit = True

    def disconnect(self):
        if self.conn:
            self.conn.close()
        if self.sa_engine:
            self.sa_engine.dispose()

    def initialize_schema(self):
        """
        Create necessary tables and convert them to TimescaleDB hypertables.
        """
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            # Enable TimescaleDB extension if not exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            
            # Create options tick data table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS options_ticks (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    strike REAL NOT NULL,
                    option_type VARCHAR(4) NOT NULL,
                    expiry_date DATE NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    bid_price REAL,
                    bid_volume INTEGER,
                    ask_price REAL,
                    ask_volume INTEGER
                );
            """)
            
            # Convert to hypertable partitioned by time
            # Check if it's already a hypertable to prevent errors
            cur.execute("""
                SELECT create_hypertable('options_ticks', 'time', if_not_exists => TRUE);
            """)

            # Create L2 Order Book Snapshot table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS options_orderbook_snapshots (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    bid_levels JSONB, -- stores [{"price": p, "volume": v}, ...]
                    ask_levels JSONB  -- stores [{"price": p, "volume": v}, ...]
                );
            """)
            
            # Convert to hypertable
            cur.execute("""
                SELECT create_hypertable('options_orderbook_snapshots', 'time', if_not_exists => TRUE);
            """)
            
            # Create index on symbol for faster querying
            cur.execute("CREATE INDEX IF NOT EXISTS ix_symbol_time ON options_ticks (symbol, time DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_ob_symbol_time ON options_orderbook_snapshots (symbol, time DESC);")

    def insert_ticks(self, df: pd.DataFrame):
        """
        Fast insertion of tick data using execute_values.
        Expected columns: time, symbol, strike, option_type, expiry_date, price, volume, bid_price, bid_volume, ask_price, ask_volume
        """
        if df.empty:
            return
            
        if not self.conn:
            self.connect()
            
        query = """
            INSERT INTO options_ticks 
            (time, symbol, strike, option_type, expiry_date, price, volume, bid_price, bid_volume, ask_price, ask_volume)
            VALUES %s
        """
        
        # Convert DataFrame to list of tuples
        records = [tuple(x) for x in df.to_numpy()]
        
        with self.conn.cursor() as cur:
            execute_values(cur, query, records)

    def fetch_ticks(self, start_time: str, end_time: str, symbols: List[str] = None) -> pd.DataFrame:
        """
        Query tick data from TimescaleDB using SQLAlchemy engine for optimal Pandas performance.
        """
        query = "SELECT * FROM options_ticks WHERE time >= %(start_time)s AND time <= %(end_time)s"
        params = {"start_time": start_time, "end_time": end_time}
        
        if symbols:
            query += " AND symbol = ANY(%(symbols)s)"
            params["symbols"] = symbols
            
        query += " ORDER BY time ASC"
        
        # SQLAlchemy supports passing dictionaries for named parameters
        return pd.read_sql_query(query, self.sa_engine, params=params)

if __name__ == "__main__":
    # Test initialization
    import os
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/finance_db")
    # client = TimescaleDBClient(db_url)
    # client.initialize_schema()
    print("TimescaleDB Client Module Loaded.")