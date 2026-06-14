import pandas as pd
import json
import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, target_date: str, output_dir: str = "reports_output"):
        self.target_date = target_date
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_json_chart_config(self, df: pd.DataFrame, filename: str):
        """
        Generates a JSON configuration file for dynamic charting (e.g., ECharts or Plotly)
        instead of saving raster images.
        """
        # Ensure time is string for JSON serialization
        if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
            time_data = df['time'].dt.strftime('%H:%M:%S').tolist()
        else:
            time_data = df.index.tolist()

        # Simple schema matching ECharts or similar
        chart_config = {
            "title": {"text": f"PnL Attribution for {self.target_date}"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Total PnL", "Delta", "Gamma", "Theta", "Vega", "Residual"]},
            "xAxis": {"type": "category", "data": time_data},
            "yAxis": {"type": "value"},
            "series": [
                {"name": "Total PnL", "type": "line", "data": df['total_actual_pnl'].cumsum().fillna(0).tolist() if 'total_actual_pnl' in df.columns else []},
                {"name": "Delta", "type": "line", "data": df['pnl_delta'].cumsum().fillna(0).tolist() if 'pnl_delta' in df.columns else []},
                {"name": "Gamma", "type": "line", "data": df['pnl_gamma'].cumsum().fillna(0).tolist() if 'pnl_gamma' in df.columns else []},
                {"name": "Theta", "type": "line", "data": df['pnl_theta'].cumsum().fillna(0).tolist() if 'pnl_theta' in df.columns else []},
                {"name": "Vega", "type": "line", "data": df['pnl_vega'].cumsum().fillna(0).tolist() if 'pnl_vega' in df.columns else []},
                {"name": "Residual", "type": "line", "data": df['pnl_residual'].cumsum().fillna(0).tolist() if 'pnl_residual' in df.columns else []}
            ]
        }

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chart_config, f, indent=4, ensure_ascii=False)
        print(f"Chart config saved to {filepath}")

    def generate_llm_markdown_report(self, summary_metrics: dict, data_file_path: str, filename: str):
        """
        Generates a strict single-column Markdown report optimized for LLM reading.
        Avoids multi-column layouts and embeds pointers to raw data.
        """
        md_content = f"""# Options Day Trading Backtest Report
**Date:** {self.target_date}

## Overview
This report summarizes the options day trading backtest results. Detailed tick-level and order book data has been isolated to prevent context flooding.

## Key Performance Indicators (KPIs)
* **Total PnL:** {summary_metrics.get('total_pnl', 0.0):.2f}
* **Average Slippage:** {summary_metrics.get('avg_slippage', 0.0):.2f}
* **Max Drawdown:** {summary_metrics.get('max_drawdown', 0.0):.2f}
* **Event Flags:** {summary_metrics.get('event_flags_count', 0)}

## PnL Greek Attribution
* **Delta PnL:** {summary_metrics.get('pnl_delta', 0.0):.2f}
* **Gamma PnL:** {summary_metrics.get('pnl_gamma', 0.0):.2f}
* **Theta PnL:** {summary_metrics.get('pnl_theta', 0.0):.2f}
* **Vega PnL:** {summary_metrics.get('pnl_vega', 0.0):.2f}
* **Residual (Unexplained):** {summary_metrics.get('pnl_residual', 0.0):.2f}

## Data Access
For detailed analysis, the raw attribution data can be loaded via code interpreter from:
`{data_file_path}`
"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"LLM Markdown report saved to {filepath}")

    def export_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        Exports the raw dataframe to Parquet for efficient Code Interpreter loading.
        """
        filepath = os.path.join(self.output_dir, filename)
        # Using parquet is highly recommended for Pandas/LLM integrations
        df.to_parquet(filepath, index=False)
        return filepath

if __name__ == "__main__":
    # Test Reporter
    reporter = ReportGenerator("2026-06-10")
    dummy_df = pd.DataFrame({
        'time': pd.date_range('08:45', '13:45', freq='5min'),
        'total_actual_pnl': [10]*61,
        'pnl_delta': [5]*61,
        'pnl_gamma': [2]*61,
        'pnl_theta': [-1]*61,
        'pnl_vega': [3]*61,
        'pnl_residual': [1]*61
    })
    
    parquet_path = reporter.export_data(dummy_df, "attribution_data.parquet")
    reporter.generate_json_chart_config(dummy_df, "pnl_chart.json")
    
    metrics = {
        'total_pnl': 610.0,
        'avg_slippage': 0.5,
        'max_drawdown': 50.0,
        'event_flags_count': 0,
        'pnl_delta': 305.0,
        'pnl_gamma': 122.0,
        'pnl_theta': -61.0,
        'pnl_vega': 183.0,
        'pnl_residual': 61.0
    }
    reporter.generate_llm_markdown_report(metrics, parquet_path, "llm_report.md")
