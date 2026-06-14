# AI Short-Term Day Trading Module Documentation

This document outlines the architecture, core components, and functionalities of the `ai_short_term_day_trading` module. This module is an automated day-trading system utilizing a deep learning AI architecture (CNN-Transformer) to trade index futures and options based on intraday features and market momentum.

## System Architecture Overview

The system is composed of several interdependent components spanning data ingestion, AI inference, signal processing, execution simulation, and post-market reporting.

### 1. Data Processing & Engineering
- **`data_engine.py`**: The core data pipeline. Connects to the Shioaji API to fetch intraday tick/k-bar data (primarily TXF futures). It automatically calculates 29 real-time technical indicators, including VWAP bias, MACD, ATR, RSI, fast RSI, momentum explosion, and moving average slopes. It also aligns intraday data with daily context (e.g., gap amplitude, intraday trend).
- **`fetch_daily_chips.py`**: A script to download and cache daily institutional chip data (e.g., foreign investor futures positions, options open interest) which can be injected into the models as macroscopic context.
- **`delta_gamma_theta.py`**: Contains Black-Scholes formulas and calculations to dynamically compute options Greeks (Delta, Gamma, Theta) and dynamic stop-loss/take-profit boundaries based on ATR and IV.

### 2. AI Architecture
- **`composite_ai.py`**: Defines `CompositeDayTradingAI`, a hybrid neural network architecture leveraging both a **1D-CNN** and a **Transformer Encoder**.
  - **1D-CNN**: Acts as a feature extractor to capture micro-patterns in K-lines (e.g., washouts, double bottoms).
  - **Transformer**: Processes temporal dependencies, memory, and attention mechanisms for intraday sequence data.
  - **Output**: The network classifies market direction into 7 discrete classes (-3: Strong Down, 0: Hold, +3: Strong Up).

### 3. Strategy & Risk Management
- **`strategy_factory.py`**: Translates the raw neural network output into actionable trading signals. It includes a robust safety mechanism:
  - **Momentum Exhaustion Safety Net**: Suppresses AI signals if the market is overextended (e.g., extreme `vwap_bias` or `rsi_fast` over 80/90), preventing buying at the top or shorting at the bottom.
  - **Micro-Pullback Logic**: Triggers "buy the dip" (or "sell the rally") signals during strong trends when there is a shallow, localized pullback.

### 4. Simulation & Backtesting
- **`live_option_simulator_v2.py`**: A comprehensive live trading simulator for options. It manages the portfolio, handles dynamic entry/exit points, calculates running PnL, tracks capital, and generates End-Of-Day (EOD) reports.
- **`backtest_simulator.py`**: Historical backtesting engine to evaluate the strategy's equity curve, win rate, and drawdown over past data, capable of plotting trade entry/exit points.

### 5. Model Management
- **`model_manager.py`**: Handles versioning for the PyTorch models. Automatically saves checkpoints as `trading_model_vX.pth` alongside `_metadata.json` files that track hyperparameters and performance metrics.
- **`train_model.py`**: The execution script to train the neural network, managing epochs, loss calculation, and updating the model via the manager.

### 6. UI & Reporting
- **`live_dashboard.py`**: A dynamic visual dashboard to monitor real-time K-lines, model predictions, active indicators, and simulated trades.
- **`post_market_export.py`**: Automates the extraction of trade logs and performance metrics into CSV/reports at the close of the market for further analysis and reinforcement learning.

## Workflow Summary

1. **Pre-market**: Institutional chips and previous day's metrics are cached.
2. **Intraday**: `data_engine.py` streams live ticks and engineers 29 features.
3. **Inference**: The feature matrix is fed into the CNN-Transformer (`composite_ai.py`).
4. **Validation**: The raw AI score (-3 to 3) is passed to `strategy_factory.py`, which filters it through the Momentum Exhaustion Safety Net.
5. **Execution**: If the signal survives the safety checks, the simulator (`live_option_simulator_v2.py`) opens a mock position and dynamically monitors for take-profit/stop-loss conditions using dynamically calculated options Greeks.
6. **Post-market**: Models are optionally retrained (`train_model.py`) and reports are generated (`post_market_export.py`).
