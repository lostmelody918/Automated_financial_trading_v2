# 📈 複合當沖 AI 系統 (CompositeDayTradingAI) - 因果推斷架構升級規格書

## 🎯 重構目標與學術理論優化 (Project Objective & Theory)

將現有的 `CompositeDayTradingAI` 升級為具備「反事實推理（Counterfactual Reasoning）」能力的**因果時間 Transformer (Causal Temporal Transformer)**。本專案將應用於台指期與選擇權（TX/TXO）市場，並遵循以下最新量化學術原則：

| 理論基礎 | 實作邏輯與目的 |
| :--- | :--- |
| **微觀結構干預 (Microstructure Intervention)** | 摒棄單一 K 線依賴，從 Tick 數據提取**訂單流不平衡 (OFI)** 作為微觀干預變數 ($T$)，剝離虛假的價格相關性。 |
| **時間感知因果表徵 (Time-Aware Causal Representation)** | 引入「共享表徵層 (Shared Representation)」，強迫模型學習跨越不同市場波動率的靜態因果因子。 |
| **靈活雙重機器學習 (Dual-Head Counterfactual)** | 實作雙頭網路，隔離 $T=0$（無大單）與 $T=1$（有大單）的平行宇宙路徑，藉此精準計算因果效應 ($\text{CATE}$)。 |

---

## 🛠️ 任務一：重構模型主體 `composite_ai.py`

請將原有的 `CompositeDayTradingAI` 升級為新的 `CausalDayTradingAI` 類別。

**架構變更要求：**
* **保留特徵萃取：** 維持原有的 `1D-CNN` 與 `Transformer Encoder`，用於處理時序 K 線特徵 $X$。
* **新增共享表徵層：** 將 Transformer 最後一個時間步的輸出（`last_out`）透過 Linear 層壓縮為獨立於環境的底層特徵。
* **雙頭預測網路：**
  * `Outcome Head 0`: 預測當干預未發生（無異常大單）時的 7 類未來走勢。
  * `Outcome Head 1`: 預測當干預發生時的 7 類未來走勢。需將「共享特徵」與「干預方向 Tensor」進行拼接（Concatenation）後再輸入。

**參考實作結構：**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDayTradingAI(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        # --- 1. 時序特徵萃取區 ---
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.dropout_cnn = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- 2. 因果推斷區 (Shared Representation) ---
        self.fc_shared = nn.Linear(d_model, 64)
        self.dropout_shared = nn.Dropout(dropout)

        # --- 3. 反事實雙頭輸出 (Counterfactual Heads) ---
        # Head 0: 預測 T=0 (無異常大單)
        self.outcome_head_0 = nn.Linear(64, 7)
        # Head 1: 預測 T=1 (有異常大單)，輸入維度為 64 + 1 (干預方向)
        self.outcome_head_1 = nn.Linear(65, 7)

    def forward(self, x, treatment_dir):
        # x shape: [batch, seq_len, features]
        # treatment_dir shape: [batch, 1] (1: 買盤干預, -1: 賣盤干預, 0: 無干預)

        # CNN & Transformer 萃取
        x = x.transpose(1, 2)
        x = self.dropout_cnn(F.relu(self.bn1(self.conv1(x))))
        x = x.transpose(1, 2)

        out = self.transformer(x)
        last_out = out[:, -1, :]

        # 共享表徵
        shared_rep = self.dropout_shared(F.relu(self.fc_shared(last_out)))

        # 平行宇宙預測分支
        y0_logits = self.outcome_head_0(shared_rep)
        rep_with_dir = torch.cat([shared_rep, treatment_dir], dim=-1)
        y1_logits = self.outcome_head_1(rep_with_dir)

        return y0_logits, y1_logits
```

---

## 🛠️ 任務二：新增資料預處理腳本 `data_pipeline.py`

請實作處理台指期 Tick 資料的函數，以萃取干預變數 $T$。

**資料處理步驟：**
1. **讀取原始資料：** 載入包含 `datetime`、`price`、`volume`、`is_buyer_maker` 的 Tick DataFrame。
2. **判斷內外盤：** `is_buyer_maker == False` 視為主動買（外盤）；反之為主動賣（內盤）。
3. **計算 OFI：** 計算每分鐘的「淨主動攻擊量」（主動買量減去主動賣量）。
4. **定義干預變數 ($T$)：** 計算過去 60 期的滾動標準差（或 PR95）作為動態門檻。若淨主動攻擊量的絕對值大於門檻，定義 `Treatment_T = 1.0`，否則為 `0.0`。同步記錄干預方向 `Treatment_Direction`（1.0 為淨買，-1.0 為淨賣，0.0 為無干預）。
5. **降頻對齊：** 將 Tick 資料轉換（Resample）為 1 分鐘 K 線頻率。
6. **資料融合：** 將結果 DataFrame 與現有的 1 分鐘 K 線特徵 DataFrame 進行 `merge`。

---

## 🛠️ 任務三：擴展模型管理器 `model_manager.py`

修改 `TradingModelManager.save_model` 方法，支援因果模型的健康度指標追蹤。在 `metadata.json` 結構中，必須新增 `causal_metrics` 區塊：

**JSON 實作結構：**
```json
"causal_metrics": metrics.get("causal", {
    "ATE_estimation": null,          // 平均干預效應 (Average Treatment Effect)
    "factual_loss": null,            // 實際發生路徑的 Loss
    "counterfactual_variance": null  // 反事實預測方差 (衡量模型穩健性)
})
```

---

## 🛠️ 任務四：訓練引擎與實盤推論邏輯 (`train.py` / `inference.py`)

請在訓練迴圈與推論腳本中，實作嚴格的**事實/反事實隔離邏輯**。

**1. 訓練期：Factual Loss (事實損失優化)**
在訓練時，模型會同時輸出 `y0_logits` 和 `y1_logits`。請透過 Masking 機制，**確保只能用現實世界發生的那條路徑來計算梯度**。

| 樣本真實狀態 | 梯度計算來源 | 損失函數 (Loss) 實作方式 | 未發生路徑處理 |
| :--- | :--- | :--- | :--- |
| 無干預 ($T=0$) | `y0_logits` | `CrossEntropy(y0_logits, True_Y)` | 使用 `torch.where` 進行 Masking 或阻斷梯度 |
| 有干預 ($T=1$) | `y1_logits` | `CrossEntropy(y1_logits, True_Y)` | 同上 |

**2. 推論期：Causal Effect Calculation (實盤推論邏輯)**
實盤進場決策不再依賴單一預測，而是依賴條件平均處置效應 ($\text{CATE}$)。

* **步驟一：** 給定最新 K 線特徵 $X$。
* **步驟二：** 強迫模型進行兩次平行推論，取得 `y0_pred`（無干預預期漲幅）與 `y1_pred`（傳入 `treatment_dir = 1.0` 的預期漲幅）。
* **步驟三：** 計算因果效應 $\text{CATE} = y_{1} - y_{0}$。
* **步驟四：** 交易訊號輸出邏輯：僅當 `y1_pred` 方向為強勢看漲，**且** $\text{CATE} > \text{交易成本閾值}$ 時，才產生 `Buy` (作多/Buy Call) 訊號。這代表突破是由大單的因果推力所造成，而非隨機雜訊，能大幅過濾「假突破」。