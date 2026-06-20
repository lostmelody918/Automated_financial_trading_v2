import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeDayTradingAI(nn.Module):
    """
    複合 AI 模型：結合 1D-CNN (擷取微觀 K 線型態如雙底/洗盤)
    與 Transformer Encoder (處理盤中時序記憶與注意力機制)

    參考華爾街最新高頻實踐：CNN-Transformer 架構。
    """
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=2, dropout=0.3): # 增加 dropout
        super().__init__()

        # 1D-CNN 特徵萃取器
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.dropout_cnn = nn.Dropout(dropout)

        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分類頭 (輸出: 7 個類別, -3到3對應0到6)
        self.fc1 = nn.Linear(d_model, 32)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 7)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        x = x.transpose(1, 2) # [batch, features, seq_len]

        # CNN 萃取
        x = self.dropout_cnn(F.relu(self.bn1(self.conv1(x))))

        # 轉回 Transformer 輸入格式: [batch, seq_len, d_model]
        x = x.transpose(1, 2)

        # Transformer 注意力
        out = self.transformer(x)

        # 取最後一個時間步
        last_out = out[:, -1, :]

        # 全連接預測
        y = self.dropout_fc(F.relu(self.fc1(last_out)))
        logits = self.fc2(y)

        return logits

def get_time_decay_mask(seq_len, device):
    """生成具有時間衰減(Time-Decay)與因果限制的 Attention Mask"""
    mask = torch.zeros(seq_len, seq_len, device=device)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                mask[i, j] = float('-inf') # 未來不可見 (Causal)
            else:
                mask[i, j] = -0.05 * (i - j) # 距離越遠，注意力權重懲罰越大
    return mask

class CausalMultiTimeframeAI(nn.Module):
    """
    具備反事實推理與多時間尺度因果網路 (MTF Causal Network)
    - 支援 Time-Decay Attention Mask (ALiBi 概念)
    - 支援 動態閘道融合 (Gated Contextual Fusion)
    - 支援 傾向分數預測頭 (Propensity Score Head for IPW)
    """
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=3, dropout=0.3, seq_len_1m=40, seq_len_15m=20):
        super().__init__()
        self.d_model = d_model

        # --- 1分鐘線分支 (微觀) ---
        self.conv1m = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3)
        self.ln1m = nn.LayerNorm(d_model)
        self.pos_embed_1m = nn.Parameter(torch.zeros(1, seq_len_1m, d_model))

        encoder_layer_1m = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_1m = nn.TransformerEncoder(encoder_layer_1m, num_layers=num_layers)

        # --- 15分鐘線分支 (全局/環境) ---
        self.conv15m = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3)
        self.ln15m = nn.LayerNorm(d_model)
        self.pos_embed_15m = nn.Parameter(torch.zeros(1, seq_len_15m, d_model))

        encoder_layer_15m = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_15m = nn.TransformerEncoder(encoder_layer_15m, num_layers=num_layers)

        # --- Dynamic Contextual Fusion (CaTFormer with GLU) ---
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ln_cross = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)
        
        # 動態閘道：依據微觀特徵決定是否接受宏觀環境的建議
        self.gate_proj = nn.Linear(d_model, d_model)

        # --- 因果推斷區 (Shared Representation) ---
        self.fc_shared = nn.Linear(d_model * 2, 64)
        self.dropout_shared = nn.Dropout(dropout)

        # 傾向分數預測頭 (Propensity Score for IPW)
        self.propensity_head = nn.Linear(64, 1)

        # Head 0: 預測 T=0 (無異常大單)
        self.outcome_head_0 = nn.Linear(64, 7)
        # Head 1: 預測 T=1 (有異常大單)，輸入維度為 64 + 1(干預方向)
        self.outcome_head_1 = nn.Linear(65, 7)

    def forward(self, x_1m, x_15m, treatment_dir):
        device = x_1m.device
        
        # 生成時間衰減 Mask
        mask_1m = get_time_decay_mask(x_1m.size(1), device)
        mask_15m = get_time_decay_mask(x_15m.size(1), device)

        # --- 15分鐘分支 ---
        x15 = F.pad(x_15m.transpose(1, 2), (2, 0))
        x15 = self.conv15m(x15).transpose(1, 2)
        x15 = F.relu(self.ln15m(x15))
        x15 = x15 + self.pos_embed_15m
        # 加入 Time-Decay Mask
        x15_encoded = self.transformer_15m(x15, mask=mask_15m)

        # --- 1分鐘分支 ---
        x1 = F.pad(x_1m.transpose(1, 2), (2, 0))
        x1 = self.conv1m(x1).transpose(1, 2)
        x1 = F.relu(self.ln1m(x1))
        x1 = x1 + self.pos_embed_1m
        # 加入 Time-Decay Mask
        x1_encoded = self.transformer_1m(x1, mask=mask_1m)

        # --- Dynamic Contextual Fusion (Cross-Attention + Gated Fusion) ---
        # 由於 Cross-Attention 中 key sequence 長度是 15m，我們不用 target mask 影響它，僅靠 encoder 的處理
        attn_out, _ = self.cross_attn(query=x1_encoded, key=x15_encoded, value=x15_encoded)
        
        # 計算動態閘道 (Sigmoid 激勵)
        gate = torch.sigmoid(self.gate_proj(x1_encoded))
        
        # 閘道融合 (GLU)
        x1_fused = self.ln_cross(x1_encoded + gate * self.dropout_cross(attn_out))

        out1 = x1_fused[:, -1, :]
        out15 = x15_encoded[:, -1, :]

        # --- 共享表徵與因果預測 ---
        combined = torch.cat([out1, out15], dim=-1)
        shared_rep = self.dropout_shared(F.relu(self.fc_shared(combined)))

        # 傾向分數預測
        prop_logit = self.propensity_head(shared_rep)

        # 反事實分支
        y0_logits = self.outcome_head_0(shared_rep)
        rep_with_dir = torch.cat([shared_rep, treatment_dir], dim=-1)
        y1_logits = self.outcome_head_1(rep_with_dir)

        return y0_logits, y1_logits, prop_logit
