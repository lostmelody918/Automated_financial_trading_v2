import pytest
import torch
from composite_ai import CompositeDayTradingAI, CausalDayTradingAI

def test_composite_ai_forward():
    batch_size = 4
    seq_len = 10
    input_dim = 15
    
    # Initialize the model
    model = CompositeDayTradingAI(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
    model.eval()
    
    # Create random input tensor [batch, seq_len, features]
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
        
    # Check output shape
    assert logits.shape == (batch_size, 7)
    
    # Check if outputs are finite
    assert torch.isfinite(logits).all()

def test_causal_day_trading_ai_forward():
    batch_size = 4
    seq_len = 10
    input_dim = 15
    
    # Initialize the causal model
    model = CausalDayTradingAI(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
    model.eval()
    
    # Create random input tensors
    x = torch.randn(batch_size, seq_len, input_dim)
    treatment_dir = torch.randn(batch_size, 1) # e.g. T=1, -1, or 0
    
    # Forward pass
    with torch.no_grad():
        y0_logits, y1_logits = model(x, treatment_dir)
        
    # Check output shapes
    assert y0_logits.shape == (batch_size, 7)
    assert y1_logits.shape == (batch_size, 7)
    
    # Check if outputs are finite
    assert torch.isfinite(y0_logits).all()
    assert torch.isfinite(y1_logits).all()

