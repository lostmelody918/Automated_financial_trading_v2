import pytest
import os
import json
import torch
import torch.nn as nn
from model_manager import TradingModelManager

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

def test_trading_model_manager(tmp_path):
    model_dir = tmp_path / "saved_models"
    manager = TradingModelManager(model_dir=str(model_dir))
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = {"loss": 0.5, "accuracy": 0.8}
    hyperparameters = {"lr": 0.001, "batch_size": 32}
    
    # Test saving
    manager.save_model(model, optimizer, metrics, hyperparameters)
    
    saved_files = os.listdir(model_dir)
    assert "trading_model_v1.pth" in saved_files
    assert "trading_model_v1_metadata.json" in saved_files
    
    # Test loading
    new_model = DummyModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    
    loaded_model, loaded_optimizer, version = manager.load_latest_model(new_model, new_optimizer)
    
    assert version == 1
    # Check if weights are loaded correctly by comparing a weight
    assert torch.equal(model.fc.weight, loaded_model.fc.weight)
