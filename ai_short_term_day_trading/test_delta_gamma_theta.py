import pytest
from datetime import datetime, timedelta
from delta_gamma_theta import calculate_bs_greeks, get_dynamic_bsm_bounds, get_api_based_dte

class DummyContract:
    def __init__(self, delivery_date):
        self.delivery_date = delivery_date

def test_calculate_bs_greeks_call():
    S = 100
    K = 100
    T = 30 / 365.0
    r = 0.015
    iv = 0.2
    
    delta, gamma, theta, price = calculate_bs_greeks(S, K, T, r, iv, option_type="Call")
    
    assert delta > 0 and delta < 1
    assert gamma > 0
    assert theta < 0
    assert price > 0

def test_calculate_bs_greeks_put():
    S = 100
    K = 100
    T = 30 / 365.0
    r = 0.015
    iv = 0.2
    
    delta, gamma, theta, price = calculate_bs_greeks(S, K, T, r, iv, option_type="Put")
    
    assert delta > -1 and delta < 0
    assert gamma > 0
    assert theta < 0
    assert price > 0

def test_get_dynamic_bsm_bounds():
    S = 20000
    K = 20000
    T = 5 / 365.0
    r = 0.015
    iv = 0.22
    atr = 50
    tp_mult = 2.0
    sl_mult = 1.0
    expected_hold_hours = 2.0
    
    hard_tp_price, hard_sl_price, delta, gamma, theta_decay = get_dynamic_bsm_bounds(
        S, K, T, r, iv, atr, tp_mult, sl_mult, expected_hold_hours, option_type="Call"
    )
    
    assert hard_tp_price > hard_sl_price
    assert hard_sl_price >= 0.5
    assert delta > 0

def test_get_api_based_dte():
    contract = DummyContract(delivery_date="2026/06/17")
    current_time = datetime(2026, 6, 15, 13, 30, 0)
    dte = get_api_based_dte(contract, current_time)
    assert round(dte, 1) == 2.0
