import numpy as np
from scipy.stats import norm
import pandas as pd
from typing import Tuple

def compute_bsm_vectorized(
    S: np.ndarray, 
    K: np.ndarray, 
    T: np.ndarray, 
    r: np.ndarray, 
    iv: np.ndarray, 
    option_types: np.ndarray
) -> pd.DataFrame:
    """
    Vectorized Black-Scholes calculations using NumPy.
    Handles multi-dimensional arrays for dynamic strikes/expiries.
    
    Parameters:
        S: Underlying Asset Price
        K: Strike Price
        T: Time to Expiration (in years)
        r: Risk-free rate (e.g., 0.02 for 2%)
        iv: Implied Volatility
        option_types: Array of strings ('Call' or 'Put')
        
    Returns:
        DataFrame containing Price, Delta, Gamma, Theta, Vega
    """
    # Safeguard against T=0 or IV=0
    T = np.maximum(T, 1e-5)
    iv = np.maximum(iv, 1e-5)
    
    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    cdf_neg_d1 = norm.cdf(-d1)
    cdf_neg_d2 = norm.cdf(-d2)
    
    is_call = (option_types == 'Call') | (option_types == 'C')
    is_put = ~is_call
    
    # Calculate Prices
    call_price = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
    put_price = K * np.exp(-r * T) * cdf_neg_d2 - S * cdf_neg_d1
    price = np.where(is_call, call_price, put_price)
    
    # Calculate Delta
    call_delta = cdf_d1
    put_delta = cdf_d1 - 1.0
    delta = np.where(is_call, call_delta, put_delta)
    
    # Calculate Gamma (same for Call and Put)
    gamma = pdf_d1 / (S * iv * np.sqrt(T))
    
    # Calculate Vega (same for Call and Put, divided by 100 to represent 1% change)
    vega = (S * pdf_d1 * np.sqrt(T)) / 100.0
    
    # Calculate Theta (converted to daily decay by dividing by 365)
    call_theta = (- (S * pdf_d1 * iv) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2) / 365.0
    put_theta = (- (S * pdf_d1 * iv) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * cdf_neg_d2) / 365.0
    theta = np.where(is_call, call_theta, put_theta)
    
    return pd.DataFrame({
        'Price': price,
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega
    })

def imply_volatility(
    market_prices: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    option_types: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4
) -> np.ndarray:
    """
    Vectorized Newton-Raphson to find Implied Volatility.
    """
    iv = np.full_like(market_prices, 0.2) # Initial guess 20%
    
    for _ in range(max_iter):
        greeks = compute_bsm_vectorized(S, K, T, r, iv, option_types)
        prices = greeks['Price'].values
        vega = greeks['Vega'].values * 100.0 # Revert to actual derivative
        
        diff = market_prices - prices
        if np.max(np.abs(diff)) < tol:
            break
            
        vega = np.maximum(vega, 1e-5) # Prevent division by zero
        iv = iv + diff / vega
        iv = np.clip(iv, 1e-4, 5.0) # Restrict IV bounds
        
    return iv
