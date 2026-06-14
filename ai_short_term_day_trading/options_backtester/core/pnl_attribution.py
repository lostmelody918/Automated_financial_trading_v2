import pandas as pd
import numpy as np

def calculate_pnl_attribution(df: pd.DataFrame, initial_premium: float) -> dict:
    """
    Calculates Greek PnL attribution using the Trapezoidal Rule.
    Equation: dP ≈ Δ⋅dS + 1/2 Γ⋅dS^2 + Θ⋅dt + V⋅dσ + Residual
    
    Args:
        df: DataFrame with consecutive rows representing time steps t, t+1.
            Required columns: S, iv, Delta, Gamma, Theta, Vega, Price
        initial_premium: The cost/credit of entering the position.
            
    Returns:
        Dictionary containing cumulative PnL components and event flags.
    """
    # Calculate differences between t and t+1
    df['dS'] = df['S'].diff()
    df['dIV'] = df['iv'].diff() * 100.0 # Assuming Vega is per 1% IV change
    df['dt'] = df['time'].diff().dt.total_seconds() / 86400.0 # days
    df['dP_actual'] = df['Price'].diff()
    
    # Trapezoidal approximation (average of t and t-1)
    delta_avg = (df['Delta'] + df['Delta'].shift(1)) / 2.0
    gamma_avg = (df['Gamma'] + df['Gamma'].shift(1)) / 2.0
    theta_avg = (df['Theta'] + df['Theta'].shift(1)) / 2.0
    vega_avg = (df['Vega'] + df['Vega'].shift(1)) / 2.0
    
    # Attribute PnL
    df['pnl_delta'] = delta_avg * df['dS']
    df['pnl_gamma'] = 0.5 * gamma_avg * (df['dS'] ** 2)
    # Theta is usually given in per-day, so multiply by dt in days
    # Wait, our dt is in days, but BS Theta is usually per day decay
    # Let's align: if theta is per day, dt must be in days.
    df['pnl_theta'] = theta_avg * df['dt']
    df['pnl_vega'] = vega_avg * df['dIV']
    
    # Calculate theoretical PnL change
    df['dP_theoretical'] = df['pnl_delta'] + df['pnl_gamma'] + df['pnl_theta'] + df['pnl_vega']
    
    # Residual is the unexplained portion
    df['pnl_residual'] = df['dP_actual'] - df['dP_theoretical']
    
    # Sum up totals
    total_delta = df['pnl_delta'].sum()
    total_gamma = df['pnl_gamma'].sum()
    total_theta = df['pnl_theta'].sum()
    total_vega = df['pnl_vega'].sum()
    total_residual = df['pnl_residual'].sum()
    actual_pnl = df['dP_actual'].sum()
    
    # Event Flag generation
    flagged = False
    flag_reason = ""
    threshold = initial_premium * 0.05
    
    if abs(total_residual) > threshold:
        flagged = True
        flag_reason = f"Residual Error ({total_residual:.2f}) exceeded 5% of Initial Premium ({threshold:.2f})."
        
    return {
        "pnl_delta": total_delta,
        "pnl_gamma": total_gamma,
        "pnl_theta": total_theta,
        "pnl_vega": total_vega,
        "pnl_residual": total_residual,
        "total_actual_pnl": actual_pnl,
        "event_flag": flagged,
        "flag_reason": flag_reason,
        "attribution_df": df # Return the dataframe for visualization
    }
