import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def gen_synth_data(
    start_date: str = "2023-01-01",
    years: int = 3,
    base_value: float = 100.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic time series data with trend, seasonality, and noise.

    Parameters:
    - start_date (str): The starting date of the time series.
    - years (int): Number of years to generate data for.
    - base_value (float): The base value around which the time series fluctuates.

    Returns:
    - pd.DataFrame: A DataFrame containing the generated time series data.
    """
    try:
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        # Generate date range
        days = years * 365
        date_range = pd.date_range(start=start_date, periods=days, freq='D') # freq='D' for daily frequency
        # Generate synthetic data
        # Components: trend, seasonality, noise, holidays
        # Trend component (linear increase)
        trend = np.linspace(0, years * 3, days)
        # Seasonal component (quarterly seasonality)
        seasonal = 10 * np.sin(2 * np.pi * date_range.dayofyear /(365.25/4))
        # Noise component (with holiday spikes)
        noise = np.random.normal(0, 5, days)
        # Holiday spikes (randomly chosen days)
        holidays = np.random.choice(days, size=int(0.05 * days), replace=False)
        # Introduce spikes on holidays
        noise[holidays] += np.random.normal(20, 10, len(holidays))
        # Combine components (additive only, ensure non-negativity)
        values = base_value + trend + seasonal + noise
        # Ensure no negative values
        clipped_values = np.clip(values, a_min=0, a_max=None)
        # Create Time-series DataFrame
        synth_data = pd.DataFrame({'ds': date_range, 'y': clipped_values})
        return synth_data
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        return e, pd.DataFrame()

# Example usage
if __name__ == "__main__":
    data = gen_synth_data()
    print(data.head())
