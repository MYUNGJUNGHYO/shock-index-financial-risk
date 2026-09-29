"""Reproducible, descriptive Shock Index calculation. No fitted SVJ parameters."""
import numpy as np
import pandas as pd


def analyze(frame: pd.DataFrame, window: int = 20, threshold: float = 3.7, hybrid: bool = False) -> pd.DataFrame:
    if window < 2 or threshold <= 0:
        raise ValueError('window must be >= 2 and threshold must be positive')
    if not {'date', 'close'} <= set(frame.columns):
        raise ValueError('CSV requires date and close columns')
    out = frame[['date', 'close']].copy()
    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    out['close'] = pd.to_numeric(out['close'], errors='coerce')
    out = out.dropna().loc[lambda d: d.close > 0].sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    if len(out) <= window + 1:
        raise ValueError('Need more than window + 1 valid observations')
    out['log_return'] = np.log(out.close).diff()
    out['return_pct'] = out.close.pct_change() * 100
    out['volatility_lagged'] = out.log_return.rolling(window).std(ddof=1).shift(1)
    out['shock_index'] = out.log_return.abs().div(out.volatility_lagged.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out['detected'] = out.shock_index.ge(threshold)
    if hybrid:
        out['detected'] |= out.return_pct.abs().ge(4)
    out['direction'] = np.where(out.detected, np.where(out.log_return > 0, 'up', 'down'), 'normal')
    return out
