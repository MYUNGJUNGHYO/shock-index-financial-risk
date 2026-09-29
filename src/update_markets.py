"""Download daily historical KOSPI and USD/KRW series for the static dashboard."""
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

SERIES = {'kospi': ('KOSPI', '^KS11'), 'usdkrw': ('USD/KRW', 'KRW=X')}
output = {'updated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'provider': 'Yahoo Finance via yfinance', 'markets': {}}
for key, (name, ticker) in SERIES.items():
    frame = yf.download(ticker, period='5y', interval='1d', auto_adjust=True, progress=False, threads=False, timeout=30)
    if frame.empty:
        raise RuntimeError(f'No data returned for {ticker}; keeping previous deployment')
    close = frame['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    clean = pd.DataFrame({'date': pd.to_datetime(close.index).strftime('%Y-%m-%d'), 'close': pd.to_numeric(close.values, errors='coerce')})
    clean = clean.dropna().loc[lambda d: d.close > 0].drop_duplicates('date', keep='last').sort_values('date')
    if len(clean) < 65:
        raise RuntimeError(f'Insufficient observations for {ticker}: {len(clean)}')
    output['markets'][key] = {'name': name, 'ticker': ticker, 'latest_date': clean.date.iloc[-1], 'rows': [{'date': r.date, 'close': round(float(r.close), 6)} for r in clean.itertuples(index=False)]}
    print(f'{name}: {len(clean)} rows, latest {clean.date.iloc[-1]}')
path = Path('data/markets.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(output, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
