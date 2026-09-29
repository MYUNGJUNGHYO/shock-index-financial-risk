"""Build a compact daily-close universe of currently listed KOSPI equities."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf

listing = fdr.StockListing('KOSPI')
if listing is None or listing.empty or not {'Symbol', 'Name'}.issubset(listing.columns):
    raise RuntimeError('KOSPI listing is unavailable')

names = {}
for row in listing[['Symbol', 'Name']].itertuples(index=False):
    code = str(row.Symbol).strip().zfill(6)
    if len(code) == 6 and code.isdigit():
        names[code + '.KS'] = str(row.Name).strip()

stocks = {}
failures = []
tickers = list(names)
for offset in range(0, len(tickers), 25):
    batch = tickers[offset:offset + 25]
    frame = None
    for attempt in range(2):
        try:
            frame = yf.download(batch, period='1y', interval='1d', auto_adjust=True,
                                progress=False, threads=True, timeout=25, group_by='ticker')
            break
        except Exception as exc:
            print(f'Batch {offset // 25 + 1} attempt {attempt + 1}: {exc}')
            time.sleep(3)
    for ticker in batch:
        try:
            if frame is None or frame.empty:
                raise ValueError('empty batch')
            close = frame[ticker]['Close'] if isinstance(frame.columns, pd.MultiIndex) else frame['Close']
            clean = pd.DataFrame({'date': pd.to_datetime(close.index).strftime('%Y-%m-%d'),
                                  'close': pd.to_numeric(close.to_numpy(), errors='coerce')})
            clean = clean.dropna().loc[lambda d: d.close > 0].drop_duplicates('date', keep='last').sort_values('date')
            if len(clean) < 65:
                raise ValueError(f'only {len(clean)} rows')
            code = ticker[:6]
            stocks[code] = {'name': names[ticker], 'rows': [[r.date, round(float(r.close), 4)]
                                                      for r in clean.itertuples(index=False)]}
        except Exception as exc:
            failures.append(f'{ticker}: {exc}')
    print(f'Processed {min(offset + 25, len(tickers))}/{len(tickers)}; usable {len(stocks)}', flush=True)

# A partial Yahoo outage must not silently turn a universe ranking into a tiny sample.
if len(stocks) < max(100, int(len(tickers) * 0.6)):
    raise RuntimeError(f'Only {len(stocks)}/{len(tickers)} KOSPI tickers returned usable data')

payload = {'updated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
           'provider': 'FinanceDataReader listing; Yahoo Finance adjusted daily closes',
           'listed_count': len(tickers), 'loaded_count': len(stocks),
           'period': '1y', 'stocks': stocks}
path = Path('data/stocks.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(f'Saved {len(stocks)}/{len(tickers)}; unavailable {len(failures)}')
