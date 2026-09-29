"""Build a compact daily-close universe of currently listed KOSPI equities."""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd

listing = fdr.StockListing('KOSPI')
if listing is None or listing.empty:
    raise RuntimeError('KOSPI listing is empty')
code_column = next((col for col in ('Code', 'Symbol') if col in listing.columns), None)
if code_column is None or 'Name' not in listing.columns:
    raise RuntimeError(f'Unexpected KOSPI listing columns: {list(listing.columns)}')

names = {}
for code_value, name_value in listing[[code_column, 'Name']].itertuples(index=False, name=None):
    code = str(code_value).strip().zfill(6)
    if len(code) == 6 and code.isdigit():
        names[code + '.KS'] = str(name_value).strip()

stocks = {}
failures = []
tickers = list(names)
start = f'{datetime.now(timezone.utc).year - 1}-01-01'

def load_stock(ticker):
    code = ticker[:6]
    for attempt in range(3):
        try:
            frame = fdr.DataReader('NAVER:' + code, start)
            if frame is None or frame.empty:
                raise ValueError('empty data')
            close = frame['Close']
            clean = pd.DataFrame({'date': pd.to_datetime(close.index).strftime('%Y-%m-%d'),
                                  'close': pd.to_numeric(close.to_numpy(), errors='coerce')})
            clean = clean.dropna().loc[lambda d: d.close > 0].drop_duplicates('date', keep='last').sort_values('date').tail(255)
            if len(clean) < 65:
                raise ValueError(f'only {len(clean)} rows')
            return code, {'name': names[ticker], 'rows': [[r.date, round(float(r.close), 4)]
                                                    for r in clean.itertuples(index=False)]}, None
        except Exception as exc:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return code, None, f'{ticker}: {exc}'

with ThreadPoolExecutor(max_workers=6) as pool:
    for index, future in enumerate(as_completed([pool.submit(load_stock, ticker) for ticker in tickers]), 1):
        code, entry, error = future.result()
        if entry:
            stocks[code] = entry
        else:
            failures.append(error)
        if index % 50 == 0 or index == len(tickers):
            print(f'Processed {index}/{len(tickers)}; usable {len(stocks)}', flush=True)

# A partial Yahoo outage must not silently turn a universe ranking into a tiny sample.
if len(stocks) < max(100, int(len(tickers) * 0.6)):
    raise RuntimeError(f'Only {len(stocks)}/{len(tickers)} KOSPI tickers returned usable data')

payload = {'updated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
           'provider': 'FinanceDataReader KOSPI listing and Naver daily closes',
           'listed_count': len(tickers), 'loaded_count': len(stocks),
           'period': '1y', 'stocks': stocks}
path = Path('data/stocks.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(f'Saved {len(stocks)}/{len(tickers)}; unavailable {len(failures)}')
