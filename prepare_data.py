"""Convert a locally obtained date/close CSV to static-site JSON. No download or data redistribution."""
import argparse
import json
from pathlib import Path
import pandas as pd
from shock_index import analyze

p = argparse.ArgumentParser()
p.add_argument('input', help='CSV with date,close columns')
p.add_argument('output', help='JSON output path')
p.add_argument('--name', default='Uploaded market series')
p.add_argument('--window', type=int, default=20)
p.add_argument('--threshold', type=float, default=3.7)
a = p.parse_args()
result = analyze(pd.read_csv(a.input), a.window, a.threshold)
rows = [{'date': r.date.strftime('%Y-%m-%d'), 'close': round(r.close, 6)} for r in result.itertuples()]
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps({'name': a.name, 'kind': 'market', 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {len(rows)} rows to {a.output}')
