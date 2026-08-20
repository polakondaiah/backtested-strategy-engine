#!/bin/bash
set -e
# Backtested Trading Strategy Engine - automated pipeline
# Closes Linux/shell gap: validation, timestamps, logging.

LOG="pipeline.log"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== Pipeline start ==="
log "PWD: $PROJECT_DIR"
log "Python: $(python3 --version)"

if [ ! -f "data/prices.csv" ]; then
  log "Fetching data..."
  python3 fetch_data.py 2>&1 | tee -a "$LOG"
else
  log "Data exists: $(wc -l < data/prices.csv) lines in data/prices.csv"
fi

# validation
log "Validating data..."
python3 -c "
import pandas as pd
from pathlib import Path
df=pd.read_csv('data/prices.csv')
assert not df.empty, 'empty prices.csv'
assert {'Date','Ticker','Close'}.issubset(df.columns), 'missing cols'
print(f'rows={len(df)} tickers={df.Ticker.nunique()} nulls={df.isna().sum().sum()}')
# missing dates
import pandas as pd
dates=pd.to_datetime(df.Date)
print(f'range {dates.min()} -> {dates.max()}')
" 2>&1 | tee -a "$LOG"

log "Running backtest..."
python3 -m src.backtest 2>&1 | tee -a "$LOG"

log "Benchmark (Python vs C++)..."
# try to build C++ if not present
if ! python3 -c "import pnl_cpp" 2>/dev/null; then
  log "Building C++ extension..."
  c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) src/cpp/pnl.cpp -o pnl_cpp$(python3-config --extension-suffix) 2>&1 | tee -a "$LOG" || log "C++ build failed (continuing with Python-only)"
fi
python3 benchmark.py 2>&1 | tee -a "$LOG"

log "Pipeline done. Results in results/"
ls -lh results/ 2>&1 | tee -a "$LOG"

# Cron example (uncomment to automate daily at 6am):
# 0 6 * * * cd /path/to/project && bash run_pipeline.sh >> pipeline.log 2>&1
