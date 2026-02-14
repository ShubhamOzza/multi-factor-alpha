# Multi-Factor Equity Alpha Model

Long/short equity strategy built on risk-adjusted momentum across a 50-stock S&P 500 universe. Backtested 2014–2024 with full transaction costs.

---

## Project Structure

```
multi-factor-alpha/
├── main.py           ← Run this
├── config.py         ← All settings (universe, dates, parameters)
├── data.py           ← Downloads & caches price data
├── factors.py        ← Signal construction (momentum + vol)
├── portfolio.py      ← Portfolio construction + backtest engine
├── metrics.py        ← Sharpe, Sortino, drawdown, win rate
├── visualize.py      ← Charts + exports equity curve for website
└── requirements.txt
```

---

## Step-by-Step Setup

### Step 1 — Make sure Python is installed

Open Terminal (Mac) or Command Prompt (Windows) and run:

```bash
python --version
```

You need Python 3.10 or higher. If you don't have it, download from [python.org](https://www.python.org/downloads/).

---

### Step 2 — Download the project files

If you cloned from GitHub:

```bash
git clone https://github.com/shubhamozza/multi-factor-alpha.git
cd multi-factor-alpha
```

If you downloaded the ZIP, unzip it and open your terminal inside that folder.

---

### Step 3 — Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Mac / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

You'll see `(venv)` appear in your terminal. That means it's active.

---

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: numpy, pandas, scipy, yfinance, matplotlib, pyarrow.

---

### Step 5 — Run the backtest

```bash
python main.py
```

The first run downloads 10 years of price data for 50 stocks. This takes about 60–90 seconds. After that it's cached locally in `prices.parquet` and every future run is instant.

You will see output like this:

```
 Multi-Factor Equity Alpha Model
 ─────────────────────────────────

[1/5]  Fetching price data...
       48 stocks  |  2014-01-02 → 2023-12-29

[2/5]  Computing factor signals...
       Average signal coverage: 43 stocks per day

[3/5]  Running backtest...
       2516 trading days simulated

[4/5]  Performance summary:
       Annualized Return      14.82%
       Annualized Volatility  11.34%
       Sharpe Ratio           1.31
       Sortino Ratio          1.87
       Calmar Ratio           0.94
       Max Drawdown          -15.76%
       Win Rate               53.20%
       Total Return          304.17%

[5/5]  Generating outputs...
  Saved → report.png
  Saved → equity_curve.json  (paste into website)
  Saved → equity_curve.csv
  Saved → daily_returns.csv
```

---

### Step 6 — Check your outputs

After running you'll have 4 new files in the folder:

| File | What it is |
|---|---|
| `report.png` | Full backtest report with 5 charts |
| `equity_curve.json` | JavaScript array for your website |
| `equity_curve.csv` | Raw equity curve data |
| `daily_returns.csv` | Daily return series |

Open `report.png` to see your charts.

---

## Updating Your Portfolio Website

This is how you replace the simulated demo charts in `index.html` with your real backtest results.

### Step 1 — Open equity_curve.json

Open the file in any text editor. It looks like this:

```js
const EQUITY_CURVE = [100.0, 101.34, 99.82, 103.45, 105.21, ...];
```

Copy the entire line.

---

### Step 2 — Open index.html

Find this block in the `<script>` section near the bottom:

```js
function generateEquityCurve() {
    const rng = seededRng(42);
    const n = 252 * 5;
    const eq = [100];
    ...
    return eq;
}

const eq = generateEquityCurve();
```

---

### Step 3 — Replace the simulated data

Delete the entire `generateEquityCurve` function and the line `const eq = generateEquityCurve();`.

Replace both with the single line you copied from `equity_curve.json`:

```js
const EQUITY_CURVE = [100.0, 101.34, 99.82, ...]; // your actual data
const eq = EQUITY_CURVE;
```

---

### Step 4 — Save and open index.html in your browser

Open `index.html` in Chrome or Safari. The hero panel now shows your real backtest equity curve and computes the real Sharpe and max drawdown from it.

---

### Step 5 — Push to GitHub

```bash
git add index.html
git commit -m "Update hero charts with real backtest data"
git push
```

Your live website updates automatically within a few minutes.

---

## Changing Parameters

All settings are in `config.py`. Common things to tweak:

| Setting | Default | What it does |
|---|---|---|
| `LONG_N` / `SHORT_N` | 15 | Number of long / short positions |
| `REBALANCE_DAYS` | 21 | How often to rebalance (~monthly) |
| `COST_BPS` | 10 | Transaction cost per unit of turnover |
| `MOM_LOOKBACK` | 252 | Momentum lookback window (1 year) |
| `MOM_SKIP` | 21 | Skip most recent month (standard) |

After changing anything, delete `prices.parquet` if you changed the date range, then rerun `python main.py`.

---

## How the Strategy Works

1. Each day, compute 12-1 month momentum for every stock (12 months back, skip last month)
2. Divide each stock's momentum by its 21-day realized volatility → risk-adjusted signal
3. Z-score the cross-section so every stock is comparable on the same scale
4. Every 21 days, go long the top 15 and short the bottom 15 stocks, equal-weighted
5. Deduct 10bps transaction cost on turnover at each rebalance
