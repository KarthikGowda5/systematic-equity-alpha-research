# Systematic Equity Alpha Research & Portfolio Optimization Framework

An end-to-end quantitative equity research platform built on **CRSP** and **Compustat** data for developing, validating, combining, and implementing cross‑sectional alpha signals under realistic portfolio construction, cost, and risk constraints.

---

# Overview

This project builds a full institutional-style research pipeline for systematic equity strategies including:

• point‑in‑time data engineering  
• alpha signal construction  
• signal validation using Information Coefficient (IC) analysis  
• signal correlation analysis and orthogonalization  
• IC‑weighted composite alpha construction  
• long‑short portfolio backtesting  
• constrained portfolio optimization  
• transaction cost modeling  
• risk diagnostics and performance attribution  

The framework is designed to answer the practical research question:

**Which signals generate persistent cross‑sectional alpha, and how much survives realistic implementation constraints such as turnover, transaction costs, and risk‑neutral portfolio construction?**

---

# Research Pipeline

```
CRSP + Compustat + CCM Linking
            │
            ▼
Point‑in‑Time Panel Construction
            │
            ▼
Alpha Signal Library
            │
            ▼
Signal Validation (IC / Decay / Regimes)
            │
            ▼
Correlation Analysis + Orthogonalization
            │
            ▼
IC‑Weighted Composite Alpha
            │
            ▼
Backtesting + Portfolio Optimization
            │
            ▼
Transaction Costs + Risk Diagnostics
            │
            ▼
Performance Attribution
```

---

# Repository Structure

```
src/
├── data/               # data ingestion / merge / inspection utilities
├── alphas/             # alpha signal definitions
├── signals/            # validation / orthogonalization / signal research logic
├── backtest/           # engine, optimizer, costs, constraints, reporting
├── risk/               # risk diagnostics and stress/risk metrics
├── attribution/        # performance attribution logic
├── experiments/        # research sweep/orchestration tools
└── runs/               # CLI entry points

data/
├── raw/                # source data
├── processed/          # research-ready parquet panels
└── outputs/            # generated backtests, validation reports, sweeps

docs/                   # curated documentation / figures
research/               # curated experiment summaries
```

---

# Data

The project uses:

• **CRSP daily stock file** for prices, returns, and trading activity  
• **Compustat fundamentals** for accounting variables  
• **CCM linking table** to connect CRSP securities to Compustat firms  
• sector classification data  
• daily factor and market data where required  

The processed research panel includes:

• point‑in‑time linking  
• lagged fundamentals to avoid look‑ahead bias  
• duplicate resolution and sanity checks  
• liquidity filtering  
• derived features for signal construction  

---

# Alpha Signals

Implemented signals include:

• Operating Profitability (OP)  
• Book‑to‑Equity (BE)  
• 12‑1 Momentum  
• Residual Momentum  
• Short‑Term Reversal (STR)  
• Idiosyncratic Volatility  
• Beta‑based signals  
• Volatility autoregression features  

Signals are evaluated individually and in combination.

---

# Signal Validation

Signals are validated using:

• rolling cross‑sectional **Information Coefficient (IC)**  
• **Newey‑West adjusted t‑statistics**  
• forward return **decay curves** across horizons  
• rolling IC stability analysis  
• subperiod comparisons  
• regime comparisons (e.g., volatility regimes)

Outputs include:

• `ic_series.csv`  
• `rolling_ic.csv`  
• `decay.csv`  
• `ic_summary.json`  
• `signal_page.md`

---

# Composite Signal Construction

Multi‑signal alpha construction supports:

• cross‑signal correlation analysis  
• threshold‑based orthogonalization  
• cluster orthogonalization  
• IC‑weighted signal aggregation  
• incremental Sharpe analysis  

This helps build diversified composite signals while reducing redundant exposures.

---

# Portfolio Construction

The backtesting engine supports:

• configurable rebalance frequencies  
• cross‑sectional long‑short quantile portfolios  
• equal‑weight and optimized portfolios  
• liquidity filtering using ADV thresholds  

Core implementation is in:

`src/backtest/engine.py`

---

# Portfolio Optimization

The optimizer supports:

• market beta neutrality  
• sector neutrality  
• turnover penalties  
• gross exposure limits  
• covariance‑aware portfolio construction  

This enables comparison between:

• naive signal portfolios  
• unconstrained optimization  
• risk‑controlled optimization

---

# Transaction Cost Modeling

Implementation‑aware backtests include:

• bid‑ask spread costs  
• volatility‑based slippage  
• participation constraints  
• turnover‑driven costs

These allow evaluation of whether alpha survives realistic execution conditions.

---

# Risk Diagnostics

Risk tools monitor:

• market beta exposure  
• sector exposures  
• drawdowns and volatility  
• stress scenarios  
• factor diagnostics

Example scripts:

```
python src/runs/run_risk_report.py
python src/runs/analyse_optimizer_exposures.py
```

---

# Performance Attribution

The attribution layer decomposes returns into:

• factor‑driven return  
• residual alpha  
• transaction cost drag  
• sector or sleeve contribution

Example:

```
python src/runs/run_attribution.py
```

---

# Example Commands

Run alpha validation:

```
python src/runs/run_alpha_validation.py --alpha op
```

Build IC‑weighted composite:

```
python src/runs/run_IC_weightsavg.py \
  --signals op beta_sig STR \
  --orth \
  --orth_threshold 0.30 \
  --roll 252 \
  --horizon 5
```

Run baseline long‑short backtest:

```
python src/runs/run_baseline.py \
  --signal op \
  --data data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet \
  --set REBALANCE_FREQUENCY=W \
  --set REBALANCE_DAY=FRI \
  --set LONG_QUANTILE=0.9 \
  --set SHORT_QUANTILE=0.1 \
  --set UNIVERSE=ADV_1000
```

Run baseline with optimizer:

```
python src/runs/run_baseline.py \
  --signal composite_icw_rank \
  --data data/processed/panel_plus_composite.parquet \
  --set REBALANCE_FREQUENCY=W \
  --set REBALANCE_DAY=FRI \
  --set OPT_BETA_NEUTRAL=True \
  --set OPT_BETA_COL=beta_mkt \
  --set OPT_SECTOR_NEUTRAL=True \
  --set OPT_SECTOR_COL=sector \
  --set OPT_TURNOVER_GAMMA=0.05 \
  --use_optimizer 1
```

Run baseline with optimizer and transaction costs:

```
python src/runs/run_baseline.py \
  --signal composite_icw_rank \
  --data data/processed/panel_plus_composite.parquet \
  --set REBALANCE_FREQUENCY=W \
  --set REBALANCE_DAY=FRI \
  --set OPT_BETA_NEUTRAL=True \
  --set OPT_BETA_COL=beta_mkt \
  --set OPT_SECTOR_NEUTRAL=True \
  --set OPT_SECTOR_COL=sector \
  --set OPT_TURNOVER_GAMMA=0.05 \
  --set APPLY_SPREAD_COST=True \
  --set SPREAD_CAP=0.002 \
  --set APPLY_VOL_SLIPPAGE=True \
  --set VOL_SLIP_LOOKBACK=20 \
  --set VOL_SLIP_K=0.02 \
  --set VOL_CAP=0.05 \
  --set APPLY_TURNOVER_COST=True \
  --set TURNOVER_COST_PER_DOLLAR=0.0005 \
  --use_optimizer 1
```

Run frequency research sweep:

```
python src/experiments/sweep_frequency_research.py \
  --data data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet \
  --signals op be idiosyncratic_vol beta_sig momentum12_1 Residual_mom STR \
  --frequencies W BW TW M \
  --rebalance-day FRI
```

---

# Key Findings

Examples of insights generated by the framework:

• orthogonalized and IC‑weighted composite signals outperform individual signals on a gross basis  
• constrained optimization reduces unintended market beta and sector exposures  
• transaction costs significantly reduce performance for high‑turnover strategies  
• biweekly or triweekly rebalancing can provide a better turnover–decay tradeoff than weekly or monthly implementations  

---

# Reproducibility

The research workflow can be reproduced end-to-end using the following steps.

### 1. Build the research panel

python src/build_crsp_phase1.py  
python src/build_funda_phase2.py  
python src/Merge_final.py  

This creates the point-in-time research dataset:

data/processed/crsp_daily_with_lagged_funda_phase3.parquet

---

### 2. Validate alpha signals

python src/runs/run_alpha_validation.py --alpha op  
python src/runs/run_alpha_validation.py --alpha momentum12_1  
python src/runs/run_alpha_validation.py --alpha STR  

These generate IC statistics, decay curves, rolling IC diagnostics, and regime analysis.

---

### 3. Construct the composite alpha

python src/runs/run_IC_weightsavg.py \
  --signals op beta_sig STR momentum12_1 Residual_mom \
  --orth \
  --orth_threshold 0.30 \
  --roll 252 \
  --horizon 5

---

### 4. Run implementation-aware portfolio backtest

python src/runs/run_baseline.py \
  --signal composite_icw_rank \
  --data data/processed/panel_plus_composite.parquet \
  --set REBALANCE_FREQUENCY=W \
  --set REBALANCE_DAY=FRI \
  --set OPT_BETA_NEUTRAL=True \
  --set OPT_SECTOR_NEUTRAL=True \
  --set APPLY_SPREAD_COST=True \
  --set APPLY_VOL_SLIPPAGE=True \
  --use_optimizer 1

---

# Tech Stack

• Python  
• Pandas  
• NumPy  
• SciPy  
• CVXPY (portfolio optimization)  
• Parquet data storage  
• CRSP and Compustat datasets  

---

# Notes

This repository is designed as a **quantitative research platform**, not just a single backtest.

The goal is to study the full lifecycle of systematic equity strategies:

**raw data → alpha research → portfolio construction → implementation costs → risk & attribution**