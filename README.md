# Gold AI Agent V9 — Quant Mathematics & Machine Learning Lab

Standalone research dashboard prototype built with Streamlit.

## Included modules

- Overview and synthetic OHLC candlestick chart
- Statistics: return distribution, rolling volatility, Z-score
- Numerical calculus: velocity, acceleration, jerk
- Linear algebra: correlation/covariance matrices and PCA
- Econometrics: OLS regression and residual diagnostics
- Machine learning: Random Forest, Logistic Regression, confusion matrix, feature importance
- Monte Carlo simulation and illustrative risk metrics
- Toy optimization surface
- Black–Scholes educational Greeks sandbox
- Validation Lab and System Monitor
- Downloadable synthetic dataset

## Important limitations

This prototype uses **synthetic data only**. It does not connect to Twelve Data, Exness, or any live data source. The auxiliary DXY, yield, and silver series are synthetic proxies, not real market data. Machine-learning outputs are educational demonstrations, not live trading signals. There is no automatic order execution.

This is a standalone prototype and does not replace or modify the main Gold AI Agent V9 strategy.

## Deploy on Streamlit Community Cloud

1. Upload `app.py` and `requirements.txt` to the root of a GitHub repository.
2. Open https://share.streamlit.io/ and connect the repository.
3. Select the branch (usually `main`) and set the main file path to `app.py`.
4. Deploy.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
