# ================================================================
# GOLD AI AGENT V9
# QUANT MATHEMATICS & MACHINE LEARNING LAB
# STANDALONE PROTOTYPE | SYNTHETIC DATA | NO API KEY | NO AUTO ORDER
# ================================================================

import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

st.set_page_config(
    page_title="Gold AI Agent V9 | Quant Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG, PANEL, BLUE, WHITE, GREEN, RED, YELLOW = (
    "#07111F", "#0D1B2A", "#39A7FF", "#EAF2FF",
    "#00D395", "#FF5263", "#FFC857"
)

st.markdown("""
<style>
.stApp {background:#07111F;color:#EAF2FF;}
[data-testid="stSidebar"] {background:#0A1725;}
h1,h2,h3 {color:#EAF2FF !important;}
div[data-testid="stMetric"] {
 background:#0D1B2A; border:1px solid #20364D;
 padding:13px; border-radius:8px;
}
div[data-testid="stMetricValue"] {color:#39A7FF;}
div[data-testid="stMetricLabel"] {color:#8EA3BB;}
.block-container {padding-top:1.1rem;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("⚡ GOLD AI AGENT V9")
st.sidebar.caption("QUANTITATIVE RESEARCH TERMINAL")
st.sidebar.warning("SIMULATION MODE — NOT LIVE MARKET DATA")

page = st.sidebar.radio("QUANT LAB MODULE", [
    "Overview", "Statistics", "Calculus", "Linear Algebra",
    "Econometrics", "Machine Learning", "Monte Carlo & Risk",
    "Optimization", "Greeks & Portfolio", "Validation Lab",
    "System Monitor"
])

st.sidebar.divider()
st.sidebar.subheader("Simulation Settings")
seed = st.sidebar.number_input("Random seed", 1, 999999, 42, 1)
n_bars = st.sidebar.slider("Synthetic observations", 500, 3000, 1500, 100)
start_price = st.sidebar.number_input("Synthetic starting price", 100.0, 10000.0, 4200.0, 10.0)
drift = st.sidebar.slider("Simulated drift", -0.0005, 0.0005, 0.00002, 0.00001, format="%.5f")
volatility = st.sidebar.slider("Simulated volatility", 0.0005, 0.01, 0.002, 0.0005, format="%.4f")

if st.sidebar.button("🔄 Generate New Simulation", use_container_width=True):
    st.session_state["simulation_seed"] = int(seed) + 1
effective_seed = int(st.session_state.get("simulation_seed", seed))

@st.cache_data(show_spinner=False)
def generate_data(n, seed_value, initial_price, mu, sigma):
    rg = np.random.default_rng(seed_value)
    regime = np.ones(n)
    regime[n // 3: 2 * n // 3] = 1.6
    regime[2 * n // 3:] = 0.8
    returns = mu + rg.normal(0, sigma, n) * regime
    close = initial_price * np.exp(np.cumsum(returns))
    open_ = np.r_[initial_price, close[:-1]]
    intrabar = np.abs(rg.normal(sigma * 0.65, sigma * 0.35, n))
    high = np.maximum(open_, close) * (1 + intrabar)
    low = np.minimum(open_, close) * (1 - intrabar)
    volume = rg.lognormal(10, 0.35, n)
    time = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq="min")
    df = pd.DataFrame({
        "time": time, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "log_return": np.log(close / np.r_[initial_price, close[:-1]])
    })
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_20"] = df["close"].pct_change(20)
    df["volatility_20"] = df["log_return"].rolling(20).std()
    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
    smooth = df["close"].rolling(5, min_periods=1).mean().to_numpy()
    df["velocity"] = np.gradient(smooth)
    df["acceleration"] = np.gradient(df["velocity"].to_numpy())
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    # Auxiliary synthetic assets: these are not actual DXY, yields, or silver.
    dxy_r = -0.25 * df["log_return"].fillna(0).to_numpy() + rg.normal(0, sigma, n)
    yield_r = -0.10 * df["log_return"].fillna(0).to_numpy() + rg.normal(0, sigma * 0.7, n)
    silver_r = 0.60 * df["log_return"].fillna(0).to_numpy() + rg.normal(0, sigma * 0.8, n)
    df["dxy_proxy"] = 100 * np.exp(np.cumsum(dxy_r))
    df["yield_proxy"] = 4 * np.exp(np.cumsum(yield_r))
    df["silver_proxy"] = 30 * np.exp(np.cumsum(silver_r))
    df["future_return"] = df["close"].shift(-5) / df["close"] - 1
    df["target_up"] = (df["future_return"] > 0).astype(int)
    return df

df = generate_data(int(n_bars), effective_seed, float(start_price), float(drift), float(volatility))
FEATURES = ["momentum_5", "momentum_20", "volatility_20", "zscore_20",
            "velocity", "acceleration", "range_pct", "volume_z"]
feature_data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["future_return"]).copy()

def finish_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
        font_color=WHITE, height=height,
        margin=dict(l=15, r=15, t=55, b=15),
        legend=dict(orientation="h")
    )
    return fig

def line_fig(data, x, y, title):
    return finish_fig(px.line(data, x=x, y=y, title=title))

def metrics(items):
    cols = st.columns(len(items))
    for col, (label, value, help_) in zip(cols, items):
        col.metric(label, value, help=help_)

st.title("GOLD AI AGENT V9")
st.markdown("**QUANT MATHEMATICS · MACHINE LEARNING · RESEARCH TERMINAL**")
st.caption(f"SYNTHETIC DATA ONLY | Seed {effective_seed} | {len(df):,} bars | {datetime.now():%Y-%m-%d %H:%M:%S}")
st.info("Prototype mandiri. Tidak terhubung ke Twelve Data atau Exness. Tidak mengirim order. Seluruh data dan cross-asset proxy adalah sintetis.")

if page == "Overview":
    r = df["log_return"].dropna()
    total = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    metrics([
        ("Synthetic Price", f"{df.close.iloc[-1]:,.2f}", "Bukan harga live"),
        ("Total Return", f"{total:.2%}", "Sepanjang simulasi"),
        ("Return Std Dev", f"{r.std():.6f}", "Per bar"),
        ("Observations", f"{len(df):,}", "Bar sintetis"),
    ])
    left, right = st.columns([2, 1])
    with left:
        fig = go.Figure(go.Candlestick(
            x=df.time, open=df.open, high=df.high, low=df.low, close=df.close,
            increasing_line_color=BLUE, decreasing_line_color=WHITE,
            increasing_fillcolor=BLUE, decreasing_fillcolor=WHITE
        ))
        fig.update_layout(title="Synthetic Candlestick Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(finish_fig(fig, 450), use_container_width=True)
    with right:
        st.subheader("Quant Snapshot")
        st.metric("Mean Log Return", f"{r.mean():.6f}")
        st.metric("Skewness", f"{r.skew():.3f}")
        st.metric("Excess Kurtosis", f"{r.kurtosis():.3f}")
        st.metric("20-Bar Momentum", f"{df.momentum_20.iloc[-1]:.2%}")
    st.plotly_chart(line_fig(df, "time", "volume", "Synthetic Volume"), use_container_width=True)

elif page == "Statistics":
    r = df.log_return.dropna()
    metrics([
        ("Mean", f"{r.mean():.6f}", "Average log return"),
        ("Std Dev", f"{r.std():.6f}", "Return dispersion"),
        ("Skewness", f"{r.skew():.3f}", "Distribution asymmetry"),
        ("Excess Kurtosis", f"{r.kurtosis():.3f}", "Tail shape statistic"),
    ])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(finish_fig(px.histogram(r, nbins=60, histnorm="probability density", title="Log Return Distribution")), use_container_width=True)
    with c2:
        st.plotly_chart(finish_fig(px.box(r, points="outliers", title="Return Outliers")), use_container_width=True)
    st.plotly_chart(line_fig(df, "time", "volatility_20", "Rolling 20-Bar Volatility"), use_container_width=True)
    st.plotly_chart(line_fig(df, "time", "zscore_20", "Rolling Price Z-Score"), use_container_width=True)
    st.caption("Volatility mengukur variasi return, bukan arah. Z-score adalah jarak harga dari rata-rata rolling dalam satuan standar deviasi.")

elif page == "Calculus":
    window = st.slider("Smoothing window", 1, 30, 5)
    smooth = df.close.rolling(window, min_periods=1).mean().to_numpy()
    velocity_ = np.gradient(smooth)
    acceleration_ = np.gradient(velocity_)
    jerk = np.gradient(acceleration_)
    calc = pd.DataFrame({"time": df.time, "smoothed_price": smooth, "velocity": velocity_, "acceleration": acceleration_, "jerk": jerk})
    metrics([
        ("Velocity", f"{velocity_[-1]:.5f}", "Price units/bar"),
        ("Acceleration", f"{acceleration_[-1]:.6f}", "Change in velocity"),
        ("Jerk", f"{jerk[-1]:.7f}", "Change in acceleration"),
    ])
    for col, title in [("smoothed_price", "Smoothed Price"), ("velocity", "First Derivative — Velocity"),
                       ("acceleration", "Second Derivative — Acceleration"), ("jerk", "Third Derivative — Jerk")]:
        st.plotly_chart(line_fig(calc, "time", col, title), use_container_width=True)
    st.warning("Turunan numerik sensitif terhadap noise. Ini bukan sinyal trading.")

elif page == "Linear Algebra":
    returns_matrix = pd.DataFrame({
        "XAU_proxy": df.close.pct_change(), "DXY_proxy": df.dxy_proxy.pct_change(),
        "Yield_proxy": df.yield_proxy.pct_change(), "Silver_proxy": df.silver_proxy.pct_change()
    }).dropna()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(finish_fig(px.imshow(returns_matrix.corr(), text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", title="Pearson Correlation Matrix")), use_container_width=True)
    with c2:
        st.plotly_chart(finish_fig(px.imshow(returns_matrix.cov(), text_auto=".6f",
                    color_continuous_scale="Viridis", title="Return Covariance Matrix")), use_container_width=True)
    standardized = StandardScaler().fit_transform(returns_matrix)
    eigvals, eigvecs = np.linalg.eigh(np.cov(standardized, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    explained = eigvals / eigvals.sum()
    pca = pd.DataFrame({"Component": [f"PC{i+1}" for i in range(len(explained))],
                        "Explained Variance": explained, "Cumulative Variance": np.cumsum(explained)})
    st.plotly_chart(finish_fig(px.bar(pca, x="Component", y="Explained Variance", title="PCA — Explained Variance")), use_container_width=True)
    st.dataframe(pca, use_container_width=True)
    st.subheader("PCA Loadings")
    st.dataframe(pd.DataFrame(eigvecs, index=returns_matrix.columns,
                  columns=[f"PC{i+1}" for i in range(len(explained))]).style.format("{:.4f}"), use_container_width=True)
    st.caption("Cross-asset series di sini adalah proxy sintetis, bukan data pasar aktual.")

elif page == "Econometrics":
    econ = pd.DataFrame({"y": df.log_return, "x1": df.dxy_proxy.pct_change(),
                         "x2": df.yield_proxy.pct_change(), "x3": df.silver_proxy.pct_change()}).dropna()
    X = np.column_stack([np.ones(len(econ)), econ[["x1", "x2", "x3"]].to_numpy()])
    y = econ.y.to_numpy()
    split = int(len(econ) * 0.75)
    beta = np.linalg.lstsq(X[:split], y[:split], rcond=None)[0]
    actual, pred = y[split:], X[split:] @ beta
    residual = actual - pred
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - np.sum(residual**2) / ss_tot if ss_tot else np.nan
    metrics([("OOS R²", f"{r2:.4f}", "Chronological test segment"),
             ("RMSE", f"{np.sqrt(np.mean(residual**2)):.6f}", "Test-set error"),
             ("MAE", f"{np.mean(np.abs(residual)):.6f}", "Test-set error")])
    st.dataframe(pd.DataFrame({"Variable": ["Intercept", "DXY proxy", "Yield proxy", "Silver proxy"], "Coefficient": beta}), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(finish_fig(px.scatter(x=actual, y=pred, labels={"x":"Actual","y":"Predicted"}, title="OOS Actual vs Predicted")), use_container_width=True)
    with c2:
        fig = px.scatter(x=pred, y=residual, labels={"x":"Predicted","y":"Residual"}, title="Residuals vs Predicted")
        fig.add_hline(y=0, line_dash="dash", line_color=YELLOW)
        st.plotly_chart(finish_fig(fig), use_container_width=True)
    st.warning("OLS ini demonstrasi dengan proxy sintetis; koefisien tidak menggambarkan hubungan ekonomi nyata.")

elif page == "Machine Learning":
    ml = feature_data.copy()
    X_all = ml[FEATURES]
    y_all = ml.target_up.astype(int)
    split = int(len(ml) * 0.75)
    X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
    y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]
    if len(X_test) < 30 or y_train.nunique() < 2:
        st.error("Data belum cukup untuk evaluasi ML.")
    else:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        rf = RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=8,
                                    class_weight="balanced", random_state=effective_seed)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=effective_seed)
        rf.fit(X_train, y_train)
        lr.fit(X_train_s, y_train)
        pred = rf.predict(X_test)
        prob = rf.predict_proba(X_test)[:, 1]
        lr_pred = lr.predict(X_test_s)
        metrics([
            ("RF Accuracy", f"{accuracy_score(y_test, pred):.2%}", "Classification accuracy"),
            ("RF Precision", f"{precision_score(y_test, pred, zero_division=0):.2%}", "Positive predictive value"),
            ("RF Recall", f"{recall_score(y_test, pred, zero_division=0):.2%}", "True positive rate"),
            ("RF F1", f"{f1_score(y_test, pred, zero_division=0):.3f}", "Harmonic mean"),
        ])
        try:
            st.metric("Random Forest ROC-AUC", f"{roc_auc_score(y_test, prob):.3f}")
        except ValueError:
            st.metric("Random Forest ROC-AUC", "N/A")
        c1, c2 = st.columns(2)
        with c1:
            cm = confusion_matrix(y_test, pred, labels=[0, 1])
            st.plotly_chart(finish_fig(px.imshow(cm, text_auto=True,
                x=["Predicted Down", "Predicted Up"], y=["Actual Down", "Actual Up"],
                title="Random Forest Confusion Matrix", color_continuous_scale="Blues")), use_container_width=True)
        with c2:
            imp = pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_}).sort_values("Importance")
            st.plotly_chart(finish_fig(px.bar(imp, x="Importance", y="Feature", orientation="h",
                title="Feature Importance")), use_container_width=True)
        comparison = pd.DataFrame({
            "Model": ["Random Forest", "Logistic Regression"],
            "Accuracy": [accuracy_score(y_test, pred), accuracy_score(y_test, lr_pred)],
            "Precision": [precision_score(y_test, pred, zero_division=0), precision_score(y_test, lr_pred, zero_division=0)],
            "Recall": [recall_score(y_test, pred, zero_division=0), recall_score(y_test, lr_pred, zero_division=0)],
            "F1": [f1_score(y_test, pred, zero_division=0), f1_score(y_test, lr_pred, zero_division=0)]
        })
        st.subheader("Model Comparison")
        st.dataframe(comparison.style.format({"Accuracy":"{:.2%}", "Precision":"{:.2%}", "Recall":"{:.2%}", "F1":"{:.3f}"}), use_container_width=True)
        prob_df = pd.DataFrame({"time": ml.time.iloc[split:].to_numpy(), "probability_up": prob})
        st.plotly_chart(line_fig(prob_df, "time", "probability_up", "Model Probability of Up Move"), use_container_width=True)
        st.warning("Target adalah arah return sintetis lima bar ke depan. Model ini bukan sinyal trading live.")

elif page == "Monte Carlo & Risk":
    c1, c2 = st.columns(2)
    with c1:
        n_paths = st.slider("Number of paths", 100, 1000, 300, 100)
    with c2:
        horizon = st.slider("Horizon (bars)", 30, 500, 120, 30)
    rg = np.random.default_rng(effective_seed + 777)
    s0 = float(df.close.iloc[-1])
    r = df.log_return.dropna()
    mu, sig = float(r.mean()), float(r.std())
    shocks = rg.normal(size=(horizon, n_paths))
    paths = np.zeros((horizon + 1, n_paths))
    paths[0] = s0
    for t in range(1, horizon + 1):
        paths[t] = paths[t-1] * np.exp((mu - 0.5*sig**2) + sig*shocks[t-1])
    fig = go.Figure()
    x = np.arange(horizon + 1)
    for i in range(min(30, n_paths)):
        fig.add_trace(go.Scatter(x=x, y=paths[:, i], mode="lines", line=dict(width=1), opacity=.3, showlegend=False))
    p05, p25, p50, p75, p95 = [np.percentile(paths, q, axis=1) for q in [5,25,50,75,95]]
    fig.add_trace(go.Scatter(x=x, y=p95, line=dict(width=0), showlegend=False, name="95th"))
    fig.add_trace(go.Scatter(x=x, y=p05, line=dict(width=0), fill="tonexty", fillcolor="rgba(57,167,255,.12)", name="5–95% interval"))
    fig.add_trace(go.Scatter(x=x, y=p75, line=dict(width=0), showlegend=False, name="75th"))
    fig.add_trace(go.Scatter(x=x, y=p25, line=dict(width=0), fill="tonexty", fillcolor="rgba(0,211,149,.14)", name="25–75% interval"))
    fig.add_trace(go.Scatter(x=x, y=p50, line=dict(color=BLUE, width=3), name="Median path"))
    fig.update_layout(title="Monte Carlo Price Fan Chart", xaxis_title="Future Bar")
    st.plotly_chart(finish_fig(fig, 500), use_container_width=True)
    ending_ret = paths[-1] / s0 - 1
    var95 = np.percentile(ending_ret, 5)
    cvar95 = ending_ret[ending_ret <= var95].mean()
    metrics([("Median Ending Price", f"{np.median(paths[-1]):,.2f}", "Median simulated outcome"),
             ("5th Percentile Return", f"{var95:.2%}", "Lower-tail outcome"),
             ("CVaR 5%", f"{cvar95:.2%}", "Average worst 5% returns")])
    st.plotly_chart(finish_fig(px.histogram(x=paths[-1], nbins=45, title="Ending Price Distribution")), use_container_width=True)
    med_eq = p50 / p50[0]
    dd = med_eq / np.maximum.accumulate(med_eq) - 1
    st.plotly_chart(finish_fig(px.area(x=x, y=dd, labels={"x":"Future Bar","y":"Drawdown"}, title="Median Path Drawdown")), use_container_width=True)
    st.warning("Monte Carlo bukan ramalan dan tidak memasukkan execution cost atau perubahan regime secara realistis.")

elif page == "Optimization":
    st.caption("Toy objective untuk menguji optimisasi dan grafik; bukan mencari parameter trading profitable.")
    a = np.linspace(-3, 3, 240)
    b = np.linspace(-3, 3, 240)
    A, B = np.meshgrid(a, b)
    Z = np.sin(A*1.5)*np.cos(B*1.2) - .12*(A**2+B**2) + .15*np.sin(A*B)
    idx = np.unravel_index(np.argmax(Z), Z.shape)
    best_a, best_b, best_z = A[idx], B[idx], Z[idx]
    metrics([("Best Parameter A", f"{best_a:.3f}", "Toy objective"),
             ("Best Parameter B", f"{best_b:.3f}", "Toy objective"),
             ("Objective Score", f"{best_z:.4f}", "Not a trading return")])
    fig = go.Figure(go.Contour(x=a, y=b, z=Z, colorscale="Viridis", contours=dict(showlabels=True)))
    fig.add_trace(go.Scatter(x=[best_a], y=[best_b], mode="markers", marker=dict(size=12, color=RED, symbol="x"), name="Best grid point"))
    fig.update_layout(title="Non-Convex Optimization Surface", xaxis_title="Parameter A", yaxis_title="Parameter B")
    st.plotly_chart(finish_fig(fig, 520), use_container_width=True)
    st.markdown("Research workflow: define objective and transaction costs, optimize on training data, validate separately, then run walk-forward and out-of-sample tests.")

elif page == "Greeks & Portfolio":
    st.subheader("Black–Scholes Educational Sandbox")
    c1, c2, c3 = st.columns(3)
    with c1:
        S = st.number_input("Underlying price (S)", 10.0, 10000.0, 4200.0, 10.0)
        K = st.number_input("Strike price (K)", 10.0, 10000.0, 4200.0, 10.0)
    with c2:
        T = st.slider("Time to expiry (years)", .01, 3.0, .25, .01)
        sig = st.slider("Annual volatility", .01, 1.5, .25, .01)
    with c3:
        rate = st.slider("Risk-free rate", -.02, .15, .04, .005)
        q = st.slider("Dividend/carry yield", 0.0, .10, 0.0, .005)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (rate-q+.5*sig**2)*T)/(sig*sqrtT)
    d2 = d1-sig*sqrtT
    cdf = lambda z: .5*(1+math.erf(z/math.sqrt(2)))
    pdf = lambda z: math.exp(-.5*z*z)/math.sqrt(2*math.pi)
    call = S*math.exp(-q*T)*cdf(d1)-K*math.exp(-rate*T)*cdf(d2)
    delta = math.exp(-q*T)*cdf(d1)
    gamma = math.exp(-q*T)*pdf(d1)/(S*sig*sqrtT)
    vega = S*math.exp(-q*T)*pdf(d1)*sqrtT/100
    theta = (-S*math.exp(-q*T)*pdf(d1)*sig/(2*sqrtT)-rate*K*math.exp(-rate*T)*cdf(d2)+q*S*math.exp(-q*T)*cdf(d1))/365
    rho = K*T*math.exp(-rate*T)*cdf(d2)/100
    metrics([("Call Value", f"{call:.4f}", "Theoretical value"),
             ("Delta", f"{delta:.4f}", "Sensitivity to underlying"),
             ("Gamma", f"{gamma:.6f}", "Change in delta"),
             ("Vega / vol point", f"{vega:.4f}", "Volatility sensitivity")])
    metrics([("Theta / day", f"{theta:.4f}", "Calendar-day approximation"),
             ("Rho / 1% rate", f"{rho:.4f}", "Rate sensitivity")])
    spots = np.linspace(max(1, S*.7), S*1.3, 120)
    values, deltas = [], []
    for spot in spots:
        d1i = (math.log(spot/K)+(rate-q+.5*sig**2)*T)/(sig*sqrtT)
        d2i = d1i-sig*sqrtT
        values.append(spot*math.exp(-q*T)*cdf(d1i)-K*math.exp(-rate*T)*cdf(d2i))
        deltas.append(math.exp(-q*T)*cdf(d1i))
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(finish_fig(px.line(x=spots, y=values, labels={"x":"Underlying","y":"Call value"}, title="Call Value vs Underlying")), use_container_width=True)
    with c2:
        st.plotly_chart(finish_fig(px.line(x=spots, y=deltas, labels={"x":"Underlying","y":"Delta"}, title="Call Delta vs Underlying")), use_container_width=True)
    rets = df[["close","dxy_proxy","yield_proxy","silver_proxy"]].pct_change().dropna().to_numpy()
    weights = np.array([.4,.2,.2,.2])
    cov = np.cov(rets, rowvar=False)
    portfolio_vol = math.sqrt(max(0, float(weights.T @ cov @ weights)))
    st.metric("Synthetic Portfolio Volatility per Bar", f"{portfolio_vol:.6f}")
    st.caption("Greeks ini untuk opsi vanilla teoritis, bukan valuasi langsung XAU/USD CFD. Seri portofolio sintetis.")

elif page == "Validation Lab":
    ml = feature_data.copy()
    X_all, y_all = ml[FEATURES], ml.target_up.astype(int)
    split = int(len(ml)*.75)
    X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
    y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]
    if len(X_test) < 30 or y_train.nunique() < 2:
        st.error("Sampel belum cukup untuk validasi.")
    else:
        model = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=8,
                                       class_weight="balanced", random_state=effective_seed)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:,1]
        pred = (prob >= .5).astype(int)
        baseline = np.full(len(y_test), int(y_train.mean() >= .5))
        metrics([("Model Accuracy", f"{accuracy_score(y_test,pred):.2%}", "Test split"),
                 ("Majority Baseline", f"{accuracy_score(y_test,baseline):.2%}", "Naive benchmark")])
        # Illustrative only: five-bar forward returns overlap.
        future_r = ml.future_return.iloc[split:].to_numpy()
        position = np.where(prob >= .5, 1.0, -1.0)
        cost = st.slider("Assumed cost per trade (return units)", 0.0, .001, .0001, .0001)
        net = position*future_r-cost
        equity = np.cumprod(1+net)
        drawdown = equity/np.maximum.accumulate(equity)-1
        metrics([("Toy Total Return", f"{equity[-1]-1:.2%}", "Not real strategy performance"),
                 ("Toy Max Drawdown", f"{drawdown.min():.2%}", "Illustrative"),
                 ("Mean Return", f"{net.mean():.6f}", "Overlapping targets")])
        out = pd.DataFrame({"time":ml.time.iloc[split:].to_numpy(),"equity":equity,"drawdown":drawdown})
        c1,c2 = st.columns(2)
        with c1:
            st.plotly_chart(line_fig(out,"time","equity","Toy Equity Curve"),use_container_width=True)
        with c2:
            st.plotly_chart(line_fig(out,"time","drawdown","Toy Drawdown"),use_container_width=True)
        st.warning("Not a valid trading backtest: targets overlap and sizing, realistic spread, slippage, and execution are omitted.")

elif page == "System Monitor":
    status = pd.DataFrame([
        ["Synthetic data engine","ONLINE","Generated locally"],
        ["Twelve Data API","DISABLED","No API key"],
        ["Exness feed","DISABLED","No broker connection"],
        ["Machine learning","ON-DEMAND","Trained on ML/validation pages"],
        ["Order execution","DISABLED","No automatic orders"],
        ["Live strategy modification","DISABLED","Standalone prototype"],
    ], columns=["Component","Status","Details"])
    st.dataframe(status, use_container_width=True, hide_index=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.tail(30), use_container_width=True)
    st.download_button("Download Synthetic Dataset CSV", df.to_csv(index=False).encode("utf-8"),
                       "gold_quant_synthetic_data.csv", "text/csv", use_container_width=True)

st.divider()
st.caption("GOLD AI AGENT V9 | QUANT LAB PROTOTYPE | SYNTHETIC DATA ONLY | NO AUTO ORDER | NOT FINANCIAL ADVICE")
