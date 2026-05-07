import os
from datetime import datetime
from xml.etree import ElementTree as ET

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


MODEL_DIR = "predictions"
RL_ARTIFACT_DIR = "rl_artifacts"
RL_BACKTEST_DIR = "rl_backtests"
FEATURE_COLS = [
    "High",
    "Low",
    "Open",
    "Volume",
    "MA_7",
    "MA_21",
    "Volatility_7",
    "Volatility_21",
    "High_Low_Spread",
    "Open_Close_Change",
] + [f"Return_lag{i}" for i in range(1, 31)]
COMPANY_TICKERS = {
    "Reliance": "RELIANCE.NS",
    "Adani": "ADANIENT.NS",
    "Mahindra_Mahindra": "M&M.NS",
    "Maruti_Suzuki": "MARUTI.NS",
    "Nestle_India": "NESTLEIND.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Larsen & Toubro": "LT.NS",
}


st.set_page_config(page_title="Fincast Live Dashboard", layout="wide")


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        return str(value)
    except Exception:
        return os.getenv(name, default)


NEWS_API_KEY = get_secret("NEWS_API_KEY", "")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(13,148,136,0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(234,88,12,0.14), transparent 22%),
            #0a0f1a;
    }
    .block-container {
        padding-top: 2.2rem;
        max-width: 1280px;
    }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        overflow: hidden;
    }
    .fincast-panel {
        background: linear-gradient(180deg, rgba(15,23,42,0.78), rgba(15,23,42,0.58));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 22px;
        padding: 1.05rem 1.2rem;
        margin-bottom: 1rem;
    }
    .fincast-kicker {
        color: #94a3b8;
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .fincast-state {
        display: inline-block;
        margin: 0.15rem 0.35rem 0.15rem 0;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.18);
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_currency(value: float) -> str:
    return f"₹ {value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def infer_signal_strength(q_values: list[float]) -> tuple[str, str]:
    ordered = sorted(q_values, reverse=True)
    edge = ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
    if edge >= 0.015:
        return "High conviction", "#22c55e"
    if edge >= 0.005:
        return "Moderate conviction", "#f59e0b"
    return "Low conviction", "#94a3b8"


def render_state_buckets(state_buckets: dict) -> None:
    chips = "".join(
        f"<span class='fincast-state'>{key.replace('_', ' ')}: {value}</span>"
        for key, value in state_buckets.items()
    )
    st.markdown(chips, unsafe_allow_html=True)


def annualized_return(daily_returns: pd.Series) -> float:
    growth = (1 + daily_returns).prod()
    periods = len(daily_returns)
    if periods == 0 or growth <= 0:
        return 0.0
    return growth ** (252 / periods) - 1


def sharpe_ratio(daily_returns: pd.Series) -> float:
    volatility = daily_returns.std(ddof=0)
    if volatility == 0 or np.isnan(volatility):
        return 0.0
    return np.sqrt(252) * daily_returns.mean() / volatility


def max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1
    return float(drawdown.min()) if not drawdown.empty else 0.0


def summarize_strategy(daily_returns: pd.Series, equity_curve: pd.Series) -> dict:
    wins = (daily_returns > 0).mean() if len(daily_returns) else 0.0
    return {
        "cumulative_return": float(equity_curve.iloc[-1] - 1) if len(equity_curve) else 0.0,
        "annualized_return": float(annualized_return(daily_returns)),
        "sharpe_ratio": float(sharpe_ratio(daily_returns)),
        "max_drawdown": float(max_drawdown(equity_curve)),
        "win_rate": float(wins),
    }


def encode_state(row: pd.Series, bins: dict) -> int:
    state_parts = []
    for column in ("Return_lag1", "Trend_Signal", "Volatility_7", "Momentum_3"):
        bucket = np.digitize(row[column], bins[column][1:-1], right=False)
        state_parts.append(int(bucket))

    state = 0
    base = 1
    for bucket in state_parts:
        state += bucket * base
        base *= 3
    return state


def decode_state_details(row: pd.Series, bins: dict) -> dict:
    details = {}
    for column in ("Return_lag1", "Trend_Signal", "Volatility_7", "Momentum_3"):
        details[column] = int(np.digitize(row[column], bins[column][1:-1], right=False))
    return details


def build_live_rl_frames(feature_frame: pd.DataFrame, rl_artifact: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    policy_frame = feature_frame.copy()
    policy_frame["Trend_Signal"] = (policy_frame["MA_7"] - policy_frame["MA_21"]) / policy_frame["Close"]
    policy_frame["Momentum_3"] = policy_frame["Return"].rolling(3).mean()
    policy_frame["Next_Return"] = policy_frame["Return"].shift(-1)
    policy_frame.dropna(subset=["Trend_Signal", "Momentum_3", "Return_lag1", "Volatility_7"], inplace=True)

    q_table = rl_artifact["q_table"]
    bins = rl_artifact["bins"]
    actions_map = {-1: "Sell", 0: "Hold", 1: "Buy"}

    policy_frame["State_ID"] = policy_frame.apply(lambda row: encode_state(row, bins), axis=1)
    policy_frame["Action"] = policy_frame["State_ID"].map(lambda state_id: int([-1, 0, 1][int(np.argmax(q_table[state_id]))]))
    policy_frame["Action_Label"] = policy_frame["Action"].map(actions_map)

    latest_policy_row = policy_frame.iloc[-1]
    latest_signal = {
        "date": str(policy_frame.index[-1].date()),
        "action": int(latest_policy_row["Action"]),
        "label": latest_policy_row["Action_Label"],
        "q_values": q_table[int(latest_policy_row["State_ID"])].tolist(),
        "state_id": int(latest_policy_row["State_ID"]),
        "state_buckets": decode_state_details(latest_policy_row, bins),
    }

    backtest_frame = policy_frame.dropna(subset=["Next_Return"]).copy()
    transaction_cost = rl_artifact.get("config", {}).get("transaction_cost", 0.001)
    transaction_costs = np.zeros(len(backtest_frame))
    if len(backtest_frame) > 1:
        transaction_costs[1:] = (
            backtest_frame["Action"].to_numpy()[1:] != backtest_frame["Action"].to_numpy()[:-1]
        ).astype(float) * transaction_cost

    backtest_frame["Strategy_Return"] = backtest_frame["Action"] * backtest_frame["Next_Return"] - transaction_costs
    backtest_frame["Buy_Hold_Return"] = backtest_frame["Next_Return"]
    backtest_frame["Strategy_Equity"] = (1 + backtest_frame["Strategy_Return"]).cumprod()
    backtest_frame["Buy_Hold_Equity"] = (1 + backtest_frame["Buy_Hold_Return"]).cumprod()

    metrics = summarize_strategy(backtest_frame["Strategy_Return"], backtest_frame["Strategy_Equity"])
    benchmark_metrics = summarize_strategy(backtest_frame["Buy_Hold_Return"], backtest_frame["Buy_Hold_Equity"])
    return policy_frame, backtest_frame, latest_signal, {"metrics": metrics, "benchmark_metrics": benchmark_metrics}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_history(ticker: str) -> pd.DataFrame:
    history = yf.download(ticker, period="5y", interval="1d", auto_adjust=False, progress=False)
    if history.empty:
        raise ValueError(f"No live data returned for {ticker}")

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    history = history.reset_index()
    history.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history.dropna(subset=["Date"], inplace=True)
    history.sort_values("Date", inplace=True)
    history.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    history.set_index("Date", inplace=True)
    return history


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    data["Return"] = data["Close"].pct_change()
    data["MA_7"] = data["Close"].rolling(7).mean()
    data["MA_21"] = data["Close"].rolling(21).mean()
    data["Volatility_7"] = data["Return"].rolling(7).std()
    data["Volatility_21"] = data["Return"].rolling(21).std()
    data["High_Low_Spread"] = data["High"] - data["Low"]
    data["Open_Close_Change"] = data["Open"] - data["Close"]

    for lag in range(1, 31):
        data[f"Return_lag{lag}"] = data["Return"].shift(lag)

    data.dropna(subset=["Close"] + FEATURE_COLS, inplace=True)
    return data


@st.cache_resource(show_spinner=False)
def load_rl_artifact(company: str) -> dict:
    return joblib.load(os.path.join(RL_ARTIFACT_DIR, f"{company}_q_learning.pkl"))


@st.cache_resource(show_spinner=False)
def load_xgb_model(company: str):
    return joblib.load(os.path.join(MODEL_DIR, f"{company}_xgb_model.pkl"))


def get_xgb_forecast(company: str, latest_row: pd.Series, feature_frame: pd.DataFrame) -> tuple[dict | None, str | None]:
    try:
        model = load_xgb_model(company)
        x_latest = feature_frame[FEATURE_COLS].iloc[[-1]]
        next_return = float(model.predict(x_latest)[0])
        predicted_close = float(latest_row["Close"] * (1 + next_return))
        importance = model.get_booster().get_score(importance_type="weight")
        importance_df = pd.DataFrame(
            {"Feature": list(importance.keys()), "Importance": list(importance.values())}
        ).sort_values(by="Importance", ascending=False)
        return {
            "predicted_close": predicted_close,
            "predicted_return": next_return,
            "importance_df": importance_df,
        }, None
    except Exception as exc:
        message = str(exc)
        if "libomp" in message or "OpenMP" in message:
            message = "XGBoost is installed, but the OpenMP runtime is missing on this machine."
        return None, message


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_feed(feed_url: str) -> list[dict]:
    response = requests.get(feed_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    root = ET.fromstring(response.content)

    articles = []
    for item in root.findall(".//item")[:8]:
        articles.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "url": (item.findtext("link") or "").strip(),
                "description": (item.findtext("description") or "").strip(),
                "published": (item.findtext("pubDate") or "").strip(),
            }
        )
    return articles


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_newsapi_articles(query: str | None = None) -> list[dict]:
    if not NEWS_API_KEY:
        return []

    params = {
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": 8,
        "sortBy": "publishedAt",
    }
    endpoint = "https://newsapi.org/v2/top-headlines"

    if query:
        endpoint = "https://newsapi.org/v2/everything"
        params["q"] = query
    else:
        params["category"] = "business"

    response = requests.get(endpoint, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    payload = response.json()
    return [
        {
            "title": item.get("title", "").strip(),
            "url": item.get("url", "").strip(),
            "description": (item.get("description") or "").strip(),
            "published": (item.get("publishedAt") or "").strip(),
            "source": (item.get("source") or {}).get("name", "").strip(),
        }
        for item in payload.get("articles", [])
        if item.get("title") and item.get("url")
    ]


def fetch_general_news() -> tuple[list[dict], str | None]:
    try:
        articles = fetch_newsapi_articles()
        if articles:
            return articles, None
        return fetch_news_feed(
            "https://news.google.com/rss/search?q=Indian%20stock%20market%20when:7d&hl=en-IN&gl=IN&ceid=IN:en"
        ), None
    except Exception as exc:
        try:
            return fetch_news_feed(
                "https://news.google.com/rss/search?q=Indian%20stock%20market%20when:7d&hl=en-IN&gl=IN&ceid=IN:en"
            ), str(exc)
        except Exception:
            return [], str(exc)


def fetch_company_news(company: str, ticker: str) -> tuple[list[dict], str | None]:
    try:
        query_string = f"{company.replace('_', ' ')} {ticker.split('.')[0]} stock"
        articles = fetch_newsapi_articles(query_string)
        if articles:
            return articles, None
        query = company.replace(" ", "%20").replace("&", "and")
        articles = fetch_news_feed(
            f"https://news.google.com/rss/search?q={query}%20{ticker.split('.')[0]}%20stock&hl=en-IN&gl=IN&ceid=IN:en"
        )
        return articles, None
    except Exception as exc:
        try:
            query = company.replace(" ", "%20").replace("&", "and")
            articles = fetch_news_feed(
                f"https://news.google.com/rss/search?q={query}%20{ticker.split('.')[0]}%20stock&hl=en-IN&gl=IN&ceid=IN:en"
            )
            return articles, str(exc)
        except Exception:
            return [], str(exc)


st.markdown("<div class='fincast-kicker'>Live Market Dashboard</div>", unsafe_allow_html=True)
st.title("Fincast")
st.subheader("Live stock data, XGBoost forecasting, and reinforcement-learning strategy insights.")

company = st.sidebar.selectbox("Select Company", sorted(COMPANY_TICKERS))
show_news = st.sidebar.checkbox("Show Live Market News", value=True)
refresh = st.sidebar.button("Refresh Live Data")
if refresh:
    st.cache_data.clear()
    st.cache_resource.clear()

ticker = COMPANY_TICKERS[company]

try:
    raw_history = fetch_live_history(ticker)
    feature_frame = prepare_features(raw_history)
    data_error = None
except Exception as exc:
    raw_history = None
    feature_frame = None
    data_error = str(exc)

rl_artifact = load_rl_artifact(company)

st.markdown(f"<div class='fincast-kicker'>Ticker</div><p>{ticker}</p>", unsafe_allow_html=True)

if data_error or feature_frame is None or feature_frame.empty:
    st.error(f"Unable to fetch live market data right now: {data_error}")
    st.stop()

latest_row = feature_frame.iloc[-1]
latest_raw = raw_history.loc[latest_row.name]
xgb_forecast, xgb_error = get_xgb_forecast(company, latest_row, feature_frame)
live_rl_policy, live_rl_backtest, live_rl_signal, live_rl_summary = build_live_rl_frames(feature_frame, rl_artifact)
signal_strength, signal_color = infer_signal_strength(live_rl_signal["q_values"])

st.subheader(f"{company} Live Snapshot")
top_left, top_mid, top_right = st.columns(3)
top_left.metric("Latest Close", format_currency(float(latest_raw["Close"])))
top_mid.metric("Day Change", format_percent(float(latest_row["Return_lag1"])))
top_right.metric("Q-Learning Signal", live_rl_signal["label"])

col1, col2 = st.columns([1.05, 1.7])
with col1:
    latest_ohlc = {
        "Open": float(latest_raw["Open"]),
        "High": float(latest_raw["High"]),
        "Low": float(latest_raw["Low"]),
        "Close": float(latest_raw["Close"]),
    }
    st.markdown("<div class='fincast-panel'>", unsafe_allow_html=True)
    st.write("Latest OHLC")
    ohlc_df = pd.DataFrame([latest_ohlc], index=[latest_row.name.strftime("%d %b %Y")]).map(
        lambda value: f"{value:,.2f}"
    )
    st.table(ohlc_df)
    st.caption(f"Volume: {int(latest_raw['Volume']):,}")
    st.markdown("</div>", unsafe_allow_html=True)

    metric_a, metric_b = st.columns(2)
    metric_a.metric("RL Return", format_percent(live_rl_summary["metrics"]["cumulative_return"]))
    metric_b.metric("Sharpe Ratio", f"{live_rl_summary['metrics']['sharpe_ratio']:.2f}")

    metric_c, metric_d = st.columns(2)
    metric_c.metric("Max Drawdown", format_percent(live_rl_summary["metrics"]["max_drawdown"]))
    metric_d.metric("Buy & Hold", format_percent(live_rl_summary["benchmark_metrics"]["cumulative_return"]))

    st.markdown(
        f"<div class='fincast-panel'><div class='fincast-kicker'>Policy Read</div>"
        f"<h4 style='margin:0.2rem 0 0.5rem 0; color:{signal_color};'>{signal_strength}</h4></div>",
        unsafe_allow_html=True,
    )
    st.write("Current RL state buckets")
    render_state_buckets(live_rl_signal["state_buckets"])

with col2:
    last_two_years = raw_history[raw_history.index >= (raw_history.index.max() - pd.DateOffset(years=2))]
    price_fig = go.Figure()
    price_fig.add_trace(
        go.Scatter(
            x=last_two_years.index,
            y=last_two_years["Close"],
            mode="lines",
            name="Live Close Price",
            line=dict(color="#0F766E", width=3),
        )
    )
    if xgb_forecast:
        price_fig.add_trace(
            go.Scatter(
                x=[latest_row.name, latest_row.name + pd.Timedelta(days=1)],
                y=[float(latest_raw["Close"]), xgb_forecast["predicted_close"]],
                mode="lines+markers",
                name="XGBoost Forecast",
                line=dict(color="#EA580C", width=3, dash="dot"),
            )
        )

    price_fig.update_layout(
        title=f"{company} Live Close Price History",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=470,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.35)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(price_fig, use_container_width=True)

tabs = st.tabs(["XGBoost Forecast", "Q-Learning Strategy", "News"])

with tabs[0]:
    st.subheader("Supervised Next-Day Forecast")
    if xgb_forecast:
        pred_col1, pred_col2 = st.columns(2)
        pred_col1.metric("Predicted Close Price", format_currency(xgb_forecast["predicted_close"]))
        pred_col2.metric("Predicted Return", f"{xgb_forecast['predicted_return']:.4%}")

        fig_imp = px.bar(
            xgb_forecast["importance_df"],
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            height=540,
            title="Feature Importance",
        )
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(2,6,23,0.35)",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("XGBoost forecast is unavailable.")
        if xgb_error:
            st.caption(xgb_error)

with tabs[1]:
    st.subheader("Reinforcement Learning Backtest")
    st.caption(
        f"Live policy replay through {live_rl_signal['date']}. "
        f"The last saved artifact snapshot was {rl_artifact['latest_signal']['date']}."
    )
    equity_fig = go.Figure()
    equity_fig.add_trace(
        go.Scatter(
            x=live_rl_backtest.index,
            y=live_rl_backtest["Strategy_Equity"],
            mode="lines",
            name="Q-Learning Equity",
            line=dict(color="#EA580C", width=3),
        )
    )
    equity_fig.add_trace(
        go.Scatter(
            x=live_rl_backtest.index,
            y=live_rl_backtest["Buy_Hold_Equity"],
            mode="lines",
            name="Buy & Hold Equity",
            line=dict(color="#1D4ED8", width=2),
        )
    )
    equity_fig.update_layout(
        template="plotly_dark",
        height=500,
        yaxis_title="Growth of 1.0",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.35)",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(equity_fig, use_container_width=True)

    recent_signals = live_rl_policy[["Close", "Action_Label"]].tail(10).copy()
    realized_returns = live_rl_backtest["Strategy_Return"].reindex(recent_signals.index)
    recent_signals["Strategy_Return"] = realized_returns
    recent_signals = recent_signals.reset_index().rename(columns={"index": "Date"})
    recent_signals["Date"] = recent_signals["Date"].dt.strftime("%d %b %Y")
    recent_signals["Close"] = recent_signals["Close"].map(lambda value: f"{value:,.2f}")
    recent_signals["Strategy_Return"] = recent_signals["Strategy_Return"].map(
        lambda value: format_percent(value) if pd.notna(value) else "Pending"
    )
    recent_signals.columns = ["Date", "Close", "Action", "Strategy Return"]
    st.write("Recent RL decisions")
    st.dataframe(recent_signals, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Latest Market News")
    if not show_news:
        st.info("Enable live market news from the sidebar to load headlines.")
    else:
        company_articles, company_news_error = fetch_company_news(company, ticker)
        general_articles, general_news_error = fetch_general_news()

        if company_articles:
            st.markdown("#### Company News")
            for article in company_articles[:5]:
                st.markdown(f"### [{article['title']}]({article['url']})")
                if article["description"]:
                    st.write(article["description"])
                if article["published"]:
                    st.caption(article["published"])
                st.markdown("---")

        if general_articles:
            st.markdown("#### Market News")
            for article in general_articles[:5]:
                st.markdown(f"### [{article['title']}]({article['url']})")
                if article["description"]:
                    st.write(article["description"])
                if article["published"]:
                    st.caption(article["published"])
                st.markdown("---")

        if not company_articles and not general_articles:
            st.warning("Unable to load live news right now.")
            if company_news_error:
                st.caption(f"Company news error: {company_news_error}")
            if general_news_error:
                st.caption(f"Market news error: {general_news_error}")

st.markdown(
    f"""
    ---
    <div style='text-align: center;'>
        <p>Live data source: Yahoo Finance | News source: Google News RSS</p>
        <p>Last app refresh: {datetime.now().strftime("%d %b %Y, %I:%M %p")}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
