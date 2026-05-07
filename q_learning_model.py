import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd


DATA_DIR = "features"
ARTIFACT_DIR = "rl_artifacts"
BACKTEST_DIR = "rl_backtests"
ACTIONS = np.array([-1, 0, 1])  # Sell, Hold, Buy
ACTION_LABELS = {-1: "Sell", 0: "Hold", 1: "Buy"}
RANDOM_SEED = 42


@dataclass
class QLearningConfig:
    episodes: int = 140
    alpha: float = 0.18
    gamma: float = 0.92
    epsilon_start: float = 0.30
    epsilon_end: float = 0.02
    transaction_cost: float = 0.001


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


def create_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data["Trend_Signal"] = (data["MA_7"] - data["MA_21"]) / data["Close"]
    data["Momentum_3"] = data["Return"].rolling(3).mean()
    data["Next_Return"] = data["Return"].shift(-1)
    data.dropna(inplace=True)
    return data


def build_state_bins(df: pd.DataFrame) -> dict:
    features = {
        "Return_lag1": [-np.inf, df["Return_lag1"].quantile(0.33), df["Return_lag1"].quantile(0.66), np.inf],
        "Trend_Signal": [-np.inf, df["Trend_Signal"].quantile(0.33), df["Trend_Signal"].quantile(0.66), np.inf],
        "Volatility_7": [-np.inf, df["Volatility_7"].quantile(0.33), df["Volatility_7"].quantile(0.66), np.inf],
        "Momentum_3": [-np.inf, df["Momentum_3"].quantile(0.33), df["Momentum_3"].quantile(0.66), np.inf],
    }
    return {name: np.array(edges, dtype=float) for name, edges in features.items()}


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


def train_q_learning(df: pd.DataFrame, config: QLearningConfig) -> tuple[np.ndarray, dict]:
    np.random.seed(RANDOM_SEED)
    bins = build_state_bins(df)
    states = df.apply(lambda row: encode_state(row, bins), axis=1).to_numpy(dtype=int)
    next_returns = df["Next_Return"].to_numpy(dtype=float)
    q_table = np.zeros((81, len(ACTIONS)), dtype=float)

    epsilon_values = np.linspace(config.epsilon_start, config.epsilon_end, config.episodes)
    for episode in range(config.episodes):
        epsilon = epsilon_values[episode]
        previous_action = 0
        for step in range(len(states) - 1):
            state = states[step]
            if np.random.random() < epsilon:
                action_index = np.random.randint(len(ACTIONS))
            else:
                action_index = int(np.argmax(q_table[state]))

            action = ACTIONS[action_index]
            reward = action * next_returns[step]
            if action != previous_action:
                reward -= config.transaction_cost

            next_state = states[step + 1]
            td_target = reward + config.gamma * np.max(q_table[next_state])
            q_table[state, action_index] += config.alpha * (td_target - q_table[state, action_index])
            previous_action = action

    return q_table, bins


def backtest_policy(df: pd.DataFrame, q_table: np.ndarray, bins: dict, config: QLearningConfig) -> pd.DataFrame:
    states = df.apply(lambda row: encode_state(row, bins), axis=1).to_numpy(dtype=int)
    action_indexes = q_table[states].argmax(axis=1)
    actions = ACTIONS[action_indexes]

    transaction_costs = np.zeros(len(actions))
    transaction_costs[1:] = (actions[1:] != actions[:-1]).astype(float) * config.transaction_cost
    strategy_returns = actions * df["Next_Return"].to_numpy(dtype=float) - transaction_costs
    buy_hold_returns = df["Next_Return"].to_numpy(dtype=float)

    results = pd.DataFrame(
        {
            "Date": df.index,
            "Close": df["Close"].to_numpy(dtype=float),
            "Next_Return": buy_hold_returns,
            "Action": actions,
            "Action_Label": [ACTION_LABELS[int(action)] for action in actions],
            "Strategy_Return": strategy_returns,
            "Buy_Hold_Return": buy_hold_returns,
        }
    )
    results["Strategy_Equity"] = (1 + results["Strategy_Return"]).cumprod()
    results["Buy_Hold_Equity"] = (1 + results["Buy_Hold_Return"]).cumprod()
    return results


def save_company_outputs(company_name: str, artifact: dict, backtest_df: pd.DataFrame) -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    joblib.dump(artifact, os.path.join(ARTIFACT_DIR, f"{company_name}_q_learning.pkl"))
    backtest_df.to_csv(os.path.join(BACKTEST_DIR, f"{company_name}_q_learning_backtest.csv"), index=False)


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    config = QLearningConfig()

    feature_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_features.csv")]
    for file_name in feature_files:
        company_name = file_name.replace("_features.csv", "")
        file_path = os.path.join(DATA_DIR, file_name)
        print(f"\nTraining Q-learning policy for {company_name}...")

        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        df = create_state_columns(df)

        q_table, bins = train_q_learning(df, config)
        backtest_df = backtest_policy(df, q_table, bins, config)
        metrics = summarize_strategy(backtest_df["Strategy_Return"], backtest_df["Strategy_Equity"])
        benchmark_metrics = summarize_strategy(backtest_df["Buy_Hold_Return"], backtest_df["Buy_Hold_Equity"])
        latest_row = df.iloc[-1]
        latest_state = encode_state(latest_row, bins)
        latest_action = int(ACTIONS[int(np.argmax(q_table[latest_state]))])

        artifact = {
            "company": company_name,
            "config": config.__dict__,
            "metrics": metrics,
            "benchmark_metrics": benchmark_metrics,
            "latest_signal": {
                "date": str(df.index[-1].date()),
                "action": latest_action,
                "label": ACTION_LABELS[latest_action],
                "q_values": q_table[latest_state].tolist(),
                "state_id": int(latest_state),
                "state_buckets": decode_state_details(latest_row, bins),
            },
            "q_table": q_table,
            "bins": bins,
        }
        save_company_outputs(company_name, artifact, backtest_df)

        print(f"  Strategy cumulative return: {metrics['cumulative_return']:.2%}")
        print(f"  Strategy sharpe ratio:      {metrics['sharpe_ratio']:.2f}")
        print(f"  Buy & hold cumulative:      {benchmark_metrics['cumulative_return']:.2%}")
        print(f"  Latest action:              {artifact['latest_signal']['label']}")

    print("\nQ-learning training complete for all companies.")


if __name__ == "__main__":
    main()
