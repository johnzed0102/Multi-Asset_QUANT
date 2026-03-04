from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


DATA_PATH = Path("/Users/john/Desktop/Multi-Asset_QUANT/processed_data/asset_return_panel.csv")
OUTPUT_DIR = Path("/Users/john/Desktop/Multi-Asset_QUANT/回测引擎")

ASSETS = ["hs300", "sp500", "cgb10y", "gold", "policy_bond", "credit_bond"]
WINDOW = 252
TARGET_VOL = 0.05


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "date" not in df.columns:
        raise ValueError("asset_return_panel.csv 缺少 date 列")

    missing_assets = [asset for asset in ASSETS if asset not in df.columns]
    if missing_assets:
        raise ValueError(f"asset_return_panel.csv 缺少资产列: {missing_assets}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="first")
    df = df[df["date"] >= pd.Timestamp("2016-01-04")].reset_index(drop=True)

    for asset in ASSETS:
        df[asset] = pd.to_numeric(df[asset], errors="coerce")

    return df[["date", *ASSETS]]


def identify_rebalance_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    grouped = df.groupby(df["date"].dt.to_period("M"))["date"].max().sort_values()
    return pd.DatetimeIndex(grouped.values)


def compute_covariance(window_returns: pd.DataFrame, eligible_assets: list[str]) -> pd.DataFrame:
    cov = window_returns[eligible_assets].cov()
    cov = cov.replace([np.inf, -np.inf], np.nan)
    cov = cov.dropna(how="all", axis=0).dropna(how="all", axis=1)

    if cov.empty:
        return cov

    cov = cov.reindex(index=eligible_assets, columns=eligible_assets)
    if cov.isna().any().any():
        return pd.DataFrame()

    cov_values = cov.to_numpy(dtype=float)
    cov_values = (cov_values + cov_values.T) / 2.0
    cov_values = cov_values + np.eye(len(eligible_assets)) * 1e-10
    return pd.DataFrame(cov_values, index=eligible_assets, columns=eligible_assets)


def solve_risk_parity(cov: pd.DataFrame) -> np.ndarray:
    n_assets = cov.shape[0]
    if n_assets == 1:
        return np.array([1.0])

    sigma = cov.to_numpy(dtype=float)
    init_w = np.full(n_assets, 1.0 / n_assets)

    def objective(w: np.ndarray) -> float:
        port_var = float(w.T @ sigma @ w)
        if port_var <= 0:
            return 1e6
        mrc = sigma @ w
        rc = w * mrc / port_var
        target_rc = np.full(n_assets, 1.0 / n_assets)
        return float(np.sum((rc - target_rc) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n_assets)]

    result = minimize(
        objective,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not result.success or np.any(np.isnan(result.x)):
        return init_w

    w = np.clip(result.x, 0.0, 1.0)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return init_w
    return w / w_sum


def apply_vol_target(weights: np.ndarray, cov: pd.DataFrame, target_vol: float = TARGET_VOL) -> tuple[np.ndarray, float, float]:
    sigma = cov.to_numpy(dtype=float)
    port_var = float(weights.T @ sigma @ weights)
    realized_vol = float(np.sqrt(max(0.0, 252.0 * port_var)))

    if realized_vol > 0:
        scale = min(1.0, target_vol / realized_vol)
    else:
        scale = 1.0

    scaled_weights = weights * scale
    cash_weight = float(1.0 - scaled_weights.sum())
    return scaled_weights, cash_weight, realized_vol


def compute_performance(returns: pd.Series) -> dict[str, float]:
    clean_ret = returns.dropna()
    if clean_ret.empty:
        return {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
        }

    n = len(clean_ret)
    cumulative = float((1.0 + clean_ret).prod())
    annual_return = cumulative ** (252.0 / n) - 1.0
    annual_vol = float(clean_ret.std(ddof=0) * np.sqrt(252.0))
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    nav = (1.0 + clean_ret).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rebalance_dates = identify_rebalance_dates(df)
    indexed = df.set_index("date")

    weight_records = []
    rebalance_weight_map: dict[pd.Timestamp, pd.Series] = {}
    stage_records = []

    previous_rebalance_weights = None
    first_valid_rebalance_date = None

    for reb_date in rebalance_dates:
        row = {"date": reb_date, **{asset: np.nan for asset in ASSETS}, "cash": np.nan}

        hist = indexed.loc[:reb_date, ASSETS]
        if len(hist) < WINDOW:
            row["asset_count"] = np.nan
            weight_records.append(row)
            continue

        window_returns = hist.tail(WINDOW)
        eligible_assets = [
            asset
            for asset in ASSETS
            if pd.notna(window_returns.iloc[-1][asset]) and window_returns[asset].notna().any()
        ]

        if len(eligible_assets) == 0:
            row["asset_count"] = 0
            weight_records.append(row)
            continue

        cov = compute_covariance(window_returns, eligible_assets)
        if cov.empty:
            row["asset_count"] = len(eligible_assets)
            weight_records.append(row)
            continue

        w_rp = solve_risk_parity(cov)
        w_scaled, cash_weight, _ = apply_vol_target(w_rp, cov, target_vol=TARGET_VOL)

        new_weights = pd.Series(0.0, index=ASSETS + ["cash"], dtype="float64")
        new_weights.loc[eligible_assets] = w_scaled
        new_weights.loc["cash"] = cash_weight

        if previous_rebalance_weights is not None:
            smoothed = 0.7 * previous_rebalance_weights + 0.3 * new_weights
            ineligible_assets = [asset for asset in ASSETS if asset not in eligible_assets]
            smoothed.loc[ineligible_assets] = 0.0
            smoothed.loc["cash"] = 1.0 - smoothed.loc[ASSETS].sum()
            final_weights = smoothed
        else:
            final_weights = new_weights

        if first_valid_rebalance_date is None:
            first_valid_rebalance_date = reb_date

        previous_rebalance_weights = final_weights
        rebalance_weight_map[reb_date] = final_weights

        for col in ASSETS + ["cash"]:
            row[col] = float(final_weights[col])
        row["asset_count"] = len(eligible_assets)
        weight_records.append(row)

        stage_records.append({"date": reb_date, "asset_count": len(eligible_assets)})

    weights_full = pd.DataFrame(weight_records).sort_values("date").reset_index(drop=True)

    daily_weights = pd.DataFrame(index=indexed.index, columns=ASSETS + ["cash"], dtype="float64")
    current_weights = pd.Series(np.nan, index=ASSETS + ["cash"], dtype="float64")
    for dt in indexed.index:
        if dt in rebalance_weight_map:
            current_weights = rebalance_weight_map[dt]
        daily_weights.loc[dt] = current_weights.values

    lagged_weights = daily_weights[ASSETS].shift(1)
    returns_only = indexed[ASSETS]

    portfolio_returns = []
    for dt in indexed.index:
        w = lagged_weights.loc[dt]
        r = returns_only.loc[dt]

        if w.isna().all():
            portfolio_returns.append(np.nan)
            continue

        pos_weight_missing_ret = ((w.fillna(0) > 0) & r.isna()).any()
        if pos_weight_missing_ret:
            portfolio_returns.append(np.nan)
            continue

        value = float((w.fillna(0.0) * r.fillna(0.0)).sum())
        portfolio_returns.append(value)

    nav_df = pd.DataFrame({
        "date": indexed.index,
        "portfolio_return": portfolio_returns,
    })
    nav_df["nav"] = (1.0 + nav_df["portfolio_return"].fillna(0.0)).cumprod()

    backtest_mask = (nav_df["date"] >= pd.Timestamp("2018-01-01")) & (nav_df["date"] <= pd.Timestamp("2023-12-31"))
    valid_mask = (nav_df["date"] >= pd.Timestamp("2024-01-01")) & (nav_df["date"] <= pd.Timestamp("2025-12-01"))

    backtest_stats = compute_performance(nav_df.loc[backtest_mask, "portfolio_return"])
    valid_stats = compute_performance(nav_df.loc[valid_mask, "portfolio_return"])

    results_df = pd.DataFrame(
        [backtest_stats, valid_stats],
        index=["backtest_2018_2023", "validation_2024_2025"],
    ).reset_index().rename(columns={"index": "period"})

    print("1. 第一个有效调仓日:", first_valid_rebalance_date.date() if first_valid_rebalance_date else None)

    stage_df = pd.DataFrame(stage_records)
    print("2. 每个阶段资产数量变化情况:")
    if stage_df.empty:
        print("- 无有效调仓记录")
    else:
        stage_df = stage_df.sort_values("date")
        change_points = stage_df[stage_df["asset_count"].ne(stage_df["asset_count"].shift(1))]
        for _, row in change_points.iterrows():
            print(f"- {row['date'].date()}: asset_count={int(row['asset_count'])}")

    valid_weights = weights_full.dropna(subset=ASSETS + ["cash"], how="all")
    if len(valid_weights) >= 2:
        max_change = float(valid_weights[ASSETS + ["cash"]].diff().abs().max().max())
    else:
        max_change = np.nan
    print("3. 最大单次权重变化幅度:", max_change)

    avg_cash = float(valid_weights["cash"].mean()) if not valid_weights.empty else np.nan
    print("4. 平均现金比例:", avg_cash)

    print("5. 回测区间与验证区间统计结果:")
    print(results_df)

    weights_out = weights_full[["date", *ASSETS, "cash"]].copy()
    return weights_out, nav_df, results_df


def save_outputs(weights_df: pd.DataFrame, nav_df: pd.DataFrame, results_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "weights.csv"
    nav_path = output_dir / "nav_series.csv"
    results_path = output_dir / "backtest_results.csv"

    weights_df.to_csv(weights_path, index=False)
    nav_df.to_csv(nav_path, index=False)
    results_df.to_csv(results_path, index=False)

    print("\n已保存文件:")
    print("-", weights_path)
    print("-", nav_path)
    print("-", results_path)


def main() -> None:
    data = load_data(DATA_PATH)
    weights_df, nav_df, results_df = run_backtest(data)
    save_outputs(weights_df, nav_df, results_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()