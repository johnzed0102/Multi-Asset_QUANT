from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path("/Users/john/Desktop/Multi-Asset_QUANT")
RETURNS_PATH = ROOT_DIR / "processed_data/asset_return_panel.csv"
MACRO_PATH = ROOT_DIR / "processed_data/macro_panel.csv"
OUTPUT_DIR = ROOT_DIR / "回测引擎_macro_stable"

ASSETS = ["hs300", "sp500", "cgb10y", "gold", "policy_bond", "credit_bond"]
BASE_BUDGET = {
    "hs300": 0.20,
    "sp500": 0.20,
    "cgb10y": 0.20,
    "gold": 0.20,
    "policy_bond": 0.10,
    "credit_bond": 0.10,
}

WINDOW = 252
TARGET_VOL = 0.05
MACRO_TILT = 0.02

MAX_WEIGHT = 0.40
MAX_WEIGHT_CHANGE = 0.20
MIN_CASH = 0.05
TURNOVER_EPS = 1e-4

PERF_PERIODS = [
    ("2018-01-01_2023-12-31", pd.Timestamp("2018-01-01"), pd.Timestamp("2023-12-31")),
    ("2024-01-01_2025-12-01", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-01")),
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取输入数据，按时间升序并去重。"""
    ret_df = pd.read_csv(RETURNS_PATH)
    macro_df = pd.read_csv(MACRO_PATH)

    for col in ["date", *ASSETS]:
        if col not in ret_df.columns:
            raise ValueError(f"资产收益率缺少列: {col}")
    for col in ["date", "PMI", "CPI", "CN10Y"]:
        if col not in macro_df.columns:
            raise ValueError(f"宏观数据缺少列: {col}")

    ret_df["date"] = pd.to_datetime(ret_df["date"], errors="coerce")
    ret_df = ret_df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ASSETS:
        ret_df[col] = pd.to_numeric(ret_df[col], errors="coerce")

    macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce")
    macro_df = macro_df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["PMI", "CPI", "CN10Y"]:
        macro_df[col] = pd.to_numeric(macro_df[col], errors="coerce")

    return ret_df, macro_df


def identify_rebalance_dates(ret_df: pd.DataFrame) -> pd.DatetimeIndex:
    """识别每月最后一个交易日。"""
    month_last = ret_df.groupby(ret_df["date"].dt.to_period("M"))["date"].max().sort_values()
    return pd.DatetimeIndex(month_last.values)


def compute_covariance(window_returns: pd.DataFrame, eligible_assets: list[str]) -> pd.DataFrame:
    """计算协方差矩阵并进行轻微正定扰动。"""
    cov = window_returns[eligible_assets].cov().reindex(index=eligible_assets, columns=eligible_assets)
    if cov.isna().any().any():
        return pd.DataFrame()

    sigma = cov.to_numpy(dtype=float)
    sigma = (sigma + sigma.T) / 2.0
    sigma = sigma + np.eye(len(eligible_assets)) * 1e-10
    return pd.DataFrame(sigma, index=eligible_assets, columns=eligible_assets)


def solve_risk_budget_weights(cov: pd.DataFrame, budgets: np.ndarray, max_iter: int = 2000, tol: float = 1e-8) -> np.ndarray:
    """自定义迭代法求解风险预算权重（不使用外部优化器）。"""
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0], dtype=float)

    sigma = cov.to_numpy(dtype=float)
    b = np.maximum(budgets, 0.0)
    b = b / b.sum()

    w = np.maximum(b.copy(), 1e-12)
    w = w / w.sum()

    for _ in range(max_iter):
        sigma_w = sigma @ w
        rc = w * sigma_w
        rc_sum = rc.sum()
        if rc_sum <= 0:
            break

        rc_ratio = rc / rc_sum
        if np.max(np.abs(rc_ratio - b)) < tol:
            break

        target_rc = b * rc_sum
        ratio = target_rc / np.maximum(rc, 1e-12)
        w = w * np.power(ratio, 0.5)
        w = np.maximum(w, 1e-12)
        w = w / w.sum()

    return w


def get_macro_state(reb_date: pd.Timestamp, macro_df: pd.DataFrame) -> dict[str, int]:
    """仅使用历史信息计算当月宏观三维信号。"""
    macro = macro_df.copy()
    macro["period"] = macro["date"].dt.to_period("M")
    period = reb_date.to_period("M")

    current = macro[macro["period"] == period]
    if current.empty:
        current = macro[macro["period"] < period].tail(1)
        if current.empty:
            return {"growth": 0, "inflation": 0, "rate": 0}

    hist = macro[macro["period"] < period]
    if hist.empty:
        return {"growth": 0, "inflation": 0, "rate": 0}

    current_row = current.iloc[-1]
    med_pmi = hist["PMI"].median()
    med_cpi = hist["CPI"].median()
    med_cn10y = hist["CN10Y"].median()

    return {
        "growth": int(pd.notna(current_row["PMI"]) and pd.notna(med_pmi) and current_row["PMI"] > med_pmi),
        "inflation": int(pd.notna(current_row["CPI"]) and pd.notna(med_cpi) and current_row["CPI"] > med_cpi),
        "rate": int(pd.notna(current_row["CN10Y"]) and pd.notna(med_cn10y) and current_row["CN10Y"] > med_cn10y),
    }


def adjust_risk_budget(signals: dict[str, int], eligible_assets: list[str]) -> pd.Series:
    """基础预算 + 宏观冲击（±0.02 对称）后归一化。"""
    budget = pd.Series(BASE_BUDGET, dtype="float64")

    if signals.get("growth", 0) == 1:
        budget["hs300"] += MACRO_TILT
        budget["sp500"] += MACRO_TILT
        budget["cgb10y"] = max(0.0, budget["cgb10y"] - MACRO_TILT)
        budget["policy_bond"] = max(0.0, budget["policy_bond"] - MACRO_TILT)

    if signals.get("inflation", 0) == 1:
        budget["gold"] += MACRO_TILT
        budget["cgb10y"] = max(0.0, budget["cgb10y"] - MACRO_TILT / 2)
        budget["policy_bond"] = max(0.0, budget["policy_bond"] - MACRO_TILT / 2)

    if signals.get("rate", 0) == 1:
        moved = 0.0
        for asset in ["cgb10y", "policy_bond"]:
            cut = min(MACRO_TILT, budget[asset])
            budget[asset] -= cut
            moved += cut

        if "credit_bond" in eligible_assets:
            budget["credit_bond"] += moved
        else:
            budget["gold"] += moved

    budget = budget.clip(lower=0.0)
    budget = budget[eligible_assets]
    if budget.sum() <= 0:
        budget[:] = 1.0 / len(budget)
    else:
        budget = budget / budget.sum()

    return budget


def enforce_max_weight(weights: pd.Series, max_weight: float) -> pd.Series:
    """单资产上限：截断后在未达上限资产中按比例再分配。"""
    w = weights.copy().clip(lower=0.0)
    total = float(w.sum())
    if total <= 0:
        return w
    w = w / total

    for _ in range(20):
        over = w > max_weight
        if not over.any():
            break

        capped = w.copy()
        capped[over] = max_weight
        residual = 1.0 - capped[over].sum()
        free = (~over) & (w > 0)

        if residual <= 0 or not free.any():
            capped[:] = 0.0
            capped.loc[over] = 1.0 / over.sum()
            w = capped
            break

        free_sum = w[free].sum()
        capped[free] = w[free] / free_sum * residual
        capped[~over & ~free] = 0.0
        w = capped

    return w * total


def enforce_weight_change(weights_assets: pd.Series, prev_weights_assets: pd.Series, max_change: float) -> pd.Series:
    """单次权重变化限制（每资产绝对变动不超过阈值）。"""
    if prev_weights_assets is None:
        return weights_assets

    adjusted = weights_assets.copy()
    lower_bounds = pd.Series(0.0, index=ASSETS, dtype="float64")
    for asset in ASSETS:
        diff = adjusted[asset] - prev_weights_assets[asset]
        adjusted[asset] = prev_weights_assets[asset] + float(np.clip(diff, -max_change, max_change))
        lower_bounds[asset] = max(0.0, prev_weights_assets[asset] - max_change)

    adjusted = adjusted.clip(lower=0.0)
    total = float(adjusted.sum())
    if total > 1.0:
        excess = total - 1.0
        headroom = (adjusted - lower_bounds).clip(lower=0.0)
        available = float(headroom.sum())
        if available > 0:
            reduction = headroom / available * excess
            adjusted = (adjusted - reduction).clip(lower=0.0)
        else:
            adjusted = adjusted / total
    return adjusted


def apply_vol_target(weights_assets: pd.Series, cov: pd.DataFrame, target_vol: float) -> tuple[pd.Series, float, float]:
    """目标波动率缩放（不杠杆，scale<=1）。"""
    sigma = cov.to_numpy(dtype=float)
    w = weights_assets.loc[cov.index].to_numpy(dtype=float)

    port_var = float(w.T @ sigma @ w)
    annual_vol = float(np.sqrt(max(0.0, 252.0 * port_var)))

    scale = 1.0
    if annual_vol > target_vol and annual_vol > 0:
        scale = target_vol / annual_vol
    scale = min(1.0, scale)

    scaled = weights_assets * scale
    cash = float(max(0.0, 1.0 - scaled.sum()))
    return scaled, cash, annual_vol


def enforce_min_cash(weights_assets: pd.Series, cash_weight: float, min_cash: float) -> tuple[pd.Series, float]:
    """最低现金比例约束。"""
    w = weights_assets.copy()
    cash = float(cash_weight)

    if cash >= min_cash:
        return w, cash

    target_risky = 1.0 - min_cash
    current_risky = float(w.sum())
    if current_risky > 0:
        w = w * (target_risky / current_risky)
    return w, min_cash


def compute_performance(returns: pd.Series) -> dict[str, float]:
    """计算绩效指标。"""
    r = returns.dropna()
    if r.empty:
        return {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
        }

    n = len(r)
    total = float((1.0 + r).prod())
    annual_return = total ** (252.0 / n) - 1.0
    annual_vol = float(r.std(ddof=0) * np.sqrt(252.0))
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    nav = (1.0 + r).cumprod()
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


def run_backtest(ret_df: pd.DataFrame, macro_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """执行回测：严格遵循既定风控顺序。"""
    rebalance_dates = identify_rebalance_dates(ret_df)
    ret_idx = ret_df.set_index("date")

    rebalance_map: dict[pd.Timestamp, pd.Series] = {}
    weight_records = []
    prev_weights_assets = None

    for reb_date in rebalance_dates:
        row = {"date": reb_date, **{asset: np.nan for asset in ASSETS}, "cash": np.nan}
        hist = ret_idx.loc[:reb_date, ASSETS]
        if len(hist) < WINDOW:
            weight_records.append(row)
            continue

        window_returns = hist.tail(WINDOW)
        eligible_assets = [
            asset for asset in ASSETS if window_returns[asset].notna().any() and pd.notna(window_returns.iloc[-1][asset])
        ]
        if not eligible_assets:
            weight_records.append(row)
            continue

        cov = compute_covariance(window_returns, eligible_assets)
        if cov.empty:
            weight_records.append(row)
            continue

        # 1) 基础风险预算
        # 2) 宏观冲击
        signals = get_macro_state(reb_date, macro_df)
        budget = adjust_risk_budget(signals, eligible_assets)

        # 3) 风险平价权重计算
        w_rp = solve_risk_budget_weights(cov, budget.to_numpy(dtype=float))
        weights_assets = pd.Series(0.0, index=ASSETS, dtype="float64")
        weights_assets.loc[eligible_assets] = w_rp

        # 4) 单资产权重上限
        weights_assets.loc[eligible_assets] = enforce_max_weight(weights_assets.loc[eligible_assets], MAX_WEIGHT)

        # 5) 单次权重变化限制
        weights_assets = enforce_weight_change(weights_assets, prev_weights_assets, MAX_WEIGHT_CHANGE - TURNOVER_EPS)

        # 6) 目标波动率缩放
        weights_for_cov = weights_assets.loc[eligible_assets]
        weights_scaled, cash_weight, _ = apply_vol_target(weights_for_cov, cov, TARGET_VOL)
        weights_assets = pd.Series(0.0, index=ASSETS, dtype="float64")
        weights_assets.loc[eligible_assets] = weights_scaled

        # 7) 最低现金比例约束
        weights_assets, cash_weight = enforce_min_cash(weights_assets, cash_weight, MIN_CASH)

        final_assets = weights_assets.copy()

        # 最终一致性裁剪：确保输出权重变化也满足 max_weight_change
        final_assets = enforce_weight_change(final_assets, prev_weights_assets, MAX_WEIGHT_CHANGE - TURNOVER_EPS)
        final_assets = final_assets.clip(lower=0.0)
        if final_assets.sum() > 1.0:
            final_assets = final_assets / final_assets.sum()

        final_cash = float(max(0.0, 1.0 - final_assets.sum()))
        final_assets, final_cash = enforce_min_cash(final_assets, final_cash, MIN_CASH)

        final_full = pd.Series(0.0, index=ASSETS + ["cash"], dtype="float64")
        final_full.loc[ASSETS] = final_assets.values
        final_full.loc["cash"] = float(max(0.0, 1.0 - final_full.loc[ASSETS].sum()))

        rebalance_map[reb_date] = final_full
        prev_weights_assets = final_full.loc[ASSETS].copy()

        for col in ASSETS + ["cash"]:
            row[col] = float(final_full[col])
        weight_records.append(row)

    weights_full = pd.DataFrame(weight_records).sort_values("date").reset_index(drop=True)

    daily_weights = pd.DataFrame(index=ret_idx.index, columns=ASSETS + ["cash"], dtype="float64")
    current = pd.Series(np.nan, index=ASSETS + ["cash"], dtype="float64")
    for dt in ret_idx.index:
        if dt in rebalance_map:
            current = rebalance_map[dt]
        daily_weights.loc[dt] = current.values

    lagged_weights = daily_weights[ASSETS].shift(1)
    portfolio_returns = []
    for dt in ret_idx.index:
        w = lagged_weights.loc[dt]
        r = ret_idx.loc[dt, ASSETS]

        if w.isna().all():
            portfolio_returns.append(np.nan)
            continue

        missing_for_held = ((w.fillna(0.0) > 0) & r.isna()).any()
        if missing_for_held:
            portfolio_returns.append(np.nan)
            continue

        portfolio_returns.append(float((w.fillna(0.0) * r.fillna(0.0)).sum()))

    nav_df = pd.DataFrame({"date": ret_idx.index, "portfolio_return": portfolio_returns})
    nav_df["nav"] = (1.0 + nav_df["portfolio_return"].fillna(0.0)).cumprod()

    perf_rows = []
    for period_name, start_dt, end_dt in PERF_PERIODS:
        mask = (nav_df["date"] >= start_dt) & (nav_df["date"] <= end_dt)
        perf_rows.append({"period": period_name, **compute_performance(nav_df.loc[mask, "portfolio_return"])})
    results_df = pd.DataFrame(perf_rows)

    valid_weights = weights_full.dropna(subset=ASSETS + ["cash"], how="all")
    max_single_weight = float(valid_weights[ASSETS].max().max()) if not valid_weights.empty else np.nan
    max_weight_change = float(valid_weights[ASSETS].diff().abs().max().max()) if len(valid_weights) >= 2 else np.nan
    avg_cash = float(valid_weights["cash"].mean()) if not valid_weights.empty else np.nan

    vol_check_df = results_df[["period", "annual_volatility"]].copy()
    vol_check_df["vol_leq_target"] = vol_check_df["annual_volatility"] <= TARGET_VOL

    print("1）最大单资产权重:", max_single_weight)
    print("2）最大单次权重变化:", max_weight_change)
    print("3）平均现金比例:", avg_cash)
    print("4）两阶段绩效表:")
    print(results_df)
    print("5）两阶段年化波动率是否 <= 0.05:")
    print(vol_check_df)

    return weights_full[["date", *ASSETS, "cash"]].copy(), nav_df, results_df


def save_outputs(weights_df: pd.DataFrame, nav_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    """保存输出到稳定版目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weights_path = OUTPUT_DIR / "weights_macro_stable.csv"
    nav_path = OUTPUT_DIR / "nav_series_macro_stable.csv"
    results_path = OUTPUT_DIR / "backtest_results_macro_stable.csv"

    weights_df.to_csv(weights_path, index=False)
    nav_df.to_csv(nav_path, index=False)
    results_df.to_csv(results_path, index=False)

    print("\n已保存文件:")
    print("-", weights_path)
    print("-", nav_path)
    print("-", results_path)


def main() -> None:
    ret_df, macro_df = load_data()
    weights_df, nav_df, results_df = run_backtest(ret_df, macro_df)
    save_outputs(weights_df, nav_df, results_df)


if __name__ == "__main__":
    main()
