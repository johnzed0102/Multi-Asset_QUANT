from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path("/Users/john/Desktop/模拟盘课题")
RETURNS_PATH = ROOT_DIR / "processed_data/asset_return_panel.csv"
MACRO_PATH = ROOT_DIR / "processed_data/macro_panel.csv"
OUTPUT_DIR = ROOT_DIR / "回测引擎_macro"

ASSETS = ["hs300", "sp500", "cgb10y", "gold", "policy_bond", "credit_bond"]
BOND_ASSETS = ["cgb10y", "policy_bond", "credit_bond"]
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取并标准化资产收益与宏观面板。"""
    ret_df = pd.read_csv(RETURNS_PATH)
    macro_df = pd.read_csv(MACRO_PATH)

    required_ret_cols = ["date", *ASSETS]
    required_macro_cols = ["date", "PMI", "CPI", "CN10Y"]

    for col in required_ret_cols:
        if col not in ret_df.columns:
            raise ValueError(f"资产收益率缺少列: {col}")
    for col in required_macro_cols:
        if col not in macro_df.columns:
            raise ValueError(f"宏观数据缺少列: {col}")

    ret_df["date"] = pd.to_datetime(ret_df["date"], errors="coerce")
    ret_df = ret_df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    ret_df = ret_df.reset_index(drop=True)

    for col in ASSETS:
        ret_df[col] = pd.to_numeric(ret_df[col], errors="coerce")

    macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce")
    macro_df = macro_df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    macro_df = macro_df.reset_index(drop=True)

    for col in ["PMI", "CPI", "CN10Y"]:
        macro_df[col] = pd.to_numeric(macro_df[col], errors="coerce")

    return ret_df, macro_df


def identify_rebalance_dates(ret_df: pd.DataFrame) -> pd.DatetimeIndex:
    """识别每月最后一个交易日作为调仓日。"""
    month_last = ret_df.groupby(ret_df["date"].dt.to_period("M"))["date"].max().sort_values()
    return pd.DatetimeIndex(month_last.values)


def compute_covariance(window_returns: pd.DataFrame, eligible_assets: list[str]) -> pd.DataFrame:
    """计算协方差矩阵，并确保为可用的正定近似矩阵。"""
    cov = window_returns[eligible_assets].cov()
    cov = cov.reindex(index=eligible_assets, columns=eligible_assets)

    if cov.isna().any().any():
        return pd.DataFrame()

    cov_values = cov.to_numpy(dtype=float)
    cov_values = (cov_values + cov_values.T) / 2.0
    cov_values = cov_values + np.eye(len(eligible_assets)) * 1e-10
    return pd.DataFrame(cov_values, index=eligible_assets, columns=eligible_assets)


def solve_risk_budget_weights(cov: pd.DataFrame, budgets: np.ndarray, max_iter: int = 2000, tol: float = 1e-8) -> np.ndarray:
    """
    使用自定义迭代法求解风险预算权重（非黑箱优化库）。
    约束：w >= 0, sum(w) = 1。
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0], dtype=float)

    sigma = cov.to_numpy(dtype=float)
    b = np.maximum(budgets, 0.0)
    b = b / b.sum()

    w = b.copy()
    w = np.maximum(w, 1e-12)
    w = w / w.sum()

    for _ in range(max_iter):
        sigma_w = sigma @ w
        rc = w * sigma_w
        rc_sum = rc.sum()
        if rc_sum <= 0:
            break

        rc_ratio = rc / rc_sum
        error = np.max(np.abs(rc_ratio - b))
        if error < tol:
            break

        target_rc = b * rc_sum
        update_ratio = target_rc / np.maximum(rc, 1e-12)
        w = w * np.power(update_ratio, 0.5)
        w = np.maximum(w, 1e-12)
        w = w / w.sum()

    return w


def get_macro_state(reb_date: pd.Timestamp, macro_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
    """使用当月宏观值与历史中位数（仅过去）构造三维宏观信号。"""
    macro_period = reb_date.to_period("M")
    macro_work = macro_df.copy()
    macro_work["period"] = macro_work["date"].dt.to_period("M")

    current_rows = macro_work[macro_work["period"] == macro_period]
    if current_rows.empty:
        current_rows = macro_work[macro_work["period"] < macro_period].tail(1)
        if current_rows.empty:
            return {"growth": 0, "inflation": 0, "rate": 0}, {}

    current = current_rows.iloc[-1]
    hist = macro_work[macro_work["period"] < macro_period]
    if hist.empty:
        return {"growth": 0, "inflation": 0, "rate": 0}, {
            "PMI": float(current["PMI"]),
            "CPI": float(current["CPI"]),
            "CN10Y": float(current["CN10Y"]),
        }

    med_pmi = hist["PMI"].median()
    med_cpi = hist["CPI"].median()
    med_cn10y = hist["CN10Y"].median()

    signals = {
        "growth": int(pd.notna(current["PMI"]) and pd.notna(med_pmi) and current["PMI"] > med_pmi),
        "inflation": int(pd.notna(current["CPI"]) and pd.notna(med_cpi) and current["CPI"] > med_cpi),
        "rate": int(pd.notna(current["CN10Y"]) and pd.notna(med_cn10y) and current["CN10Y"] > med_cn10y),
    }
    values = {
        "PMI": float(current["PMI"]) if pd.notna(current["PMI"]) else np.nan,
        "CPI": float(current["CPI"]) if pd.notna(current["CPI"]) else np.nan,
        "CN10Y": float(current["CN10Y"]) if pd.notna(current["CN10Y"]) else np.nan,
    }
    return signals, values


def adjust_risk_budget(signals: dict[str, int], eligible_assets: list[str]) -> tuple[pd.Series, float]:
    """
    根据宏观信号调整风险预算。
    返回：资产预算（归一化后）与额外现金倾向（rate信号且无credit时）。
    """
    budget = pd.Series(BASE_BUDGET, dtype="float64")
    cash_tilt = 0.0

    bond_set = [asset for asset in BOND_ASSETS if asset in budget.index]

    if signals.get("growth", 0) == 1:
        budget["hs300"] += 0.05
        budget["sp500"] += 0.05
        deduct_assets = [a for a in bond_set if budget[a] > 0]
        if deduct_assets:
            deduct_each = 0.10 / len(deduct_assets)
            for a in deduct_assets:
                budget[a] = max(0.0, budget[a] - deduct_each)

    if signals.get("inflation", 0) == 1:
        budget["gold"] += 0.05
        deduct_assets = [a for a in bond_set if budget[a] > 0]
        if deduct_assets:
            deduct_each = 0.05 / len(deduct_assets)
            for a in deduct_assets:
                budget[a] = max(0.0, budget[a] - deduct_each)

    if signals.get("rate", 0) == 1:
        delta = 0.0
        for asset in ["cgb10y", "policy_bond"]:
            cut = min(0.05, budget[asset])
            budget[asset] -= cut
            delta += cut

        if "credit_bond" in eligible_assets:
            budget["credit_bond"] += delta
        else:
            cash_tilt += delta

    budget = budget.clip(lower=0.0)
    budget = budget[eligible_assets]

    budget_sum = budget.sum()
    if budget_sum <= 0:
        budget[:] = 1.0 / len(budget)
    else:
        budget = budget / budget_sum

    return budget, cash_tilt


def apply_vol_target(weights: np.ndarray, cov: pd.DataFrame, target_vol: float, cash_tilt: float) -> tuple[np.ndarray, float, float, float]:
    """执行目标波动率缩放，并加入现金权重。"""
    sigma = cov.to_numpy(dtype=float)
    port_var = float(weights.T @ sigma @ weights)
    annual_vol = float(np.sqrt(max(0.0, 252.0 * port_var)))

    scale = 1.0
    if annual_vol > target_vol and annual_vol > 0:
        scale = target_vol / annual_vol
    scale = min(1.0, scale)

    scaled = weights * scale
    if cash_tilt > 0:
        scaled = scaled * max(0.0, 1.0 - cash_tilt)

    cash_weight = float(max(0.0, 1.0 - scaled.sum()))
    return scaled, cash_weight, annual_vol, scale


def compute_performance(returns: pd.Series) -> dict[str, float]:
    """计算年化收益、波动、夏普、最大回撤、Calmar。"""
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
    """执行宏观驱动风险预算回测。"""
    rebalance_dates = identify_rebalance_dates(ret_df)
    ret_idx = ret_df.set_index("date")

    rebalance_map: dict[pd.Timestamp, pd.Series] = {}
    weight_records = []
    state_records = []
    first_rebalance = rebalance_dates[0] if len(rebalance_dates) > 0 else None

    for reb_date in rebalance_dates:
        row = {"date": reb_date, **{a: np.nan for a in ASSETS}, "cash": np.nan}
        hist = ret_idx.loc[:reb_date, ASSETS]

        if len(hist) < WINDOW:
            weight_records.append(row)
            continue

        window_returns = hist.tail(WINDOW)
        eligible_assets = [a for a in ASSETS if window_returns[a].notna().any() and pd.notna(window_returns.iloc[-1][a])]

        if len(eligible_assets) == 0:
            weight_records.append(row)
            continue

        cov = compute_covariance(window_returns, eligible_assets)
        if cov.empty:
            weight_records.append(row)
            continue

        signals, macro_values = get_macro_state(reb_date, macro_df)
        budget, cash_tilt = adjust_risk_budget(signals, eligible_assets)

        weights_budget = solve_risk_budget_weights(cov, budget.to_numpy(dtype=float))
        weights_scaled, cash_weight, annual_vol, scale = apply_vol_target(
            weights_budget, cov, target_vol=TARGET_VOL, cash_tilt=cash_tilt
        )

        final_w = pd.Series(0.0, index=ASSETS + ["cash"], dtype="float64")
        final_w.loc[eligible_assets] = weights_scaled
        final_w.loc["cash"] = cash_weight

        rebalance_map[reb_date] = final_w

        for c in ASSETS + ["cash"]:
            row[c] = float(final_w[c])
        row["macro_growth"] = signals.get("growth", 0)
        row["macro_inflation"] = signals.get("inflation", 0)
        row["macro_rate"] = signals.get("rate", 0)
        row["realized_annual_vol"] = annual_vol
        row["vol_scale"] = scale
        weight_records.append(row)

        state_records.append(
            {
                "date": reb_date,
                "growth": signals.get("growth", 0),
                "inflation": signals.get("inflation", 0),
                "rate": signals.get("rate", 0),
                "PMI": macro_values.get("PMI", np.nan),
                "CPI": macro_values.get("CPI", np.nan),
                "CN10Y": macro_values.get("CN10Y", np.nan),
            }
        )

    weights_full = pd.DataFrame(weight_records).sort_values("date").reset_index(drop=True)

    daily_weights = pd.DataFrame(index=ret_idx.index, columns=ASSETS + ["cash"], dtype="float64")
    current = pd.Series(np.nan, index=ASSETS + ["cash"], dtype="float64")
    for dt in ret_idx.index:
        if dt in rebalance_map:
            current = rebalance_map[dt]
        daily_weights.loc[dt] = current.values

    lagged_weights = daily_weights[ASSETS].shift(1)
    port_ret = []
    for dt in ret_idx.index:
        w = lagged_weights.loc[dt]
        r = ret_idx.loc[dt, ASSETS]

        if w.isna().all():
            port_ret.append(np.nan)
            continue

        has_missing_for_held = ((w.fillna(0.0) > 0) & r.isna()).any()
        if has_missing_for_held:
            port_ret.append(np.nan)
            continue

        port_ret.append(float((w.fillna(0.0) * r.fillna(0.0)).sum()))

    nav_df = pd.DataFrame({"date": ret_idx.index, "portfolio_return": port_ret})
    nav_df["nav"] = (1.0 + nav_df["portfolio_return"].fillna(0.0)).cumprod()

    mask_bt = (nav_df["date"] >= pd.Timestamp("2018-01-01")) & (nav_df["date"] <= pd.Timestamp("2023-12-31"))
    mask_val = (nav_df["date"] >= pd.Timestamp("2024-01-01")) & (nav_df["date"] <= pd.Timestamp("2025-12-01"))

    perf_bt = compute_performance(nav_df.loc[mask_bt, "portfolio_return"])
    perf_val = compute_performance(nav_df.loc[mask_val, "portfolio_return"])

    results = pd.DataFrame([
        {"period": "2018-01-01_2023-12-31", **perf_bt},
        {"period": "2024-01-01_2025-12-01", **perf_val},
    ])

    state_df = pd.DataFrame(state_records).sort_values("date") if state_records else pd.DataFrame()
    if not state_df.empty:
        state_tuple = state_df[["growth", "inflation", "rate"]].astype(int).astype(str).agg("_".join, axis=1)
        switch_count = int((state_tuple != state_tuple.shift(1)).sum() - 1)
        switch_count = max(0, switch_count)
        growth_strong_count = int((state_df["growth"] == 1).sum())
        growth_weak_count = int((state_df["growth"] == 0).sum())
    else:
        switch_count = 0
        growth_strong_count = 0
        growth_weak_count = 0

    valid_weights = weights_full.dropna(subset=ASSETS + ["cash"], how="all")
    max_weight_change = (
        float(valid_weights[ASSETS + ["cash"]].diff().abs().max().max())
        if len(valid_weights) >= 2
        else np.nan
    )
    avg_cash = float(valid_weights["cash"].mean()) if not valid_weights.empty else np.nan

    vol_flags = results[["period", "annual_volatility"]].copy()
    vol_flags["vol_leq_target"] = vol_flags["annual_volatility"] <= TARGET_VOL

    print("1）第一个调仓日:", first_rebalance.date() if first_rebalance is not None else None)
    print("2）宏观状态切换次数:", switch_count)
    print("3）最大单次权重变化:", max_weight_change)
    print("4）平均现金比例:", avg_cash)
    print("5）两阶段绩效表:")
    print(results)
    print("6）两阶段年化波动率是否 <= 0.05:")
    print(vol_flags)
    print("增长状态频率: strong=", growth_strong_count, ", weak=", growth_weak_count, sep="")

    weights_out = weights_full[["date", *ASSETS, "cash"]].copy()
    return weights_out, nav_df, results


def save_outputs(weights_df: pd.DataFrame, nav_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    """保存宏观版回测输出到独立目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weights_path = OUTPUT_DIR / "weights_macro.csv"
    nav_path = OUTPUT_DIR / "nav_series_macro.csv"
    results_path = OUTPUT_DIR / "backtest_results_macro.csv"

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