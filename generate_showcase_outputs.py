from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = Path("/Users/john/Desktop/模拟盘课题")

SHOWCASE_DIR = ROOT_DIR / "showcase_output"
TABLES_DIR = SHOWCASE_DIR / "tables"
FIGURES_DIR = SHOWCASE_DIR / "figures"
SUMMARY_PATH = SHOWCASE_DIR / "summary.md"

INCLUDE_MACRO_TRANSITION = False
TARGET_VOL_DISPLAY = 0.05

DATE_COL = "date"
ASSET_ORDER = ["hs300", "sp500", "cgb10y", "gold", "policy_bond", "credit_bond", "cash"]
METRICS = ["annual_return", "annual_volatility", "sharpe", "max_drawdown", "calmar"]


def log(message: str) -> None:
	print(f"[showcase] {message}")


def ensure_exists(path: Path, description: str) -> None:
	if not path.exists():
		raise FileNotFoundError(f"{description} 不存在: {path}")


def parse_dates(series: pd.Series) -> pd.Series:
	try:
		return pd.to_datetime(series, errors="coerce", format="mixed")
	except TypeError:
		return pd.to_datetime(series, errors="coerce")


def read_csv_with_date(path: Path, description: str) -> pd.DataFrame:
	ensure_exists(path, description)
	log(f"读取: {path}")
	df = pd.read_csv(path)
	if DATE_COL not in df.columns:
		raise ValueError(f"{description} 缺少 {DATE_COL} 列: {path}")

	df[DATE_COL] = parse_dates(df[DATE_COL])
	before = len(df)
	df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL], keep="last")
	after = len(df)
	if after < before:
		log(f"{description} 日期清洗后保留 {after}/{before} 行")
	return df.reset_index(drop=True)


def classify_period(period_value: str) -> str | None:
	text = str(period_value)
	has_2018 = "2018" in text
	has_2023 = "2023" in text
	has_2024 = "2024" in text
	has_2025 = "2025" in text
	if has_2018 and has_2023:
		return "backtest"
	if has_2024 and has_2025:
		return "validation"
	return None


def extract_period_metrics(backtest_results: pd.DataFrame, strategy_name: str) -> dict[str, float]:
	if "period" not in backtest_results.columns:
		raise ValueError(f"{strategy_name} 的 backtest_results 缺少 period 列")

	period_map: dict[str, pd.Series] = {}
	for _, row in backtest_results.iterrows():
		key = classify_period(row["period"])
		if key is not None:
			period_map[key] = row

	if "backtest" not in period_map or "validation" not in period_map:
		raise ValueError(
			f"{strategy_name} 的 period 无法识别回测/验证区间。"
			"请检查 period 是否包含 2018/2023 和 2024/2025 信息。"
		)

	metrics_out: dict[str, float] = {}
	for metric in METRICS:
		for phase in ["backtest", "validation"]:
			value = period_map[phase].get(metric, np.nan)
			metrics_out[f"{metric}_{phase}"] = float(value) if pd.notna(value) else np.nan
	return metrics_out


def get_weight_columns(weights_df: pd.DataFrame) -> list[str]:
	cols = [c for c in ASSET_ORDER if c in weights_df.columns]
	if not cols:
		cols = [c for c in weights_df.columns if c != DATE_COL]
	return cols


def compute_weight_stats(weights_df: pd.DataFrame) -> dict[str, float]:
	weight_cols = get_weight_columns(weights_df)
	cash_col = "cash" if "cash" in weight_cols else None
	risk_cols = [c for c in weight_cols if c != "cash"]

	max_single_asset_weight = float(weights_df[risk_cols].max().max()) if risk_cols else np.nan
	max_single_step_weight_change = (
		float(weights_df[risk_cols].diff().abs().max().max()) if len(weights_df) >= 2 and risk_cols else np.nan
	)
	avg_cash = float(weights_df[cash_col].mean()) if cash_col is not None else np.nan

	return {
		"avg_cash": avg_cash,
		"max_single_asset_weight": max_single_asset_weight,
		"max_single_step_weight_change": max_single_step_weight_change,
	}


def compute_turnover_series(weights_df: pd.DataFrame) -> pd.Series:
	weight_cols = [c for c in get_weight_columns(weights_df) if c != DATE_COL]
	if not weight_cols:
		return pd.Series(dtype=float)
	turnover = 0.5 * weights_df[weight_cols].diff().abs().sum(axis=1)
	turnover.index = weights_df[DATE_COL]
	return turnover.dropna()


def summarize_turnover(turnover: pd.Series) -> dict[str, float]:
	if turnover.empty:
		return {"mean": np.nan, "p50": np.nan, "p90": np.nan, "max": np.nan}
	return {
		"mean": float(turnover.mean()),
		"p50": float(turnover.quantile(0.5)),
		"p90": float(turnover.quantile(0.9)),
		"max": float(turnover.max()),
	}


def normalize_nav(nav_df: pd.DataFrame) -> pd.Series:
	if "nav" in nav_df.columns:
		nav = pd.to_numeric(nav_df["nav"], errors="coerce")
	else:
		if "portfolio_return" not in nav_df.columns:
			raise ValueError("nav_series 缺少 nav 和 portfolio_return 列")
		ret = pd.to_numeric(nav_df["portfolio_return"], errors="coerce").fillna(0.0)
		nav = (1.0 + ret).cumprod()

	nav.index = nav_df[DATE_COL]
	nav = nav.dropna()
	if nav.empty:
		return nav
	return nav / float(nav.iloc[0])


def compute_drawdown(nav: pd.Series) -> pd.Series:
	if nav.empty:
		return nav
	return nav / nav.cummax() - 1.0


def compute_rolling_vol(nav_df: pd.DataFrame, window: int = 252) -> pd.Series:
	if "portfolio_return" not in nav_df.columns:
		raise ValueError("nav_series 缺少 portfolio_return 列，无法计算滚动波动率")
	ret = pd.to_numeric(nav_df["portfolio_return"], errors="coerce")
	vol = ret.rolling(window).std(ddof=0) * np.sqrt(252)
	vol.index = nav_df[DATE_COL]
	return vol


def setup_plot() -> None:
	plt.rcParams.update(
		{
			"figure.facecolor": "white",
			"axes.facecolor": "white",
			"font.size": 10,
			"axes.titlesize": 12,
			"axes.labelsize": 10,
			"legend.fontsize": 9,
		}
	)


def save_line_plot(
	series_dict: dict[str, pd.Series],
	title: str,
	ylabel: str,
	output_path: Path,
	hline: float | None = None,
) -> None:
	fig, ax = plt.subplots(figsize=(10, 5))
	for name, series in series_dict.items():
		cleaned = series.ffill().bfill() if isinstance(series, pd.Series) else series
		ax.plot(cleaned.index, cleaned.values, label=name, linewidth=1.8)

	if hline is not None:
		ax.axhline(hline, color="gray", linestyle="--", linewidth=1.2, label=f"TARGET_VOL={hline:.2f}")

	ax.set_title(title)
	ax.set_xlabel("Date")
	ax.set_ylabel(ylabel)
	ax.grid(alpha=0.25)
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=220, facecolor="white")
	plt.close(fig)
	log(f"生成图表: {output_path}")


def save_area_plot(weights_df: pd.DataFrame, strategy_name: str, output_path: Path) -> None:
	cols = [c for c in ASSET_ORDER if c in weights_df.columns and c != DATE_COL]
	if not cols:
		cols = [c for c in weights_df.columns if c != DATE_COL]

	plot_df = weights_df[[DATE_COL, *cols]].copy()
	plot_df = plot_df.set_index(DATE_COL).sort_index()
	plot_df = plot_df.fillna(0.0)

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.stackplot(plot_df.index, [plot_df[c].values for c in cols], labels=cols, alpha=0.9)
	ax.set_title(f"{strategy_name} Allocation (Monthly)")
	ax.set_xlabel("Date")
	ax.set_ylabel("Weight")
	ax.set_ylim(0, 1.05)
	ax.legend(loc="upper left", ncol=3, fontsize=8)
	ax.grid(alpha=0.2)
	fig.tight_layout()
	fig.savefig(output_path, dpi=220, facecolor="white")
	plt.close(fig)
	log(f"生成图表: {output_path}")


def format_pct(value: float) -> str:
	if pd.isna(value):
		return "NA"
	return f"{value:.2%}"


def format_num(value: float) -> str:
	if pd.isna(value):
		return "NA"
	return f"{value:.3f}"


def build_summary_markdown(perf_df: pd.DataFrame, turnover_df: pd.DataFrame) -> str:
	baseline = perf_df.loc["baseline"]
	macro = perf_df.loc["macro_stable"]

	lines = []
	lines.append("# 展示摘要（Baseline vs Macro Stable）")
	lines.append("")
	lines.append("## 1) 我们做了什么")
	lines.append("我们对比了 Baseline 风险平价策略与 Macro Stable 宏观驱动稳定版。")
	lines.append("宏观版本中，宏观信号仅用于调整风险预算，最终权重仍通过协方差结构映射得到。")
	lines.append("")
	lines.append("## 2) 关键约束与设定")
	lines.append("我们使用月度调仓框架，并以风险预算/风险贡献思想构建多资产组合。")
	lines.append("我们设置 TARGET_VOL=0.05 作为展示参照线，用于观察滚动波动率区间。")
	lines.append("对于单资产权重上限与单次变动限制，如需判断是否模型内硬约束，应以回测引擎代码为准；本文仅如实报告结果表现。")
	lines.append("")
	lines.append("## 3) 结果对比（自动填充）")
	lines.append("### 回测段（2018-2023）")
	lines.append(
		f"- Baseline: return={format_pct(baseline['annual_return_backtest'])}, vol={format_pct(baseline['annual_volatility_backtest'])}, "
		f"sharpe={format_num(baseline['sharpe_backtest'])}, maxDD={format_pct(baseline['max_drawdown_backtest'])}, calmar={format_num(baseline['calmar_backtest'])}"
	)
	lines.append(
		f"- Macro Stable: return={format_pct(macro['annual_return_backtest'])}, vol={format_pct(macro['annual_volatility_backtest'])}, "
		f"sharpe={format_num(macro['sharpe_backtest'])}, maxDD={format_pct(macro['max_drawdown_backtest'])}, calmar={format_num(macro['calmar_backtest'])}"
	)
	lines.append("### 验证段（2024-2025）")
	lines.append(
		f"- Baseline: return={format_pct(baseline['annual_return_validation'])}, vol={format_pct(baseline['annual_volatility_validation'])}, "
		f"sharpe={format_num(baseline['sharpe_validation'])}, maxDD={format_pct(baseline['max_drawdown_validation'])}, calmar={format_num(baseline['calmar_validation'])}"
	)
	lines.append(
		f"- Macro Stable: return={format_pct(macro['annual_return_validation'])}, vol={format_pct(macro['annual_volatility_validation'])}, "
		f"sharpe={format_num(macro['sharpe_validation'])}, maxDD={format_pct(macro['max_drawdown_validation'])}, calmar={format_num(macro['calmar_validation'])}"
	)
	lines.append(
		f"- 现金占比（avg_cash）: Baseline={format_pct(baseline['avg_cash'])}, Macro Stable={format_pct(macro['avg_cash'])}"
	)
	lines.append(
		f"- 最大单资产权重: Baseline={format_pct(baseline['max_single_asset_weight'])}, "
		f"Macro Stable={format_pct(macro['max_single_asset_weight'])}"
	)
	lines.append(
		f"- 最大单次权重变化: Baseline={format_pct(baseline['max_single_step_weight_change'])}, "
		f"Macro Stable={format_pct(macro['max_single_step_weight_change'])}"
	)
	if "baseline" in turnover_df.index and "macro_stable" in turnover_df.index:
		lines.append(
			f"- 月度换手率均值: Baseline={format_pct(turnover_df.loc['baseline', 'mean'])}, "
			f"Macro Stable={format_pct(turnover_df.loc['macro_stable', 'mean'])}"
		)
	lines.append("")
	lines.append("## 4) 对宏观版波动/集中变化的谨慎解释")
	lines.append("宏观信号通过风险预算再分配影响组合风险结构，而不是直接指定目标权重。")
	lines.append("当宏观状态持续偏向某类资产时，组合可能阶段性提高权益或信用暴露，从而带来波动率、回撤和换手率特征变化。")
	lines.append("因此展示时建议联动观察收益、波动、回撤、现金占比和换手率，避免单指标解读。")

	return "\n".join(lines) + "\n"


def main() -> None:
	setup_plot()

	baseline_paths = {
		"weights": ROOT_DIR / "回测引擎" / "weights.csv",
		"nav": ROOT_DIR / "回测引擎" / "nav_series.csv",
		"results": ROOT_DIR / "回测引擎" / "backtest_results.csv",
	}
	macro_stable_paths = {
		"weights": ROOT_DIR / "回测引擎_macro_stable" / "weights_macro_stable.csv",
		"nav": ROOT_DIR / "回测引擎_macro_stable" / "nav_series_macro_stable.csv",
		"results": ROOT_DIR / "回测引擎_macro_stable" / "backtest_results_macro_stable.csv",
	}

	if INCLUDE_MACRO_TRANSITION:
		ensure_exists(ROOT_DIR / "回测引擎_macro" / "weights_macro.csv", "macro过渡版weights")
		ensure_exists(ROOT_DIR / "回测引擎_macro" / "nav_series_macro.csv", "macro过渡版nav")
		ensure_exists(ROOT_DIR / "回测引擎_macro" / "backtest_results_macro.csv", "macro过渡版results")

	SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
	TABLES_DIR.mkdir(parents=True, exist_ok=True)
	FIGURES_DIR.mkdir(parents=True, exist_ok=True)
	log(f"输出目录: {SHOWCASE_DIR}")

	baseline_weights = read_csv_with_date(baseline_paths["weights"], "baseline weights")
	baseline_nav = read_csv_with_date(baseline_paths["nav"], "baseline nav")
	baseline_results = pd.read_csv(baseline_paths["results"])

	macro_weights = read_csv_with_date(macro_stable_paths["weights"], "macro_stable weights")
	macro_nav = read_csv_with_date(macro_stable_paths["nav"], "macro_stable nav")
	macro_results = pd.read_csv(macro_stable_paths["results"])

	perf_rows = {}
	for strategy, weights_df, results_df in [
		("baseline", baseline_weights, baseline_results),
		("macro_stable", macro_weights, macro_results),
	]:
		period_metrics = extract_period_metrics(results_df, strategy)
		weight_stats = compute_weight_stats(weights_df)
		perf_rows[strategy] = {**period_metrics, **weight_stats}

	perf_df = pd.DataFrame.from_dict(perf_rows, orient="index")
	perf_df.index.name = "strategy"
	perf_path = TABLES_DIR / "perf_comparison.csv"
	perf_df.to_csv(perf_path)
	log(f"生成表格: {perf_path}")

	baseline_turnover = compute_turnover_series(baseline_weights)
	macro_turnover = compute_turnover_series(macro_weights)

	turnover_df = pd.DataFrame.from_dict(
		{
			"baseline": summarize_turnover(baseline_turnover),
			"macro_stable": summarize_turnover(macro_turnover),
		},
		orient="index",
	)
	turnover_df.index.name = "strategy"
	turnover_path = TABLES_DIR / "turnover_summary.csv"
	turnover_df.to_csv(turnover_path)
	log(f"生成表格: {turnover_path}")

	baseline_nav_norm = normalize_nav(baseline_nav)
	macro_nav_norm = normalize_nav(macro_nav)

	save_line_plot(
		{"baseline": baseline_nav_norm, "macro_stable": macro_nav_norm},
		"NAV Comparison",
		"Normalized NAV",
		FIGURES_DIR / "nav_comparison.png",
	)
	save_line_plot(
		{
			"baseline": compute_drawdown(baseline_nav_norm),
			"macro_stable": compute_drawdown(macro_nav_norm),
		},
		"Drawdown Comparison",
		"Drawdown",
		FIGURES_DIR / "drawdown_comparison.png",
	)
	save_line_plot(
		{
			"baseline": compute_rolling_vol(baseline_nav),
			"macro_stable": compute_rolling_vol(macro_nav),
		},
		"Rolling 252D Annualized Volatility",
		"Annualized Vol",
		FIGURES_DIR / "rolling_vol_252.png",
		hline=TARGET_VOL_DISPLAY,
	)

	save_area_plot(baseline_weights, "baseline", FIGURES_DIR / "allocation_area_baseline.png")
	save_area_plot(macro_weights, "macro_stable", FIGURES_DIR / "allocation_area_macro_stable.png")

	save_line_plot(
		{"baseline": baseline_turnover, "macro_stable": macro_turnover},
		"Monthly Turnover Comparison",
		"Turnover",
		FIGURES_DIR / "turnover_timeseries.png",
	)

	baseline_cash = baseline_weights.set_index(DATE_COL)["cash"] if "cash" in baseline_weights.columns else pd.Series(dtype=float)
	macro_cash = macro_weights.set_index(DATE_COL)["cash"] if "cash" in macro_weights.columns else pd.Series(dtype=float)
	save_line_plot(
		{"baseline": baseline_cash, "macro_stable": macro_cash},
		"Cash Weight Comparison",
		"Cash Weight",
		FIGURES_DIR / "cash_weight_comparison.png",
	)

	summary_text = build_summary_markdown(perf_df, turnover_df)
	SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
	log(f"生成文档: {SUMMARY_PATH}")

	generated_files = [
		perf_path,
		turnover_path,
		FIGURES_DIR / "nav_comparison.png",
		FIGURES_DIR / "drawdown_comparison.png",
		FIGURES_DIR / "rolling_vol_252.png",
		FIGURES_DIR / "allocation_area_baseline.png",
		FIGURES_DIR / "allocation_area_macro_stable.png",
		FIGURES_DIR / "turnover_timeseries.png",
		FIGURES_DIR / "cash_weight_comparison.png",
		SUMMARY_PATH,
	]
	log("生成完成，文件列表:")
	for file_path in generated_files:
		log(f"- {file_path}")


if __name__ == "__main__":
	main()
