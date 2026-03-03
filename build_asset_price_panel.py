from pathlib import Path

import pandas as pd


RAW_DIR = Path("/Users/john/Desktop/模拟盘课题/raw_data")
PROCESSED_DIR = Path("/Users/john/Desktop/模拟盘课题/processed_data")
OUTPUT_PATH = PROCESSED_DIR / "asset_price_panel.csv"


def _read_two_row_header_excel(file_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(file_path, header=None)
    columns = raw.iloc[1].astype(str).str.strip().tolist()
    data = raw.iloc[2:].copy()
    data.columns = columns
    return data


def _to_datetime(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip()
    dt = pd.to_datetime(as_str, errors="coerce")
    if dt.notna().sum() == 0:
        dt = pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")
    return dt


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"--": None, "nan": None, "None": None, "": None})
    return pd.to_numeric(cleaned, errors="coerce")


def load_main_etf_prices(file_path: Path) -> pd.DataFrame:
    data = _read_two_row_header_excel(file_path)

    date_col = next((col for col in data.columns if "日期" in str(col)), None)
    if date_col is None:
        raise ValueError("未找到日期列")

    code_to_name = {
        "510300.SH": "hs300",
        "513500.SH": "sp500",
        "511260.SH": "cgb10y",
        "518880.SH": "gold",
        "511520.SH": "policy_bond",
    }

    selected = {"date": _to_datetime(data[date_col])}
    for code, name in code_to_name.items():
        src_col = next((col for col in data.columns if code in str(col)), None)
        if src_col is None:
            raise ValueError(f"未找到列: {code}")
        selected[name] = _to_numeric(data[src_col])

    df = pd.DataFrame(selected)
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    return df


def load_511260_511520_indices(file_path: Path) -> pd.DataFrame:
    data = _read_two_row_header_excel(file_path)

    date_col = next((col for col in data.columns if "时间" in str(col) or "日期" in str(col)), None)
    cgb_col = next((col for col in data.columns if "上证10年国债" in str(col)), None)
    policy_col = next((col for col in data.columns if "政策性金融债" in str(col)), None)

    if not all([date_col, cgb_col, policy_col]):
        raise ValueError("511260+511520指数文件列识别失败")

    df = pd.DataFrame(
        {
            "date": _to_datetime(data[date_col]),
            "cgb10y_index": _to_numeric(data[cgb_col]),
            "policy_bond_index": _to_numeric(data[policy_col]),
        }
    )
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    return df


def backfill_etf_with_index_returns(etf_price: pd.Series, index_price: pd.Series) -> pd.Series:
    merged = pd.concat([etf_price.rename("etf"), index_price.rename("idx")], axis=1).sort_index()
    idx_ret = merged["idx"].pct_change()

    first_valid = merged["etf"].first_valid_index()
    if first_valid is None:
        return merged["etf"]

    first_pos = merged.index.get_loc(first_valid)
    for pos in range(first_pos - 1, -1, -1):
        curr_date = merged.index[pos]
        next_date = merged.index[pos + 1]
        next_price = merged.at[next_date, "etf"]
        r_next = idx_ret.at[next_date]

        if pd.notna(next_price) and pd.notna(r_next) and (1 + r_next) != 0:
            merged.at[curr_date, "etf"] = next_price / (1 + r_next)

    return merged["etf"]


def load_511070_index(file_path: Path) -> pd.Series:
    data = _read_two_row_header_excel(file_path)
    date_col = next((col for col in data.columns if "时间" in str(col) or "日期" in str(col)), None)
    value_col = next((col for col in data.columns if "沪做市公司债" in str(col)), None)
    if not all([date_col, value_col]):
        raise ValueError("511070指数文件列识别失败")

    series = pd.Series(_to_numeric(data[value_col]).values, index=_to_datetime(data[date_col]), name="credit_idx")
    series = series[series.index.notna()]
    series = series[~series.index.duplicated(keep="first")].sort_index()
    series = series.where(series != 0)
    return series


def load_511070_etf(file_path: Path) -> pd.Series:
    df = pd.read_excel(file_path)
    if "日期" not in df.columns or "收盘价(元)" not in df.columns:
        raise ValueError("511070.SH 文件缺少必要列")

    series = pd.Series(_to_numeric(df["收盘价(元)"]).values, index=_to_datetime(df["日期"]), name="credit_etf")
    series = series[series.index.notna()]
    series = series[~series.index.duplicated(keep="first")].sort_index()
    return series


def build_credit_bond_series(all_dates: pd.DatetimeIndex, index_series: pd.Series, etf_series: pd.Series) -> pd.Series:
    s = pd.Series(index=all_dates, dtype="float64", name="credit_bond")

    idx_start = pd.Timestamp("2022-07-01")
    idx_end = pd.Timestamp("2025-01-21")
    etf_start = pd.Timestamp("2025-01-22")

    idx_part = index_series.reindex(all_dates)
    etf_part = etf_series.reindex(all_dates)

    mask_idx = (all_dates >= idx_start) & (all_dates <= idx_end)
    mask_etf = all_dates >= etf_start

    s.loc[mask_idx] = idx_part.loc[mask_idx]
    s.loc[mask_etf] = etf_part.loc[mask_etf]

    return s


def build_asset_price_panel() -> pd.DataFrame:
    etf_df = load_main_etf_prices(RAW_DIR / "ETF收盘价+成交金额.xlsx")
    idx_df = load_511260_511520_indices(RAW_DIR / "511260+511520指数收盘价.xlsx")

    etf_indexed = etf_df.set_index("date").sort_index()
    idx_indexed = idx_df.set_index("date").sort_index()

    cgb10y_full = backfill_etf_with_index_returns(etf_indexed["cgb10y"], idx_indexed["cgb10y_index"])
    policy_bond_full = backfill_etf_with_index_returns(
        etf_indexed["policy_bond"], idx_indexed["policy_bond_index"]
    )

    credit_idx = load_511070_index(RAW_DIR / "511070指数.xlsx")
    credit_etf = load_511070_etf(RAW_DIR / "511070.SH.xlsx")

    all_dates = pd.DatetimeIndex(
        sorted(
            set(etf_indexed.index)
            .union(idx_indexed.index)
            .union(credit_idx.index)
            .union(credit_etf.index)
        )
    )
    all_dates = all_dates[all_dates >= pd.Timestamp("2016-01-04")]

    panel = pd.DataFrame(index=all_dates)
    panel["hs300"] = etf_indexed["hs300"].reindex(all_dates)
    panel["sp500"] = etf_indexed["sp500"].reindex(all_dates)
    panel["cgb10y"] = cgb10y_full.reindex(all_dates)
    panel["gold"] = etf_indexed["gold"].reindex(all_dates)
    panel["policy_bond"] = policy_bond_full.reindex(all_dates)
    panel["credit_bond"] = build_credit_bond_series(all_dates, credit_idx, credit_etf)

    for col in ["hs300", "sp500", "cgb10y", "gold", "policy_bond", "credit_bond"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = panel.reset_index().rename(columns={"index": "date"})
    panel = panel.drop_duplicates(subset=["date"]).sort_values("date")
    panel = panel[["date", "hs300", "sp500", "cgb10y", "gold", "policy_bond", "credit_bond"]]
    return panel


def print_summary(panel: pd.DataFrame) -> None:
    print("数据起始日期:", panel["date"].min().date())
    print("数据结束日期:", panel["date"].max().date())
    print("\n每个资产的非空起始日期:")

    for col in ["hs300", "sp500", "cgb10y", "gold", "policy_bond", "credit_bond"]:
        first_valid_idx = panel[col].first_valid_index()
        first_date = panel.loc[first_valid_idx, "date"].date() if first_valid_idx is not None else None
        print(f"- {col}: {first_date}")

    print("\n各列缺失值统计:")
    print(panel.isna().sum())


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_asset_price_panel()
    panel.to_csv(OUTPUT_PATH, index=False)
    print_summary(panel)
    print(f"\n已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()



df = pd.read_csv("processed_data/asset_price_panel.csv")
print(df.dtypes)
print("\nFirst valid index per column:")
print(df.apply(lambda x: x.first_valid_index()))