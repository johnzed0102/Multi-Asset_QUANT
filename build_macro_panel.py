from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


RAW_PATH = Path("/Users/john/Desktop/模拟盘课题/raw_data/宏观数据.xlsx")
OUTPUT_PATH = Path("/Users/john/Desktop/模拟盘课题/processed_data/macro_panel.csv")


def load_macro_data(file_path: Path) -> pd.DataFrame:
    return pd.read_excel(file_path)


def _find_column(columns: list[str], keywords: list[str]) -> str:
    for col in columns:
        col_str = str(col)
        if all(k in col_str for k in keywords):
            return col
    raise ValueError(f"未找到满足关键词 {keywords} 的列")


def _to_float(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "--": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def clean_macro_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c) for c in raw_df.columns]

    date_col = _find_column(cols, ["指标名称"])
    pmi_col = _find_column(cols, ["制造业PMI"])
    cpi_col = _find_column(cols, ["CPI", "当月同比"])
    cn10y_col = _find_column(cols, ["国债", "到期收益率", "10年"])

    selected = raw_df[[date_col, pmi_col, cpi_col, cn10y_col]].copy()
    selected.columns = ["date", "PMI", "CPI", "CN10Y"]

    date_raw = selected["date"].astype(str).str.strip()
    parsed = pd.to_datetime(date_raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    mask_ymd = parsed.isna() & date_raw.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    if mask_ymd.any():
        parsed.loc[mask_ymd] = pd.to_datetime(date_raw[mask_ymd], format="%Y-%m-%d", errors="coerce")

    mask_ym = parsed.isna() & date_raw.str.match(r"^\d{4}-\d{2}$", na=False)
    if mask_ym.any():
        parsed.loc[mask_ym] = pd.to_datetime(date_raw[mask_ym], format="%Y-%m", errors="coerce")

    mask_ym_slash = parsed.isna() & date_raw.str.match(r"^\d{4}/\d{2}$", na=False)
    if mask_ym_slash.any():
        parsed.loc[mask_ym_slash] = pd.to_datetime(date_raw[mask_ym_slash], format="%Y/%m", errors="coerce")

    selected["date"] = parsed
    selected = selected.dropna(subset=["date"])
    selected["date"] = selected["date"] + MonthEnd(0)

    for col in ["PMI", "CPI", "CN10Y"]:
        selected[col] = _to_float(selected[col])

    selected = selected[(selected["date"] >= pd.Timestamp("2016-01-01")) & (selected["date"] <= pd.Timestamp("2026-02-28"))]
    return selected


def validate_macro_data(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    duplicated_mask = df["date"].duplicated(keep=False)
    duplicated_months = sorted(df.loc[duplicated_mask, "date"].dt.strftime("%Y-%m").unique().tolist())

    print("C. 是否存在重复月份")
    if duplicated_months:
        print("- 存在重复月份:", duplicated_months)
    else:
        print("- 不存在重复月份")

    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    print("A. 宏观数据起始/结束月份")
    if not df.empty:
        print("- 起始月份:", df["date"].min().strftime("%Y-%m"))
        print("- 结束月份:", df["date"].max().strftime("%Y-%m"))
    else:
        print("- 数据为空")

    print("\nB. 各列缺失值统计")
    print(df.isna().sum())

    print("\nD. 关键月份检查")
    month_targets = ["2016-01", "2018-01", "2024-01", "2025-12"]
    df_period = df.set_index(df["date"].dt.to_period("M"))
    for m in month_targets:
        period = pd.Period(m, freq="M")
        if period in df_period.index:
            row = df_period.loc[period]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            print(
                f"- {m}: PMI={row['PMI']}, CPI={row['CPI']}, CN10Y={row['CN10Y']}"
            )

    return df[["date", "PMI", "CPI", "CN10Y"]]


def save_macro_data(df: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"\n已保存: {file_path}")


def main() -> None:
    raw_df = load_macro_data(RAW_PATH)
    clean_df = clean_macro_data(raw_df)
    final_df = validate_macro_data(clean_df)
    save_macro_data(final_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()