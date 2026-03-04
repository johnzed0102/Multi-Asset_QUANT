from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("/Users/john/Desktop/Multi-Asset_QUANT/processed_data")
PRICE_PATH = PROCESSED_DIR / "asset_price_panel.csv"
RETURN_PATH = PROCESSED_DIR / "asset_return_panel.csv"
ASSET_COLUMNS = ["hs300", "sp500", "cgb10y", "gold", "policy_bond", "credit_bond"]


def load_price_panel(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "date" not in df.columns:
        raise ValueError("价格面板缺少 date 列")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="first")

    missing_columns = [col for col in ASSET_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"价格面板缺少资产列: {missing_columns}")

    for col in ASSET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["date", *ASSET_COLUMNS]].reset_index(drop=True)


def compute_returns(price_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    ret_df = pd.DataFrame({"date": price_df["date"]})
    nonpositive_nan_counts: dict[str, int] = {}

    for col in ASSET_COLUMNS:
        price = price_df[col]
        prev_price = price.shift(1)

        invalid_current = price <= 0
        invalid_previous = prev_price <= 0
        invalid_for_log = invalid_current | invalid_previous

        nonpositive_nan_counts[col] = int(invalid_for_log.fillna(False).sum())

        log_ret = np.log(price / prev_price)
        log_ret = log_ret.where(~invalid_for_log, np.nan)
        ret_df[col] = log_ret.astype("float64")

    return ret_df, nonpositive_nan_counts


def validate_return_panel(ret_df: pd.DataFrame, nonpositive_nan_counts: dict[str, int]) -> pd.DataFrame:
    print("A. 数据起始/结束日期")
    print("- 起始日期:", ret_df["date"].min().date())
    print("- 结束日期:", ret_df["date"].max().date())

    print("\nB. 各列缺失值统计（NaN count）")
    print(ret_df.isna().sum())

    print("\nC. 各列首次非空收益率日期（first valid index）")
    for col in ASSET_COLUMNS:
        first_valid_idx = ret_df[col].first_valid_index()
        first_valid_date = ret_df.loc[first_valid_idx, "date"].date() if first_valid_idx is not None else None
        print(f"- {col}: {first_valid_date}")

    print("\nD. inf / -inf 检查与替换")
    inf_counts = {}
    total_inf = 0
    for col in ASSET_COLUMNS:
        mask_inf = np.isinf(ret_df[col].to_numpy(dtype="float64", na_value=np.nan))
        count_inf = int(mask_inf.sum())
        inf_counts[col] = count_inf
        total_inf += count_inf
        if count_inf > 0:
            ret_df.loc[mask_inf, col] = np.nan
    print("- 各列 inf/-inf 数量:", inf_counts)
    print("- inf/-inf 总数量:", total_inf)

    print("\nE. 价格<=0导致的 NaN 数量")
    print(nonpositive_nan_counts)

    return ret_df


def save_return_panel(ret_df: pd.DataFrame, file_path: Path) -> None:
    output_df = ret_df[["date", *ASSET_COLUMNS]].copy()
    for col in ASSET_COLUMNS:
        output_df[col] = output_df[col].astype("float64")
    output_df.to_csv(file_path, index=False)
    print(f"\n已保存: {file_path}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    price_df = load_price_panel(PRICE_PATH)
    ret_df, nonpositive_nan_counts = compute_returns(price_df)
    ret_df = validate_return_panel(ret_df, nonpositive_nan_counts)
    save_return_panel(ret_df, RETURN_PATH)


if __name__ == "__main__":
    main()