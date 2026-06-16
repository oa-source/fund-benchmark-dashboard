import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# FOLDER SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_CSV_PATH = DATA_DIR / "extracted_funds.csv"
DATABASE_PATH = DATA_DIR / "funds_database.sqlite"

ENRICHED_FUNDS_CSV_PATH = DATA_DIR / "funds_with_vintage_quartiles.csv"
BENCHMARKS_CSV_PATH = DATA_DIR / "vintage_benchmarks.csv"


# ============================================================
# SETTINGS
# ============================================================

METRICS = ["irr", "tvpi", "dpi"]

BENCHMARK_COLUMNS = [
    "strategy",
    "geography",
    "vintage_year",
    "metric",
    "fund_count",
    "average",
    "median",
    "q1_25th_percentile",
    "q3_75th_percentile",
    "min_value",
    "max_value",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def to_number(value):
    """
    Converts values like:
    15.2%
    15.2
    1.45x
    $1,000
    into numeric values.

    For now, this is mainly for IRR / TVPI / DPI.
    """
    if value is None:
        return np.nan

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    value = value.replace("%", "")
    value = value.replace("x", "")
    value = value.replace("X", "")
    value = value.replace("$", "")
    value = value.replace("€", "")
    value = value.replace("£", "")
    value = value.replace(",", "")
    value = value.replace("*", "")
    value = value.replace("(", "")
    value = value.replace(")", "")
    value = value.strip()

    if value.lower() in ["na", "n/a", "none", "null", "tbd", "-"]:
        return np.nan

    try:
        return float(value)
    except Exception:
        return np.nan


def percentile_rank_within_group(values):
    """
    Higher value = better.
    Returns percentile rank from 0 to 100.
    """
    numeric_values = pd.to_numeric(values, errors="coerce")

    if numeric_values.notna().sum() <= 1:
        return pd.Series([np.nan] * len(values), index=values.index)

    return numeric_values.rank(pct=True, ascending=True) * 100


def percentile_to_quartile(percentile):
    """
    Q1 = top 25%
    Q2 = 25%-50%
    Q3 = 50%-75%
    Q4 = bottom 25%
    """
    if pd.isna(percentile):
        return ""

    if percentile >= 75:
        return "Q1"

    if percentile >= 50:
        return "Q2"

    if percentile >= 25:
        return "Q3"

    return "Q4"


def quartile_to_score(quartile):
    if quartile == "Q1":
        return 4
    if quartile == "Q2":
        return 3
    if quartile == "Q3":
        return 2
    if quartile == "Q4":
        return 1
    return np.nan


def score_to_overall_quartile(score):
    if pd.isna(score):
        return ""

    if score >= 3.5:
        return "Q1"

    if score >= 2.5:
        return "Q2"

    if score >= 1.5:
        return "Q3"

    return "Q4"


def vintage_result_label(overall_quartile):
    if overall_quartile == "Q1":
        return "Top quartile vs vintage"

    if overall_quartile == "Q2":
        return "Above median vs vintage"

    if overall_quartile == "Q3":
        return "Below median vs vintage"

    if overall_quartile == "Q4":
        return "Weak vs vintage"

    return "Insufficient performance data"


# ============================================================
# LOAD DATA
# ============================================================

def load_funds():
    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH)

    required_cols = [
        "fund_name",
        "manager_name",
        "vintage_year",
        "strategy",
        "geography",
        "irr",
        "tvpi",
        "dpi",
        "nav",
        "commitment",
        "paid_in",
        "distributions",
        "fund_size",
        "source_name",
        "source_file",
        "source_page",
        "raw_text",
        "data_quality_flag",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df["vintage_year"] = pd.to_numeric(df["vintage_year"], errors="coerce")

    for col in ["strategy", "geography", "fund_name", "manager_name", "data_quality_flag"]:
        df[col] = df[col].apply(clean_text)

    for metric in METRICS:
        df[f"{metric}_numeric"] = df[metric].apply(to_number)

    return df


# ============================================================
# BENCHMARK CALCULATION
# ============================================================

def build_benchmark_table(df):
    benchmark_rows = []

    for strategy in sorted(df["strategy"].dropna().unique()):
        if clean_text(strategy) == "":
            continue

        strategy_df = df[df["strategy"] == strategy].copy()

        for geography in sorted(strategy_df["geography"].dropna().unique()):
            if clean_text(geography) == "":
                continue

            peer_df = strategy_df[strategy_df["geography"] == geography].copy()

            for vintage_year in sorted(peer_df["vintage_year"].dropna().unique()):
                vintage_df = peer_df[peer_df["vintage_year"] == vintage_year].copy()

                for metric in METRICS:
                    metric_col = f"{metric}_numeric"
                    metric_values = pd.to_numeric(vintage_df[metric_col], errors="coerce").dropna()

                    if len(metric_values) == 0:
                        continue

                    benchmark_rows.append({
                        "strategy": strategy,
                        "geography": geography,
                        "vintage_year": int(vintage_year),
                        "metric": metric,
                        "fund_count": int(len(metric_values)),
                        "average": float(metric_values.mean()),
                        "median": float(metric_values.median()),
                        "q1_25th_percentile": float(metric_values.quantile(0.25)),
                        "q3_75th_percentile": float(metric_values.quantile(0.75)),
                        "min_value": float(metric_values.min()),
                        "max_value": float(metric_values.max()),
                    })

    benchmarks_df = pd.DataFrame(benchmark_rows)

    if benchmarks_df.empty:
        benchmarks_df = pd.DataFrame(columns=BENCHMARK_COLUMNS)

    return benchmarks_df


def add_fund_percentiles_and_quartiles(df):
    df = df.copy()

    for metric in METRICS:
        metric_col = f"{metric}_numeric"
        percentile_col = f"{metric}_vintage_percentile"
        quartile_col = f"{metric}_vintage_quartile"

        df[percentile_col] = np.nan
        df[quartile_col] = ""

        valid_mask = (
            df["vintage_year"].notna()
            & df["strategy"].notna()
            & df["geography"].notna()
            & df[metric_col].notna()
        )

        if valid_mask.sum() == 0:
            continue

        df.loc[valid_mask, percentile_col] = (
            df.loc[valid_mask]
            .groupby(["strategy", "geography", "vintage_year"])[metric_col]
            .transform(percentile_rank_within_group)
        )

        df[quartile_col] = df[percentile_col].apply(percentile_to_quartile)

    quartile_score_cols = []

    for metric in METRICS:
        quartile_col = f"{metric}_vintage_quartile"
        score_col = f"{metric}_quartile_score"
        df[score_col] = df[quartile_col].apply(quartile_to_score)
        quartile_score_cols.append(score_col)

    df["overall_quartile_score"] = df[quartile_score_cols].mean(axis=1, skipna=True)
    df["overall_vintage_quartile"] = df["overall_quartile_score"].apply(score_to_overall_quartile)
    df["vintage_benchmark_label"] = df["overall_vintage_quartile"].apply(vintage_result_label)

    df["internal_score"] = df["overall_quartile_score"] * 25
    df["internal_score"] = df["internal_score"].round(1)

    df["allocator_memo_short"] = df.apply(build_allocator_memo_short, axis=1)

    return df


def build_allocator_memo_short(row):
    fund_name = clean_text(row.get("fund_name", ""))
    manager_name = clean_text(row.get("manager_name", ""))
    vintage_year = row.get("vintage_year", "")
    strategy = clean_text(row.get("strategy", ""))
    geography = clean_text(row.get("geography", ""))
    source_file = clean_text(row.get("source_file", ""))

    irr = row.get("irr_numeric", np.nan)
    tvpi = row.get("tvpi_numeric", np.nan)
    dpi = row.get("dpi_numeric", np.nan)

    overall_quartile = clean_text(row.get("overall_vintage_quartile", ""))
    label = clean_text(row.get("vintage_benchmark_label", ""))

    if overall_quartile == "":
        return (
            f"{fund_name} by {manager_name} is included from {source_file}. "
            f"The extracted data currently includes vintage/strategy/geography information "
            f"but does not include enough IRR, TVPI, or DPI data to calculate a full vintage quartile."
        )

    return (
        f"{fund_name} by {manager_name} is a {vintage_year} {strategy} fund in {geography}. "
        f"Extracted performance metrics: IRR={irr}, TVPI={tvpi}, DPI={dpi}. "
        f"Overall vintage quartile: {overall_quartile}. Interpretation: {label}."
    )


# ============================================================
# SAVE
# ============================================================

def save_outputs(enriched_df, benchmarks_df):
    enriched_df.to_csv(ENRICHED_FUNDS_CSV_PATH, index=False, encoding="utf-8-sig")
    benchmarks_df.to_csv(BENCHMARKS_CSV_PATH, index=False, encoding="utf-8-sig")

    conn = sqlite3.connect(DATABASE_PATH)

    enriched_df.to_sql("funds_with_vintage_quartiles", conn, if_exists="replace", index=False)
    benchmarks_df.to_sql("vintage_benchmarks", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_enriched_fund_name ON funds_with_vintage_quartiles(fund_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_enriched_vintage ON funds_with_vintage_quartiles(vintage_year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_enriched_strategy_geo ON funds_with_vintage_quartiles(strategy, geography)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_strategy_geo_vintage ON vintage_benchmarks(strategy, geography, vintage_year)")

    conn.commit()
    conn.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("")
    print("====================================================")
    print("BUILD VINTAGE BENCHMARKS / QUARTILES")
    print("====================================================")
    print("")

    df = load_funds()

    print(f"Loaded fund rows: {len(df)}")

    performance_rows = df[
        df["irr_numeric"].notna()
        | df["tvpi_numeric"].notna()
        | df["dpi_numeric"].notna()
    ].copy()

    print(f"Rows with any IRR / TVPI / DPI data: {len(performance_rows)}")

    benchmarks_df = build_benchmark_table(df)
    enriched_df = add_fund_percentiles_and_quartiles(df)

    save_outputs(enriched_df, benchmarks_df)

    print("")
    print("====================================================")
    print("DONE")
    print("====================================================")
    print("")
    print("Files created:")
    print(f"  {ENRICHED_FUNDS_CSV_PATH}")
    print(f"  {BENCHMARKS_CSV_PATH}")
    print(f"  {DATABASE_PATH}")
    print("")
    print(f"Vintage benchmark rows created: {len(benchmarks_df)}")

    if len(benchmarks_df) == 0:
        print("")
        print("IMPORTANT:")
        print("No IRR / TVPI / DPI benchmarks were created because the current PDFs")
        print("do not appear to contain numeric performance metrics.")
        print("")
        print("This is okay. The website will still show the fund screener, sources,")
        print("fund sizes, vintages, strategy, geography, and quality flags.")
        print("")
        print("When you upload PDFs with IRR / TVPI / DPI tables, this same script")
        print("will calculate real vintage quartiles.")

    else:
        print("")
        print("Sample benchmark rows:")
        print(benchmarks_df.head(20).to_string(index=False))

    print("")
    print("Next step after this works: create app.py")


if __name__ == "__main__":
    main()