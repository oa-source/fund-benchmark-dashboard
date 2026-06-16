import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Private Fund Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATABASE_PATH = DATA_DIR / "funds_database.sqlite"
ENRICHED_FUNDS_CSV_PATH = DATA_DIR / "funds_with_vintage_quartiles.csv"
BENCHMARKS_CSV_PATH = DATA_DIR / "vintage_benchmarks.csv"


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def safe_number(value):
    if value is None:
        return None

    if pd.isna(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def format_value(value, metric=None):
    value = safe_number(value)

    if value is None:
        return "N/A"

    if metric == "irr":
        return f"{value:.1f}%"

    if metric in ["tvpi", "dpi"]:
        return f"{value:.2f}x"

    return f"{value:,.2f}"


def vintage_focus_filter(df, option, specific_vintage=None):
    df = df.copy()

    if "vintage_year" not in df.columns:
        return df

    df["vintage_year"] = pd.to_numeric(df["vintage_year"], errors="coerce")

    if option == "All available vintages":
        return df

    if option == "Mature vintages only":
        return df[df["vintage_year"] <= 2018].copy()

    if option == "Recent / developing vintages":
        return df[df["vintage_year"] >= 2019].copy()

    if option == "2008 and newer":
        return df[df["vintage_year"] >= 2008].copy()

    if option == "Specific vintage" and specific_vintage is not None:
        return df[df["vintage_year"] == specific_vintage].copy()

    return df


def make_allocator_memo(row):
    fund_name = clean_text(row.get("fund_name", ""))
    manager_name = clean_text(row.get("manager_name", ""))
    source_file = clean_text(row.get("source_file", ""))
    strategy = clean_text(row.get("strategy", ""))
    geography = clean_text(row.get("geography", ""))
    vintage_year = clean_text(row.get("vintage_year", ""))
    data_quality_flag = clean_text(row.get("data_quality_flag", ""))

    irr = format_value(row.get("irr_numeric", None), "irr")
    tvpi = format_value(row.get("tvpi_numeric", None), "tvpi")
    dpi = format_value(row.get("dpi_numeric", None), "dpi")

    internal_score = row.get("internal_score", "")
    overall_quartile = clean_text(row.get("overall_vintage_quartile", ""))
    irr_quartile = clean_text(row.get("irr_vintage_quartile", ""))
    tvpi_quartile = clean_text(row.get("tvpi_vintage_quartile", ""))
    dpi_quartile = clean_text(row.get("dpi_vintage_quartile", ""))
    label = clean_text(row.get("vintage_benchmark_label", ""))

    if overall_quartile == "":
        overall_quartile = "N/A"

    if label == "":
        label = "Insufficient performance data"

    if internal_score == "" or pd.isna(internal_score):
        internal_score_text = "N/A"
    else:
        internal_score_text = f"{float(internal_score):.1f} / 100"

    memo = f"""
### Allocator Memo

**Fund:** {fund_name}  
**Manager:** {manager_name}  
**Vintage:** {vintage_year}  
**Strategy:** {strategy}  
**Geography:** {geography}  
**Source PDF:** {source_file}  
**Data quality flag:** {data_quality_flag}  

#### Extracted metrics

| Metric | Value |
|---|---:|
| IRR | {irr} |
| TVPI | {tvpi} |
| DPI | {dpi} |

#### Benchmark result

| Item | Result |
|---|---|
| Internal score | {internal_score_text} |
| Overall vintage quartile | {overall_quartile} |
| IRR vintage quartile | {irr_quartile if irr_quartile else "N/A"} |
| TVPI vintage quartile | {tvpi_quartile if tvpi_quartile else "N/A"} |
| DPI vintage quartile | {dpi_quartile if dpi_quartile else "N/A"} |
| Plain-English result | {label} |

#### Plain-English interpretation

This fund is included in the extracted database based on the uploaded PDF evidence.  
For the current PDFs, many rows include fund name, vintage, strategy, geography, and fund size, but do not include IRR / TVPI / DPI.  
When performance PDFs are added, this memo will automatically show vintage-relative quartiles and a stronger benchmark conclusion.
"""
    return memo


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if ENRICHED_FUNDS_CSV_PATH.exists():
        funds_df = pd.read_csv(ENRICHED_FUNDS_CSV_PATH)
    else:
        funds_df = pd.DataFrame()

    if BENCHMARKS_CSV_PATH.exists():
        benchmarks_df = pd.read_csv(BENCHMARKS_CSV_PATH)
    else:
        benchmarks_df = pd.DataFrame()

    if funds_df.empty:
        return funds_df, benchmarks_df

    required_columns = [
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
        "irr_numeric",
        "tvpi_numeric",
        "dpi_numeric",
        "irr_vintage_percentile",
        "tvpi_vintage_percentile",
        "dpi_vintage_percentile",
        "irr_vintage_quartile",
        "tvpi_vintage_quartile",
        "dpi_vintage_quartile",
        "overall_vintage_quartile",
        "vintage_benchmark_label",
        "internal_score",
        "allocator_memo_short",
    ]

    for col in required_columns:
        if col not in funds_df.columns:
            funds_df[col] = ""

    funds_df["vintage_year"] = pd.to_numeric(funds_df["vintage_year"], errors="coerce")
    funds_df["source_page"] = pd.to_numeric(funds_df["source_page"], errors="coerce")

    for col in ["irr_numeric", "tvpi_numeric", "dpi_numeric", "internal_score"]:
        funds_df[col] = pd.to_numeric(funds_df[col], errors="coerce")

    text_cols = [
        "fund_name",
        "manager_name",
        "strategy",
        "geography",
        "source_name",
        "source_file",
        "raw_text",
        "data_quality_flag",
        "overall_vintage_quartile",
        "vintage_benchmark_label",
    ]

    for col in text_cols:
        funds_df[col] = funds_df[col].fillna("").astype(str)

    return funds_df, benchmarks_df


funds_df, benchmarks_df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Private Fund Benchmark")
st.sidebar.caption("PDF extraction → CSV / SQLite → dashboard")

if funds_df.empty:
    st.error("No data found. Run extract_pdfs_to_database.py and build_vintage_benchmarks.py first.")
    st.stop()

total_rows = len(funds_df)
performance_rows = funds_df[
    funds_df["irr_numeric"].notna()
    | funds_df["tvpi_numeric"].notna()
    | funds_df["dpi_numeric"].notna()
]

st.sidebar.metric("Extracted fund rows", f"{total_rows:,}")
st.sidebar.metric("Rows with IRR / TVPI / DPI", f"{len(performance_rows):,}")

st.sidebar.info(
    "The current uploaded PDFs are mostly market-map / fundraising reports. "
    "They contain many fund names, vintages, strategies, geographies, and fund sizes, "
    "but little or no IRR / TVPI / DPI performance data."
)


# ============================================================
# HEADER
# ============================================================

st.title("Private Fund Benchmark / Quartile Analysis")
st.caption("Built from uploaded PDF reports with source evidence and data quality flags.")


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "Fund Screener",
    "Benchmark / Quartile Context",
    "Data Quality & Sources",
])


# ============================================================
# TAB 1 — FUND SCREENER
# ============================================================

with tab1:
    st.header("Fund Screener")

    col1, col2, col3, col4 = st.columns(4)

    strategy_options = ["All"] + sorted([x for x in funds_df["strategy"].dropna().unique() if clean_text(x) != ""])
    geography_options = ["All"] + sorted([x for x in funds_df["geography"].dropna().unique() if clean_text(x) != ""])

    vintages = sorted([int(x) for x in funds_df["vintage_year"].dropna().unique()])

    with col1:
        selected_strategy = st.selectbox("Strategy", strategy_options, index=0)

    with col2:
        selected_geography = st.selectbox("Geography", geography_options, index=0)

    with col3:
        selected_vintage = st.selectbox("Vintage", ["All"] + vintages, index=0)

    with col4:
        selected_quality = st.selectbox(
            "Data quality",
            ["All", "clean_extraction only", "needs_review only"],
            index=0,
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        min_irr = st.number_input("Minimum IRR", value=0.0, step=1.0)

    with col6:
        min_tvpi = st.number_input("Minimum TVPI", value=0.0, step=0.1)

    with col7:
        min_dpi = st.number_input("Minimum DPI", value=0.0, step=0.1)

    with col8:
        selected_quartile = st.selectbox("Quartile", ["All", "Q1", "Q2", "Q3", "Q4"], index=0)

    filtered_df = funds_df.copy()

    if selected_strategy != "All":
        filtered_df = filtered_df[filtered_df["strategy"] == selected_strategy].copy()

    if selected_geography != "All":
        filtered_df = filtered_df[filtered_df["geography"] == selected_geography].copy()

    if selected_vintage != "All":
        filtered_df = filtered_df[filtered_df["vintage_year"] == selected_vintage].copy()

    if selected_quality == "clean_extraction only":
        filtered_df = filtered_df[
            ~filtered_df["data_quality_flag"].str.contains("needs_review", na=False)
        ].copy()

    if selected_quality == "needs_review only":
        filtered_df = filtered_df[
            filtered_df["data_quality_flag"].str.contains("needs_review", na=False)
        ].copy()

    if min_irr > 0:
        filtered_df = filtered_df[filtered_df["irr_numeric"].fillna(-999) >= min_irr].copy()

    if min_tvpi > 0:
        filtered_df = filtered_df[filtered_df["tvpi_numeric"].fillna(-999) >= min_tvpi].copy()

    if min_dpi > 0:
        filtered_df = filtered_df[filtered_df["dpi_numeric"].fillna(-999) >= min_dpi].copy()

    if selected_quartile != "All":
        filtered_df = filtered_df[filtered_df["overall_vintage_quartile"] == selected_quartile].copy()

    st.subheader("Filtered funds")
    st.caption(f"Showing {len(filtered_df):,} rows")

    display_cols = [
        "fund_name",
        "manager_name",
        "vintage_year",
        "strategy",
        "geography",
        "fund_size",
        "irr_numeric",
        "tvpi_numeric",
        "dpi_numeric",
        "overall_vintage_quartile",
        "vintage_benchmark_label",
        "source_name",
        "source_file",
        "source_page",
        "data_quality_flag",
    ]

    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        height=420,
    )

    st.divider()

    st.subheader("Selected fund detail")

    if filtered_df.empty:
        st.warning("No funds match the selected filters.")
    else:
        fund_options = (
            filtered_df["fund_name"].fillna("").astype(str)
            + " | "
            + filtered_df["source_file"].fillna("").astype(str)
            + " | page "
            + filtered_df["source_page"].fillna("").astype(str)
        ).tolist()

        selected_fund_label = st.selectbox("Select fund", fund_options)

        selected_index = fund_options.index(selected_fund_label)
        selected_row = filtered_df.iloc[selected_index]

        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)

        with detail_col1:
            st.metric("Fund IRR", format_value(selected_row.get("irr_numeric"), "irr"))

        with detail_col2:
            st.metric("Fund TVPI", format_value(selected_row.get("tvpi_numeric"), "tvpi"))

        with detail_col3:
            st.metric("Fund DPI", format_value(selected_row.get("dpi_numeric"), "dpi"))

        with detail_col4:
            st.metric("Vintage quartile", clean_text(selected_row.get("overall_vintage_quartile")) or "N/A")

        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Fund information")
            st.write(f"**Fund name:** {clean_text(selected_row.get('fund_name'))}")
            st.write(f"**Manager:** {clean_text(selected_row.get('manager_name'))}")
            st.write(f"**Vintage:** {clean_text(selected_row.get('vintage_year'))}")
            st.write(f"**Strategy:** {clean_text(selected_row.get('strategy'))}")
            st.write(f"**Geography:** {clean_text(selected_row.get('geography'))}")
            st.write(f"**Fund size:** {clean_text(selected_row.get('fund_size'))}")
            st.write(f"**Data quality:** {clean_text(selected_row.get('data_quality_flag'))}")

        with right:
            st.markdown("### Source evidence")
            st.write(f"**Source name:** {clean_text(selected_row.get('source_name'))}")
            st.write(f"**Source PDF:** {clean_text(selected_row.get('source_file'))}")
            st.write(f"**Source page:** {clean_text(selected_row.get('source_page'))}")
            st.text_area(
                "Raw evidence from PDF",
                clean_text(selected_row.get("raw_text")),
                height=170,
            )

        st.markdown(make_allocator_memo(selected_row))


# ============================================================
# TAB 2 — BENCHMARK / QUARTILE CONTEXT
# ============================================================

with tab2:
    st.header("Benchmark / Quartile Context")

    st.info(
        "The box-and-whisker quartile chart needs numeric IRR / TVPI / DPI values. "
        "Your current PDFs appear to have 0 rows with those performance metrics, "
        "so the chart will appear once performance PDFs are added."
    )

    metric_label_map = {
        "IRR (%)": "irr",
        "TVPI": "tvpi",
        "DPI": "dpi",
    }

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_label = st.selectbox("Metric", ["IRR (%)", "TVPI", "DPI"], index=0)
        metric = metric_label_map[metric_label]

    with col2:
        benchmark_strategy = st.selectbox(
            "Strategy filter",
            strategy_options,
            index=0,
            key="benchmark_strategy",
        )

    with col3:
        benchmark_geography = st.selectbox(
            "Geography filter",
            geography_options,
            index=0,
            key="benchmark_geography",
        )

    with col4:
        vintage_focus = st.selectbox(
            "Vintage Focus",
            [
                "All available vintages",
                "Mature vintages only",
                "Recent / developing vintages",
                "2008 and newer",
                "Specific vintage",
            ],
            index=0,
        )

    specific_vintage = None

    if vintage_focus == "Specific vintage":
        specific_vintage = st.selectbox("Choose specific vintage", vintages, index=0)

    chart_df = funds_df.copy()

    if benchmark_strategy != "All":
        chart_df = chart_df[chart_df["strategy"] == benchmark_strategy].copy()

    if benchmark_geography != "All":
        chart_df = chart_df[chart_df["geography"] == benchmark_geography].copy()

    chart_df = vintage_focus_filter(chart_df, vintage_focus, specific_vintage)

    metric_col = f"{metric}_numeric"
    chart_df = chart_df[chart_df[metric_col].notna()].copy()

    if metric == "dpi":
        st.warning("DPI is most useful for mature vintages. Young funds naturally have low DPI.")

    if chart_df.empty:
        st.warning(
            f"No numeric {metric_label} data available for this selection yet. "
            "Upload PDFs with IRR / TVPI / DPI performance tables, rerun the scripts, "
            "and this chart will populate."
        )
    else:
        fig = go.Figure()

        vintages_for_chart = sorted(chart_df["vintage_year"].dropna().unique())

        for vintage in vintages_for_chart:
            vintage_values = chart_df[chart_df["vintage_year"] == vintage][metric_col].dropna()

            fig.add_trace(
                go.Box(
                    y=vintage_values,
                    name=str(int(vintage)),
                    boxmean=False,
                    points="outliers",
                    hovertemplate=f"Vintage {int(vintage)}<br>{metric_label}: %{{y}}<extra></extra>",
                )
            )

            average_value = vintage_values.mean()

            fig.add_trace(
                go.Scatter(
                    x=[str(int(vintage))],
                    y=[average_value],
                    mode="markers+text",
                    marker=dict(symbol="diamond", size=12, color="green"),
                    text=[f"Avg {average_value:.2f}"],
                    textposition="top center",
                    showlegend=False,
                    hovertemplate=f"Vintage {int(vintage)} average<br>{metric_label}: {average_value:.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"{metric_label} by Vintage Year",
            xaxis_title="Vintage Year",
            yaxis_title=metric_label,
            height=650,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Benchmark table")

    if benchmarks_df.empty:
        st.warning("No benchmark table exists yet because no IRR / TVPI / DPI rows were found.")
    else:
        benchmark_display = benchmarks_df.copy()

        if benchmark_strategy != "All":
            benchmark_display = benchmark_display[benchmark_display["strategy"] == benchmark_strategy].copy()

        if benchmark_geography != "All":
            benchmark_display = benchmark_display[benchmark_display["geography"] == benchmark_geography].copy()

        benchmark_display = benchmark_display[benchmark_display["metric"] == metric].copy()

        st.dataframe(benchmark_display, use_container_width=True, height=400)


# ============================================================
# TAB 3 — DATA QUALITY & SOURCES
# ============================================================

with tab3:
    st.header("Data Quality & Sources")

    q1, q2, q3, q4 = st.columns(4)

    clean_rows = funds_df[
        ~funds_df["data_quality_flag"].astype(str).str.contains("needs_review", na=False)
    ]

    review_rows = funds_df[
        funds_df["data_quality_flag"].astype(str).str.contains("needs_review", na=False)
    ]

    with q1:
        st.metric("Total rows", f"{len(funds_df):,}")

    with q2:
        st.metric("Cleaner rows", f"{len(clean_rows):,}")

    with q3:
        st.metric("Needs-review rows", f"{len(review_rows):,}")

    with q4:
        st.metric("Source PDFs", f"{funds_df['source_file'].nunique():,}")

    st.subheader("Rows by source PDF")

    source_counts = (
        funds_df.groupby("source_file")
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
    )

    st.dataframe(source_counts, use_container_width=True, height=300)

    st.subheader("Rows by data quality flag")

    quality_counts = (
        funds_df.groupby("data_quality_flag")
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
    )

    st.dataframe(quality_counts, use_container_width=True, height=300)

    st.subheader("Needs-review rows")

    st.caption(
        "These rows may be short labels, merged labels, or extracted from complicated market-map pages. "
        "They are kept in the database so you can trace and clean them later."
    )

    review_display_cols = [
        "fund_name",
        "manager_name",
        "vintage_year",
        "strategy",
        "geography",
        "fund_size",
        "source_file",
        "source_page",
        "raw_text",
        "data_quality_flag",
    ]

    st.dataframe(
        review_rows[review_display_cols],
        use_container_width=True,
        height=450,
    )