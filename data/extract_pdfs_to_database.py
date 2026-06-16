import re
import sqlite3
from pathlib import Path

import fitz
import pandas as pd


# ============================================================
# FOLDER SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATA_DIR / "funds_database.sqlite"
FUNDS_CSV_PATH = DATA_DIR / "extracted_funds.csv"
RAW_PAGES_CSV_PATH = DATA_DIR / "raw_pdf_pages.csv"
REVIEW_CSV_PATH = DATA_DIR / "needs_review_rows.csv"


FINAL_COLUMNS = [
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


# ============================================================
# CLEANING HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\u00a0", " ")
    value = value.replace("\u2013", "-")
    value = value.replace("\u2014", "-")
    value = value.replace("ﬁ", "fi")
    value = value.replace("ﬂ", "fl")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def clean_fund_name(value):
    value = clean_text(value)

    value = re.sub(r"^\(?[$€£][0-9,\.]+\*?\)?\s*", "", value)
    value = re.sub(r"\s+\(?[$€£][0-9,\.]+\s*(m|bn|b)?\*?\)?$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\(?[0-9]+\)?\s+", "", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip(" -|,;:")


def clean_money(value):
    value = clean_text(value)
    value = value.replace("*", "")
    value = value.replace("(", "")
    value = value.replace(")", "")
    return value.strip()


def extract_money_values(text):
    text = clean_text(text)

    matches = re.findall(
        r"(\$|€|£)\s?[0-9][0-9,\.]*\s*(?:m|bn|b)?\*?",
        text,
        flags=re.IGNORECASE,
    )

    full_matches = re.findall(
        r"(?:\$|€|£)\s?[0-9][0-9,\.]*\s*(?:m|bn|b)?\*?",
        text,
        flags=re.IGNORECASE,
    )

    return [clean_money(x) for x in full_matches]


def remove_money_from_text(text):
    text = clean_text(text)
    text = re.sub(r"\(?\s*(?:\$|€|£)\s?[0-9][0-9,\.]*\s*(?:m|bn|b)?\*?\s*\)?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_vintage_year(text):
    text = clean_text(text)
    match = re.search(r"\b(19[8-9][0-9]|20[0-9]{2})\b", text)

    if match:
        year = int(match.group(1))
        if 1980 <= year <= 2035:
            return year

    return None


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_strategy(text):
    text = clean_text(text).lower()

    if any(term in text for term in [
        "venture capital",
        "venture universe",
        "venture funds",
        " vc ",
        " vc",
        "early stage",
        "seed",
    ]):
        return "venture_capital"

    if any(term in text for term in [
        "growth equity",
        "growth market",
        "growth universe",
        "north america growth",
    ]):
        return "growth"

    if any(term in text for term in [
        "buyout",
        "lbo",
        "private equity",
        "pe market",
        "equity market map",
    ]):
        return "buyout"

    return ""


def normalize_geography(text):
    text = clean_text(text).lower()

    if any(term in text for term in [
        "north america",
        "north american",
        "united states",
        "u.s.",
        "usa",
        "canada",
    ]):
        return "north_america"

    if any(term in text for term in [
        "europe",
        "european",
        "dach",
        "benelux",
        "southern europe",
        "uk and ireland",
        "nordics",
        "iberia",
        "france",
        "italy",
        "germany",
        "switzerland",
        "austria",
        "netherlands",
        "belgium",
        "luxembourg",
    ]):
        return "europe"

    if any(term in text for term in [
        "asia pacific",
        "apac",
        "asia",
        "japan",
        "china",
        "india",
        "australia",
        "korea",
    ]):
        return "asia_pacific"

    if any(term in text for term in [
        "latin america",
        "latam",
        "brazil",
        "mexico",
        "chile",
        "colombia",
    ]):
        return "latin_america"

    if any(term in text for term in [
        "emerging markets",
        "emerging market",
    ]):
        return "emerging_markets"

    if any(term in text for term in [
        "global",
        "worldwide",
    ]):
        return "global"

    return ""


def infer_source_name(source_file):
    source_file_l = source_file.lower()

    if "evercore" in source_file_l:
        return "Evercore PFG"

    if "jefferies" in source_file_l:
        return "Jefferies PFA"

    if "ubs" in source_file_l:
        return "UBS PFG"

    return "Unknown"


def infer_strategy(source_file, page_text):
    source_file_l = source_file.lower()

    if "north america growth" in source_file_l:
        return "growth"

    if "north america buyout" in source_file_l:
        return "buyout"

    if "venture" in source_file_l:
        return "venture_capital"

    return normalize_strategy(source_file + " " + page_text[:1500])


def infer_geography(source_file, page_text):
    return normalize_geography(source_file + " " + page_text[:1500])


# ============================================================
# MANAGER ESTIMATION
# ============================================================

def estimate_manager_name(fund_name):
    fund_name = clean_fund_name(fund_name)

    if fund_name == "":
        return ""

    roman_numerals = {
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII",
        "IX", "X", "XI", "XII", "XIII", "XIV", "XV",
    }

    words = fund_name.split()
    manager_words = []

    for word in words:
        clean_word = word.replace(",", "").strip()

        if clean_word in roman_numerals:
            break

        manager_words.append(word)

        if clean_word.lower() in [
            "capital",
            "partners",
            "investors",
            "equity",
            "ventures",
            "management",
            "holdings",
        ] and len(manager_words) >= 2:
            break

    manager = " ".join(manager_words).strip()

    if manager == "":
        return fund_name

    return manager


# ============================================================
# QUALITY FILTERS
# ============================================================

def is_bad_name(name):
    name = clean_fund_name(name)
    name_l = name.lower()

    if len(name) < 4:
        return True

    bad_terms = [
        "source",
        "notes",
        "fund size",
        "status/vintage",
        "confirmed in market",
        "projected",
        "table of contents",
        "private funds group",
        "strictly confidential",
        "disclaimer",
        "market map",
        "contents",
        "data as of",
        "subject to change",
        "statistics based",
        "average fund size",
        "funds currently raising",
        "photo credit",
        "cover photo",
        "potentially returning",
        "closed fund",
        "current funds",
        "all fund sizes",
        "based on preqin",
        "years represent vintages",
        "prior fund size",
    ]

    if any(term in name_l for term in bad_terms):
        return True

    if re.fullmatch(r"[0-9]+", name):
        return True

    if len(name.split()) > 9:
        return True

    return False


def quality_flag_for_name(name):
    name = clean_fund_name(name)

    if len(name.split()) <= 2:
        return "needs_review_short_name"

    if len(name.split()) >= 8:
        return "needs_review_possible_merged_name"

    return "clean_extraction"


# ============================================================
# PDF READING
# ============================================================

def extract_pdf_pages(pdf_path):
    records = []

    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_text = page.get_text("text")

        blocks = []

        raw_blocks = page.get_text("blocks")

        for block in raw_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block

            block_text = clean_text(text)

            if block_text:
                blocks.append({
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "text": block_text,
                })

        records.append({
            "source_file": pdf_path.name,
            "source_page": page_index + 1,
            "raw_page_text": page_text,
            "blocks": blocks,
        })

    doc.close()

    return records


# ============================================================
# PARSER 1: BLOCK-BASED MAP / LABEL EXTRACTION
# ============================================================

def parse_blocks(page_record):
    rows = []

    source_file = page_record["source_file"]
    source_page = page_record["source_page"]
    page_text = page_record["raw_page_text"]

    source_name = infer_source_name(source_file)
    strategy = infer_strategy(source_file, page_text)
    geography = infer_geography(source_file, page_text)

    page_vintage = None

    for line in page_text.splitlines()[:25]:
        possible = extract_vintage_year(line)
        if possible:
            page_vintage = possible
            break

    for block in page_record["blocks"]:
        block_text = clean_text(block["text"])

        money_values = extract_money_values(block_text)

        if not money_values:
            continue

        # Ignore chart axis / legend blocks
        if any(axis_term in block_text.lower() for axis_term in [
            "fund size",
            "confirmed in market",
            "projected",
            "source:",
            "notes:",
        ]):
            continue

        # A block can sometimes contain multiple fund labels.
        # Split on money values and process each piece.
        pieces = re.split(
            r"((?:\$|€|£)\s?[0-9][0-9,\.]*\s*(?:m|bn|b)?\*?)",
            block_text,
            flags=re.IGNORECASE,
        )

        for i in range(1, len(pieces), 2):
            fund_size = clean_money(pieces[i])

            before_text = pieces[i - 1]
            before_text = remove_money_from_text(before_text)

            # Keep only the closest label before the fund size.
            label_lines = [clean_fund_name(x) for x in re.split(r"\|+|\n+", before_text)]
            label_lines = [x for x in label_lines if x and not is_bad_name(x)]

            if not label_lines:
                possible_name = clean_fund_name(before_text)
            else:
                possible_name = clean_fund_name(label_lines[-1])

            # If the block is one normal label, use the cleaned full text before the money.
            if len(possible_name.split()) <= 1:
                possible_name = clean_fund_name(before_text)

            if is_bad_name(possible_name):
                continue

            local_vintage = extract_vintage_year(block_text)
            vintage_year = local_vintage or page_vintage

            quality = quality_flag_for_name(possible_name)

            rows.append({
                "fund_name": possible_name,
                "manager_name": estimate_manager_name(possible_name),
                "vintage_year": vintage_year,
                "strategy": strategy,
                "geography": geography,
                "irr": None,
                "tvpi": None,
                "dpi": None,
                "nav": None,
                "commitment": None,
                "paid_in": None,
                "distributions": None,
                "fund_size": fund_size,
                "source_name": source_name,
                "source_file": source_file,
                "source_page": source_page,
                "raw_text": block_text,
                "data_quality_flag": quality,
            })

    return rows


# ============================================================
# PARSER 2: UBS TEXT TABLE FALLBACK
# ============================================================

def parse_ubs_lines(page_record):
    rows = []

    source_file = page_record["source_file"]
    source_page = page_record["source_page"]
    page_text = page_record["raw_page_text"]

    if "ubs" not in source_file.lower():
        return rows

    source_name = infer_source_name(source_file)
    strategy = infer_strategy(source_file, page_text)
    geography = infer_geography(source_file, page_text)

    lines = [clean_text(x) for x in page_text.splitlines()]
    lines = [x for x in lines if x]

    for i, line in enumerate(lines):
        money_values = extract_money_values(line)

        if not money_values:
            continue

        if any(term in line.lower() for term in ["fund size", "source:", "notes:"]):
            continue

        fund_size = money_values[0]

        nearby_before = []

        for j in range(max(0, i - 4), i):
            candidate = clean_fund_name(lines[j])

            if is_bad_name(candidate):
                continue

            if candidate.lower() in ["raising", "held close", "closed"]:
                continue

            nearby_before.append(candidate)

        if not nearby_before:
            # Sometimes UBS has the fund name and money on the same line.
            possible_name = clean_fund_name(remove_money_from_text(line))
        else:
            possible_name = clean_fund_name(nearby_before[-1])

        if is_bad_name(possible_name):
            continue

        vintage_year = None

        for j in range(i + 1, min(len(lines), i + 5)):
            possible_year = extract_vintage_year(lines[j])
            if possible_year:
                vintage_year = possible_year
                break

        raw_text = clean_text(" | ".join(lines[max(0, i - 4):min(len(lines), i + 5)]))

        rows.append({
            "fund_name": possible_name,
            "manager_name": estimate_manager_name(possible_name),
            "vintage_year": vintage_year,
            "strategy": strategy,
            "geography": geography,
            "irr": None,
            "tvpi": None,
            "dpi": None,
            "nav": None,
            "commitment": None,
            "paid_in": None,
            "distributions": None,
            "fund_size": fund_size,
            "source_name": source_name,
            "source_file": source_file,
            "source_page": source_page,
            "raw_text": raw_text,
            "data_quality_flag": quality_flag_for_name(possible_name),
        })

    return rows


# ============================================================
# PAGE PARSER
# ============================================================

def parse_page(page_record):
    rows = []

    rows.extend(parse_blocks(page_record))
    rows.extend(parse_ubs_lines(page_record))

    return rows


# ============================================================
# CLEAN FINAL DATAFRAME
# ============================================================

def clean_funds_dataframe(df):
    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[FINAL_COLUMNS].copy()

    for col in df.columns:
        df[col] = df[col].apply(lambda x: clean_text(x) if x is not None else None)

    df["fund_name"] = df["fund_name"].apply(clean_fund_name)
    df["manager_name"] = df["manager_name"].apply(clean_fund_name)

    df["vintage_year"] = pd.to_numeric(df["vintage_year"], errors="coerce")
    df["source_page"] = pd.to_numeric(df["source_page"], errors="coerce")

    df = df[df["fund_name"].notna()].copy()
    df = df[df["fund_name"].astype(str).str.len() >= 4].copy()
    df = df[~df["fund_name"].apply(is_bad_name)].copy()

    df["_dedupe_key"] = (
        df["fund_name"].astype(str).str.lower().str.strip()
        + "|"
        + df["source_file"].astype(str).str.lower().str.strip()
        + "|"
        + df["source_page"].astype(str).str.strip()
        + "|"
        + df["fund_size"].astype(str).str.lower().str.strip()
    )

    df = df.drop_duplicates(subset=["_dedupe_key"]).copy()
    df = df.drop(columns=["_dedupe_key"])

    df = df.sort_values(
        by=["source_file", "source_page", "fund_name"],
        ascending=[True, True, True],
    ).copy()

    return df


# ============================================================
# SQLITE SAVE
# ============================================================

def save_to_sqlite(funds_df, raw_pages_df):
    conn = sqlite3.connect(DATABASE_PATH)

    funds_df.to_sql("funds", conn, if_exists="replace", index=False)
    raw_pages_df.to_sql("raw_pdf_pages", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_funds_name ON funds(fund_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_funds_vintage ON funds(vintage_year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_funds_strategy ON funds(strategy)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_funds_geo ON funds(geography)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_funds_source ON funds(source_file)")

    conn.commit()
    conn.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("")
    print("====================================================")
    print("PRIVATE FUND PDF EXTRACTION - CLEANER VERSION")
    print("====================================================")
    print("")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"ERROR: No PDFs found in: {PDF_DIR}")
        return

    print(f"PDF folder: {PDF_DIR}")
    print(f"Found {len(pdf_files)} PDF file(s).")
    print("")

    all_page_records_for_csv = []
    all_fund_rows = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        try:
            page_records = extract_pdf_pages(pdf_path)

            pdf_rows = []

            for page_record in page_records:
                all_page_records_for_csv.append({
                    "source_file": page_record["source_file"],
                    "source_page": page_record["source_page"],
                    "raw_page_text": page_record["raw_page_text"],
                })

                page_rows = parse_page(page_record)
                pdf_rows.extend(page_rows)

            all_fund_rows.extend(pdf_rows)

            print(f"  Pages read: {len(page_records)}")
            print(f"  Fund rows found before final cleaning: {len(pdf_rows)}")
            print("")

        except Exception as e:
            print(f"  ERROR reading file: {pdf_path.name}")
            print(f"  Error details: {e}")
            print("")

    raw_pages_df = pd.DataFrame(all_page_records_for_csv)
    funds_df = pd.DataFrame(all_fund_rows)
    funds_df = clean_funds_dataframe(funds_df)

    review_df = funds_df[
        funds_df["data_quality_flag"].astype(str).str.contains("needs_review", na=False)
    ].copy()

    clean_df = funds_df[
        ~funds_df["data_quality_flag"].astype(str).str.contains("needs_review", na=False)
    ].copy()

    raw_pages_df.to_csv(RAW_PAGES_CSV_PATH, index=False, encoding="utf-8-sig")
    funds_df.to_csv(FUNDS_CSV_PATH, index=False, encoding="utf-8-sig")
    review_df.to_csv(REVIEW_CSV_PATH, index=False, encoding="utf-8-sig")

    save_to_sqlite(funds_df, raw_pages_df)

    print("====================================================")
    print("DONE")
    print("====================================================")
    print("")
    print(f"Raw PDF pages saved: {len(raw_pages_df)}")
    print(f"Total extracted fund rows saved: {len(funds_df)}")
    print(f"Cleaner rows: {len(clean_df)}")
    print(f"Needs-review rows: {len(review_df)}")
    print("")
    print("Files created:")
    print(f"  {FUNDS_CSV_PATH}")
    print(f"  {REVIEW_CSV_PATH}")
    print(f"  {RAW_PAGES_CSV_PATH}")
    print(f"  {DATABASE_PATH}")
    print("")

    if not funds_df.empty:
        print("Sample rows:")
        preview_cols = [
            "fund_name",
            "manager_name",
            "vintage_year",
            "strategy",
            "geography",
            "fund_size",
            "source_file",
            "source_page",
            "data_quality_flag",
        ]

        print(funds_df[preview_cols].head(30).to_string(index=False))

    print("")
    print("Next step after this looks cleaner: build_vintage_benchmarks.py")


if __name__ == "__main__":
    main()