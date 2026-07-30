"""
data_loader.py
Handles reading the uploaded file (CSV / Excel) and running initial data inspection.
"""

import pandas as pd
import io


def load_file(uploaded_file) -> pd.DataFrame:
    """
    Reads a CSV or Excel file uploaded via Streamlit (UploadedFile object)
    and returns a DataFrame.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        # Try a few common encodings to handle special characters / non-UTF8 files
        raw_bytes = uploaded_file.read()
        for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("Could not read the CSV file. Please check the file format.")

    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel (.xlsx/.xls) file.")


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary with core inspection info about the dataset:
    - row/column counts
    - data types
    - missing values
    - duplicate rows
    - a sample of the data
    """
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2) if len(df) > 0 else missing

    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": missing.values,
        "Missing %": missing_pct.values,
        "Unique Values": [df[c].nunique() for c in df.columns],
    })

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # Try to detect date columns stored as text
    possible_date_cols = []
    for c in categorical_cols:
        if any(k in c.lower() for k in ["date", "time", "day", "month", "year"]):
            possible_date_cols.append(c)

    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "duplicates": int(df.duplicated().sum()),
        "dtypes_table": dtypes_df,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "possible_date_cols": possible_date_cols,
        "sample": df.head(10),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
    }
