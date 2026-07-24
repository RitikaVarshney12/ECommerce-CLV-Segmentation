"""
preprocessing.py
-----------------
Modular data-loading and preprocessing utilities for the
E-Commerce Customer Lifetime Value (CLV) & Segmentation project.

Steps performed (as required by the project scope):
    1. Load dataset
    2. Handle missing values
    3. Remove duplicate records
    4. Encode categorical variables
    5. Feature selection
    6. Feature scaling
    7. 80:20 train-test split (for the regression task)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Path is relative so the project works both locally and on Streamlit Cloud
DATASET_PATH = "dataset/customers.csv"

# Features used for CLV regression (must match the Streamlit input form)
REGRESSION_FEATURES = [
    "Age",
    "Total_Orders",
    "Average_Order_Value",
    "Purchase_Frequency",
    "Days_Since_Last_Purchase",
    "Customer_Tenure",
]
REGRESSION_TARGET = "CLV"

# Features used for K-Means customer segmentation
SEGMENTATION_FEATURES = [
    "Total_Spent",
    "Total_Orders",
    "Days_Since_Last_Purchase",
    "Customer_Tenure",
    "Income",
]


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the raw customers dataset from disk."""
    df = pd.read_csv(path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
        - Numeric columns  -> median imputation
        - Categorical cols -> mode imputation
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate customer records."""
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    df.attrs["duplicates_removed"] = before - after
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns (Education, Marital_Status)."""
    df = df.copy()
    encoders = {}
    for col in ["Education", "Marital_Status"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_Encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def cap_outliers(df: pd.DataFrame, columns: list, upper_quantile: float = 0.99) -> pd.DataFrame:
    """
    Winsorize extreme outliers in the given columns at the specified upper
    quantile. A small number of customers with near-zero tenure (e.g. 1 day
    since enrollment) produce extreme Purchase_Frequency / Average_Order_Value
    values that are mathematical artifacts of dividing by a tiny denominator
    rather than genuine high-value behaviour. Capping them keeps the dataset
    honest while preventing a handful of records from dominating the model.
    """
    df = df.copy()
    for col in columns:
        cap = df[col].quantile(upper_quantile)
        df[col] = np.where(df[col] > cap, cap, df[col])
    return df


def get_cleaning_summary(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    """Return a summary dict describing the cleaning steps performed."""
    missing_before = raw_df.isna().sum()
    summary = {
        "missing_values_before": missing_before[missing_before > 0].to_dict(),
        "total_missing_values": int(raw_df.isna().sum().sum()),
        "duplicates_removed": int(cleaned_df.attrs.get("duplicates_removed", 0)),
        "rows_before": int(raw_df.shape[0]),
        "rows_after": int(cleaned_df.shape[0]),
        "encoded_columns": ["Education", "Marital_Status"],
        "outlier_capped_columns": ["Purchase_Frequency", "Average_Order_Value"],
    }
    return summary


def full_preprocessing_pipeline(path: str = DATASET_PATH):
    """
    Run the complete preprocessing pipeline and return a dictionary
    containing every artifact needed by the segmentation and regression
    modules, plus a summary for the Streamlit dashboard.
    """
    raw_df = load_data(path)
    df = handle_missing_values(raw_df)
    df = remove_duplicates(df)
    df, encoders = encode_categorical(df)
    df = cap_outliers(df, ["Purchase_Frequency", "Average_Order_Value"])

    summary = get_cleaning_summary(raw_df, df)

    return {
        "raw_df": raw_df,
        "clean_df": df,
        "encoders": encoders,
        "summary": summary,
    }


def get_regression_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Feature selection + feature scaling + 80:20 train-test split
    for the CLV regression task.
    """
    X = df[REGRESSION_FEATURES].copy()
    y = df[REGRESSION_TARGET].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler


def get_segmentation_features(df: pd.DataFrame):
    """
    Feature selection + feature scaling for K-Means segmentation.
    """
    X = df[SEGMENTATION_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler
