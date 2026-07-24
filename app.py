"""
app.py
------
Streamlit dashboard for the E-Commerce Customer Lifetime Value (CLV)
& Segmentation project.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from preprocessing import (
    full_preprocessing_pipeline,
    get_regression_train_test_split,
    get_segmentation_features,
    REGRESSION_FEATURES,
    SEGMENTATION_FEATURES,
)
from segmentation import (
    compute_elbow_curve,
    compute_silhouette_scores,
    train_kmeans,
    assign_clusters,
    cluster_summary,
)
from regression import train_linear_regression, evaluate_model, predict_clv
from utils import (
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_cluster_scatter,
    plot_cluster_counts,
    plot_actual_vs_predicted,
    plot_residuals,
    save_pickle,
    load_pickle,
)

st.set_page_config(
    page_title="E-Commerce CLV & Segmentation",
    page_icon="🛒",
    layout="wide",
)

K_RANGE = range(2, 11)
CHOSEN_K = 4  # business-interpretable choice, see README / notebook for rationale


# --------------------------------------------------------------------------
# Cached data & model pipeline (recomputed once per session, then cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preprocessing dataset...")
def load_pipeline():
    pipeline = full_preprocessing_pipeline()
    return pipeline


@st.cache_resource(show_spinner="Training / loading segmentation model...")
def get_segmentation_artifacts(_clean_df):
    X_raw, X_scaled, scaler = get_segmentation_features(_clean_df)
    k_values, inertias = compute_elbow_curve(X_scaled, K_RANGE)
    _, sil_scores = compute_silhouette_scores(X_scaled, K_RANGE)
    kmeans_model, labels, sil_score = train_kmeans(X_scaled, n_clusters=CHOSEN_K)
    return {
        "X_raw": X_raw,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "k_values": k_values,
        "inertias": inertias,
        "sil_scores": sil_scores,
        "kmeans_model": kmeans_model,
        "labels": labels,
        "sil_score": sil_score,
    }


@st.cache_resource(show_spinner="Training / loading regression model...")
def get_regression_artifacts(_clean_df):
    X_train, X_test, y_train, y_test, scaler = get_regression_train_test_split(_clean_df)
    model = train_linear_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
    }


pipeline = load_pipeline()
raw_df = pipeline["raw_df"]
clean_df = pipeline["clean_df"]
summary = pipeline["summary"]

seg = get_segmentation_artifacts(clean_df)
reg = get_regression_artifacts(clean_df)

df_clustered = assign_clusters(clean_df, seg["labels"])

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("🛒 CLV & Segmentation")
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Dataset Preview",
        "Data Cleaning Summary",
        "Customer Segmentation",
        "CLV Prediction",
        "Predict Customer Lifetime Value",
    ],
)

# ==========================================================================
# 1. HOME
# ==========================================================================
if page == "Home":
    st.title("E-Commerce Customer Lifetime Value (CLV) & Segmentation")

    st.subheader("Project Overview")
    st.write(
        "This application segments e-commerce customers into meaningful "
        "groups using K-Means Clustering, and predicts each customer's "
        "Lifetime Value (CLV) using Linear Regression, all wrapped in an "
        "interactive Streamlit dashboard."
    )

    st.subheader("Objective")
    st.markdown(
        """
        1. Segment customers using **K-Means Clustering**.
        2. Predict **Customer Lifetime Value (CLV)** using **Linear Regression**.
        3. Provide an interactive dashboard for visualization and prediction.
        """
    )

    st.subheader("Tech Stack")
    col1, col2, col3 = st.columns(3)
    col1.markdown("- Python\n- Pandas\n- NumPy")
    col2.markdown("- Scikit-Learn\n- Matplotlib")
    col3.markdown("- Seaborn\n- Streamlit")

    st.info(
        "Use the sidebar to explore the dataset, review the cleaning "
        "process, inspect segmentation & prediction results, or predict "
        "the CLV of a new customer."
    )

# ==========================================================================
# 2. DATASET PREVIEW
# ==========================================================================
elif page == "Dataset Preview":
    st.title("Dataset Preview")

    col1, col2 = st.columns(2)
    col1.metric("Rows", raw_df.shape[0])
    col2.metric("Columns", raw_df.shape[1])

    st.subheader("First Five Rows")
    st.dataframe(raw_df.head())

    st.subheader("Data Types")
    dtypes_df = raw_df.dtypes.astype(str).reset_index()
    dtypes_df.columns = ["Column", "Data Type"]
    st.dataframe(dtypes_df, use_container_width=True)

# ==========================================================================
# 3. DATA CLEANING SUMMARY
# ==========================================================================
elif page == "Data Cleaning Summary":
    st.title("Data Cleaning Summary")

    st.subheader("Missing Values (Before Cleaning)")
    if summary["missing_values_before"]:
        missing_df = pd.DataFrame(
            list(summary["missing_values_before"].items()),
            columns=["Column", "Missing Count"],
        )
        st.dataframe(missing_df, use_container_width=True)
        st.caption("Missing numeric values were imputed with the column median; "
                   "missing categorical values were imputed with the column mode.")
    else:
        st.success("No missing values were found in the raw dataset.")

    st.subheader("Duplicate Removal")
    st.metric("Duplicate Records Removed", summary["duplicates_removed"])
    st.write(f"Rows before cleaning: **{summary['rows_before']}** → "
             f"Rows after cleaning: **{summary['rows_after']}**")

    st.subheader("Encoding Summary")
    st.write(f"Categorical columns label-encoded: **{', '.join(summary['encoded_columns'])}**")

    st.subheader("Outlier Handling")
    st.write(
        f"Columns winsorized at the 99th percentile: "
        f"**{', '.join(summary['outlier_capped_columns'])}** "
        "(a small number of customers with near-zero tenure produced extreme "
        "ratio values that were mathematical artifacts rather than genuine behavior)."
    )

# ==========================================================================
# 4. CUSTOMER SEGMENTATION
# ==========================================================================
elif page == "Customer Segmentation":
    st.title("Customer Segmentation (K-Means Clustering)")

    st.markdown(
        f"Segmentation features used: `{', '.join(SEGMENTATION_FEATURES)}`"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Elbow Curve")
        st.pyplot(plot_elbow_curve(seg["k_values"], seg["inertias"]))
    with col2:
        st.subheader("Silhouette Score by k")
        st.pyplot(plot_silhouette_scores(seg["k_values"], seg["sil_scores"]))

    st.info(
        f"k=2 technically yields the highest silhouette score, but produces a "
        f"trivial two-group split. **k={CHOSEN_K}** was chosen for interpretable, "
        f"actionable business segments. Its actual silhouette score is "
        f"**{seg['sil_score']:.4f}**."
    )

    st.subheader("Cluster Scatter Plot")
    scatter_x = st.selectbox("X-axis feature", SEGMENTATION_FEATURES, index=0)
    scatter_y = st.selectbox("Y-axis feature", SEGMENTATION_FEATURES, index=1)
    st.pyplot(plot_cluster_scatter(df_clustered, scatter_x, scatter_y))

    st.subheader("Customer Count per Cluster")
    st.pyplot(plot_cluster_counts(df_clustered))

    st.subheader("Cluster Profile Summary")
    st.dataframe(cluster_summary(df_clustered, SEGMENTATION_FEATURES), use_container_width=True)

# ==========================================================================
# 5. CLV PREDICTION (model evaluation)
# ==========================================================================
elif page == "CLV Prediction":
    st.title("CLV Prediction (Linear Regression)")

    st.markdown(f"Regression features used: `{', '.join(REGRESSION_FEATURES)}`")

    metrics = reg["metrics"]
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{metrics['rmse']:.2f}")
    col2.metric("R² Score", f"{metrics['r2']:.4f}")
    col3.metric("MAPE", f"{metrics['mape']:.2f}%")

    st.caption(
        "MAPE is inflated by customers with very low actual CLV, since small "
        "percentage errors on small values produce large percentages. RMSE and "
        "R² are more reliable indicators of overall model fit for this dataset."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual vs Predicted CLV")
        st.pyplot(plot_actual_vs_predicted(np.array(metrics["y_test"]), metrics["y_pred"]))
    with col2:
        st.subheader("Residual Plot")
        st.pyplot(plot_residuals(metrics["y_pred"], np.array(metrics["residuals"])))

# ==========================================================================
# 6. PREDICT CUSTOMER LIFETIME VALUE (interactive form)
# ==========================================================================
elif page == "Predict Customer Lifetime Value":
    st.title("Predict Customer Lifetime Value")
    st.write("Enter customer details below to estimate their Customer Lifetime Value.")

    with st.form("clv_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            total_orders = st.number_input("Total Orders", min_value=0, max_value=200, value=15)
            avg_order_value = st.number_input(
                "Average Order Value ($)", min_value=0.0, max_value=5000.0, value=45.0, step=1.0
            )
        with col2:
            purchase_frequency = st.number_input(
                "Purchase Frequency (orders/month)", min_value=0.0, max_value=50.0, value=1.5, step=0.1
            )
            days_since_last_purchase = st.number_input(
                "Days Since Last Purchase", min_value=0, max_value=365, value=30
            )
            customer_tenure = st.number_input(
                "Customer Tenure (days)", min_value=1, max_value=3000, value=365
            )

        submitted = st.form_submit_button("Predict CLV")

    if submitted:
        input_features = [
            age,
            total_orders,
            avg_order_value,
            purchase_frequency,
            days_since_last_purchase,
            customer_tenure,
        ]
        prediction = predict_clv(reg["model"], reg["scaler"], input_features)
        st.success(f"### Predicted Customer Lifetime Value: **${prediction:,.2f}**")
