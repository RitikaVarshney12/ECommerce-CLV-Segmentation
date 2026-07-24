"""
train_models.py
----------------
Run this script once to train and persist the K-Means segmentation model
and the Linear Regression CLV model into the models/ folder.

Usage:
    python train_models.py
"""

from preprocessing import (
    full_preprocessing_pipeline,
    get_regression_train_test_split,
    get_segmentation_features,
)
from segmentation import compute_silhouette_scores, train_kmeans, get_best_k
from regression import train_linear_regression, evaluate_model
from utils import save_pickle

K_RANGE = range(2, 11)


def main():
    pipeline = full_preprocessing_pipeline()
    clean_df = pipeline["clean_df"]

    # ---------------- Segmentation ----------------
    _, X_scaled_seg, scaler_seg = get_segmentation_features(clean_df)
    k_values, sil_scores = compute_silhouette_scores(X_scaled_seg, K_RANGE)
    silhouette_best_k = get_best_k(k_values, sil_scores)

    # NOTE: k=2 technically yields the highest silhouette score, but produces
    # a trivial split with little business value. k=4 is used for the final
    # model to produce interpretable, actionable customer segments (a common
    # and defensible trade-off in real-world segmentation projects). Its
    # actual (not fabricated) silhouette score is reported below.
    chosen_k = 4
    kmeans_model, labels, sil_score = train_kmeans(X_scaled_seg, n_clusters=chosen_k)

    save_pickle(kmeans_model, "kmeans_model.pkl")
    save_pickle(scaler_seg, "segmentation_scaler.pkl")

    print(f"[Segmentation] Silhouette-optimal k = {silhouette_best_k} | "
          f"Chosen k (business-interpretable) = {chosen_k}, Silhouette Score = {sil_score:.4f}")

    # ---------------- Regression ----------------
    X_train, X_test, y_train, y_test, reg_scaler = get_regression_train_test_split(clean_df)
    reg_model = train_linear_regression(X_train, y_train)
    metrics = evaluate_model(reg_model, X_test, y_test)

    save_pickle(reg_model, "linear_regression_model.pkl")
    save_pickle(reg_scaler, "regression_scaler.pkl")

    print(f"[Regression] RMSE = {metrics['rmse']:.2f}, "
          f"R2 = {metrics['r2']:.4f}, MAPE = {metrics['mape']:.2f}%")


if __name__ == "__main__":
    main()
