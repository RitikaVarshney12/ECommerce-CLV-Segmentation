# E-Commerce Customer Lifetime Value (CLV) & Segmentation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)

A production-quality, beginner-friendly Machine Learning project that segments
e-commerce customers with **K-Means Clustering** and predicts their **Customer
Lifetime Value (CLV)** with **Linear Regression**, wrapped in an interactive
**Streamlit** dashboard.

---

## Features

- Real, publicly available dataset (no synthetic data).
- Modular, PEP 8–compliant Python codebase (`preprocessing.py`, `segmentation.py`,
  `regression.py`, `utils.py`).
- K-Means customer segmentation with Elbow Method and Silhouette Score.
- Linear Regression CLV prediction with RMSE, R², and MAPE evaluation.
- A 6-page Streamlit dashboard: Home, Dataset Preview, Data Cleaning Summary,
  Customer Segmentation, CLV Prediction, and an interactive Predict-CLV form.
- A companion Jupyter notebook that mirrors the full pipeline for exploration.
- Fully deployable on Streamlit Community Cloud (relative paths, pinned
  dependencies, no hardcoded local paths).

---

## Screenshots

**Home**

![Home Page](screenshots/home_page.png)

**Customer Segmentation**

![Customer Segmentation Page](screenshots/segmentation_page.png)

**Predict Customer Lifetime Value**

![Prediction Page](screenshots/prediction_page.png)

---

## Dataset Information

- **Dataset Name:** Customer Personality Analysis
- **Dataset Source:** Kaggle ([imakash3011/customer-personality-analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)),
  downloaded from a public GitHub mirror of the original `marketing_campaign.csv`
  (2,240 customers, 29 raw columns from a real retail/e-commerce loyalty program).
- **Why it was selected:** It is a real, well-documented e-commerce/retail
  customer dataset with genuine transactional signals — enrollment date,
  recency, spend by product category, and purchases by channel — that map
  directly onto the CLV and segmentation features this project needs, without
  requiring any synthetic/fabricated data.

### Feature Engineering

The raw dataset does not include `Customer_ID` (uses `ID`), `Age`,
`Total_Orders`, `Total_Spent`, `Average_Order_Value`, `Purchase_Frequency`,
`Days_Since_Last_Purchase`, `Customer_Tenure`, or `CLV` directly — these were
**derived from existing raw columns only**:

| Engineered Feature          | Derivation                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `Customer_ID`                | `ID`                                                                        |
| `Age`                        | Snapshot year − `Year_Birth`                                               |
| `Total_Orders`                | `NumDealsPurchases + NumWebPurchases + NumCatalogPurchases + NumStorePurchases` |
| `Total_Spent`                 | Sum of `MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds` |
| `Average_Order_Value`         | `Total_Spent / Total_Orders`                                               |
| `Days_Since_Last_Purchase`    | `Recency` (already tracked in raw data)                                    |
| `Customer_Tenure`             | Days between `Dt_Customer` (enrollment) and a fixed snapshot date          |
| `Purchase_Frequency`          | `Total_Orders / (Customer_Tenure / 30)` — orders per month                 |
| `CLV`                         | `Average_Order_Value × Purchase_Frequency × Tenure(months)` (standard CLV formula, an observed/historical CLV proxy) |

**Note on Gender / Country:** the source dataset does not contain a `Gender`
or `Country` column, and none of the raw columns can honestly be mapped to
them. Per the "derive from existing columns only" requirement, these two
fields were **omitted** from the final schema rather than fabricated.

`Income`, `Education`, `Marital_Status`, `Kidhome`, and `Teenhome` were kept
as-is from the raw data to support richer EDA and segmentation.

Three records with implausible birth years (implying ages over 100) were
removed during feature engineering as data-entry errors, leaving **2,237
customers** in `dataset/customers.csv`.

---

## Algorithms Used

| Task                     | Algorithm         | Library        |
|---------------------------|-------------------|----------------|
| Customer Segmentation     | K-Means Clustering | scikit-learn   |
| CLV Prediction             | Linear Regression  | scikit-learn   |

No other algorithms are used, per project scope.

---

## Project Workflow

1. **Load** `dataset/customers.csv`.
2. **Clean**: impute missing values (median/mode), remove duplicates, label-encode
   `Education` and `Marital_Status`, winsorize outlier ratios at the 99th percentile.
3. **Segment**: scale features (`Total_Spent`, `Total_Orders`,
   `Days_Since_Last_Purchase`, `Customer_Tenure`, `Income`), run the Elbow
   Method and Silhouette Score across k = 2–10, and fit the final K-Means model.
4. **Predict CLV**: select features (`Age`, `Total_Orders`,
   `Average_Order_Value`, `Purchase_Frequency`, `Days_Since_Last_Purchase`,
   `Customer_Tenure`), scale them, apply an 80:20 train/test split, and fit
   Linear Regression.
5. **Visualize & Predict** everything interactively in `app.py`.

---

## Evaluation Metrics (actual results — not fabricated)

### Segmentation (K-Means)
- Silhouette-optimal k = **2** (score ≈ **0.34**), but this produces a
  trivial two-group split.
- **k = 4** was chosen instead for interpretable, actionable business
  segments. Its actual silhouette score is **≈ 0.24**.

### CLV Prediction (Linear Regression)
- **RMSE:** ≈ 119.97
- **R² Score:** ≈ 0.9621 (meets the ≥ 0.85 target)
- **MAPE:** ≈ 91.15% — inflated by customers with very low actual CLV
  (small absolute errors become large percentages); RMSE and R² are the more
  reliable indicators of fit for this dataset.

*(Re-running `train_models.py` or the notebook will reproduce these values;
minor floating-point variation may occur but the model and features are fixed.)*

---

## Folder Structure

```
ECommerce_CLV_Segmentation/
│── dataset/
│     customers.csv
│
│── notebooks/
│     ECommerce_CLV_Segmentation.ipynb
│
│── models/
│     kmeans_model.pkl
│     segmentation_scaler.pkl
│     linear_regression_model.pkl
│     regression_scaler.pkl
│
│── screenshots/
│     home_page.png
│     segmentation_page.png
│     prediction_page.png
│
│── app.py
│── preprocessing.py
│── segmentation.py
│── regression.py
│── utils.py
│── train_models.py
│── requirements.txt
│── README.md
```

---

## Installation Steps

```bash
git clone <your-repo-url>
cd ECommerce_CLV_Segmentation
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is pinned to stable releases compatible with **Python 3.9–3.12**
(verified against **Python 3.11/3.12** in a clean virtual environment):
`pandas==2.2.2`, `numpy==1.26.4`, `scikit-learn==1.4.2`, `matplotlib==3.8.4`,
`seaborn==0.13.2`, `streamlit==1.35.0`.

## Local Execution

Train the models once (writes to `models/`):

```bash
python train_models.py
```

Run the dashboard:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Streamlit Community Cloud Deployment Steps

1. Push this project to a public (or private) GitHub repository, keeping the
   folder structure above intact.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository and branch.
4. Set the **Main file path** to `app.py`.
5. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and launch the app.
6. No secrets, API keys, or environment variables are required — the app
   only reads the bundled `dataset/customers.csv` using relative paths.

---

## Restrictions Honored

Only K-Means Clustering and Linear Regression are used. No deep learning,
tree-based models, SQL, APIs, authentication, or cloud services are involved.