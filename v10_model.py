"""
DATATHON 2026 ROUND 1 - REVENUE & COGS FORECASTING
Model V10: Logistic Regression with Feature Importance Analysis

Cach su dung:
1. Dat file v10_model.py cung thu muc voi cac file data (sales.csv, sample_submission.csv)
2. Chay: python v10_model.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import os
import warnings

warnings.filterwarnings("ignore")

# 1. XAC DINH DUONG DAN
# Lay thu muc chua script (noi file code duoc dat)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cac thu muc can tim file du lieu (theo thu tu uu tien)
possible_paths = [
    script_dir,  # 1. Thu muc chua script (uutien cao nhat)
    os.getcwd(),  # 2. Thu muc hien tai
    os.path.join(os.getcwd(), "data"),  # 3. Thu muc con 'data'
]

workdir = None
for path in possible_paths:
    sales_file = os.path.join(path, "sales.csv")
    if os.path.exists(sales_file):
        workdir = path
        break

if workdir is None:
    print("KHONG TIM THAY DU LIEU!")
    exit()

print(f"Thu muc lam viec: {workdir}")

# 2. LOAD DATA
print("\n1. LOADING DATA...")

sales = pd.read_csv(os.path.join(workdir, "sales.csv"), parse_dates=["Date"])
sample = pd.read_csv(os.path.join(workdir, "sample_submission.csv"))

print(f"Sales data: {sales.shape}")
print(f"Date range: {sales['Date'].min()} to {sales['Date'].max()}")

# 3. DATA PREPARATION
print("\n2. PREPARING DATA...")

sales["month"] = sales["Date"].dt.month
sales["dow"] = sales["Date"].dt.dayofweek
sales["day"] = sales["Date"].dt.day
sales["year"] = sales["Date"].dt.year

# Bien muc tieu: is_high (phan loai ngay doanh thu cao/thap)
sales["is_high"] = (
    sales.groupby("month")["Revenue"].transform(lambda x: x > x.median()).astype(int)
)

print(f"Target: High={sales['is_high'].sum()}, Low={(sales['is_high'] == 0).sum()}")

# 4. FEATURE ENGINEERING
print("\n3. CREATING FEATURES...")


def create_features(df):
    df = df.copy()
    df["month"] = df["Date"].dt.month
    df["dow"] = df["Date"].dt.dayofweek
    df["day"] = df["Date"].dt.day
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_month_start"] = (df["day"] <= 5).astype(int)
    df["is_month_end"] = (df["day"] >= 26).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    for lag in [7, 14, 21]:
        df[f"lag_{lag}"] = df["Revenue"].shift(lag)
    for w in [7, 14]:
        df[f"rma_{w}"] = df["Revenue"].shift(1).rolling(w, min_periods=1).mean()
    return df


sales = create_features(sales)

feat_cols = [
    c
    for c in sales.columns
    if c not in ["Date", "Revenue", "COGS", "month", "dow", "year", "day", "is_high"]
]

print(f"Features ({len(feat_cols)}): {feat_cols}")

# 5. TRAIN MODEL
print("\n4. TRAINING MODEL...")

train_data = sales[sales["year"] == 2022].dropna(subset=feat_cols)

X_train = train_data[feat_cols].values
y_train = train_data["is_high"].values

print(f"Training data: {len(train_data)} days")

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("Training completed!")

# 6. FEATURE IMPORTANCE ANALYSIS (Dung cho bao cao ky thuat)
print("\n5. FEATURE IMPORTANCE ANALYSIS...")

# Lay he so cua model (chi co 1 class, lay hang dau)
coef = model.coef_[0]

# Tao bang feature importance
importance_df = pd.DataFrame(
    {"Feature": feat_cols, "Coefficient": coef, "Abs_Coefficient": np.abs(coef)}
).sort_values("Abs_Coefficient", ascending=False)

print("\n=== FEATURE IMPORTANCE (Logistic Regression Coefficients) ===")
print(importance_df.to_string(index=False))

# Giai thich cac feature quan trong nhat
print("\n=== GIAI THICH FEATURE ===")
print("Top 5 features anh huong nhieu nhat:")
for i, row in importance_df.head(5).iterrows():
    direction = "tang" if row["Coefficient"] > 0 else "giam"
    print(
        f"  {row['Feature']}: coefficient={row['Coefficient']:.4f} -> khi tang thi xac suat doanh thu cao {direction}"
    )

# Luu feature importance ra file
importance_df.to_csv(os.path.join(workdir, "feature_importance.csv"), index=False)
print("\nDa luu feature_importance.csv")

# 7. CROSS-VALIDATION (Time Series)
print("\n6. CROSS-VALIDATION (Time Series Split)...")

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

    model_cv = LogisticRegression(max_iter=1000, random_state=42)
    model_cv.fit(X_cv_train, y_cv_train)

    accuracy = model_cv.score(X_cv_val, y_cv_val)
    cv_scores.append(accuracy)
    print(f"  Fold {fold + 1}: Accuracy = {accuracy:.4f}")

print(f"\nCV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
# 8. PREDICT TEST PERIOD
print("\n7. PREDICTING TEST PERIOD...")

test_dates = pd.date_range("2023-01-01", "2024-07-01", freq="D")
print(
    f"Test: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}"
)

history = sales[["Date", "Revenue"]].copy()
proba_high = []

for i, d in enumerate(test_dates):
    temp = pd.concat(
        [history, pd.DataFrame({"Date": [d], "Revenue": [np.nan]})], ignore_index=True
    )
    temp = create_features(temp)
    feat = temp.iloc[-1][feat_cols].values.astype(float)

    if np.any(np.isnan(feat)):
        prob = 0.5
    else:
        prob = model.predict_proba(feat.reshape(1, -1))[0][1]

    proba_high.append(prob)
    history = pd.concat(
        [history, pd.DataFrame({"Date": [d], "Revenue": [prob * 4000000]})],
        ignore_index=True,
    )

    if (i + 1) % 100 == 0:
        print(f"  Predicted {i + 1}/{len(test_dates)} days...")

proba_high = np.array(proba_high)

# 9. CONVERT TO REVENUE
print("\n8. CONVERTING TO REVENUE...")

mean_revenue = sales[sales["year"] == 2022]["Revenue"].mean()
print(f"Mean revenue (2022): {mean_revenue:,.0f}")

revenue_predictions = mean_revenue * (0.8 + 0.4 * proba_high)

print(
    f"Revenue: Min={revenue_predictions.min():,.0f}, Max={revenue_predictions.max():,.0f}, Mean={revenue_predictions.mean():,.0f}"
)
cogs_ratio = (
    sales[sales["year"] == 2022]["COGS"].sum()
    / sales[sales["year"] == 2022]["Revenue"].sum()
)
cogs_predictions = revenue_predictions * cogs_ratio

print(f"COGS ratio: {cogs_ratio:.4f}")

# 10. CALCULATE PERFORMANCE METRICS
print("\n9. CALCULATING PERFORMANCE METRICS...")

# Get actual values from sample_submission
actual = sample["Revenue"].values
predicted = revenue_predictions

# Calculate metrics
mae = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted) ** 2))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100
r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

print("\n=== PERFORMANCE METRICS (vs Sample Submission) ===")
print(f"MAE  (Mean Absolute Error):         {mae:,.2f}")
print(f"RMSE (Root Mean Squared Error):      {rmse:,.2f}")
print(f"MAPE (Mean Absolute % Error):      {mape:.2f}%")
print(f"R2   (R-squared):                    {r2:.4f}")

# Additional metrics
print("\n=== ADDITIONAL METRICS ===")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
print(f"Prediction mean: {predicted.mean():,.2f}")
print(f"Actual mean: {actual.mean():,.2f}")
print(f"Difference: {(predicted.mean() - actual.mean()) / actual.mean() * 100:+.2f}%")

# Save metrics to file
metrics_df = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            "MAPE",
            "R2",
            "Prediction_Mean",
            "Actual_Mean",
            "Difference_%",
        ],
        "Value": [
            mae,
            rmse,
            mape,
            r2,
            predicted.mean(),
            actual.mean(),
            (predicted.mean() - actual.mean()) / actual.mean() * 100,
        ],
    }
)
metrics_df.to_csv(os.path.join(workdir, "metrics.csv"), index=False)
print("\nSaved metrics to metrics.csv")

# 11. SAVE

output = pd.DataFrame(
    {
        "Date": test_dates.strftime("%Y-%m-%d"),
        "Revenue": revenue_predictions.round(2),
        "COGS": cogs_predictions.round(2),
    }
)

output_path = os.path.join(workdir, "final.csv")
output.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
