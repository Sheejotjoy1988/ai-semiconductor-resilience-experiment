import os
import json
import subprocess
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "semiconductor_supply_chain.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def gpu_available():
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except Exception:
        return False


def calculate_metrics(name, y_true, y_pred, y_prob):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_prob
        )
    }


print("=" * 78)
print("AI-BASED SEMICONDUCTOR SUPPLY-CHAIN RESILIENCE EXPERIMENT")
print("=" * 78)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Observed disruption rate: {df['disrupted'].mean():.2%}")

X = df.drop(
    columns=[
        "supplier_id",
        "disrupted"
    ]
)

y = df["disrupted"]


# ---------------------------------------------------------
# ONE-HOT ENCODING
# ---------------------------------------------------------

X = pd.get_dummies(
    X,
    columns=[
        "country",
        "product_type"
    ],
    drop_first=False,
    dtype=int
)

feature_names = list(X.columns)

print(
    f"Number of model features after encoding: "
    f"{len(feature_names)}"
)


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


# ---------------------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------------------

print("\n" + "=" * 78)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 78)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

lr.fit(
    X_train_scaled,
    y_train
)

lr_pred = lr.predict(
    X_test_scaled
)

lr_prob = lr.predict_proba(
    X_test_scaled
)[:, 1]

lr_metrics = calculate_metrics(
    "Logistic Regression",
    y_test,
    lr_pred,
    lr_prob
)

print(lr_metrics)


# ---------------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------------

print("\n" + "=" * 78)
print("MODEL 2: RANDOM FOREST")
print("=" * 78)

rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=14,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(
    X_test
)

rf_prob = rf.predict_proba(
    X_test
)[:, 1]

rf_metrics = calculate_metrics(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)

print(rf_metrics)


# ---------------------------------------------------------
# XGBOOST
# ---------------------------------------------------------

print("\n" + "=" * 78)
print("MODEL 3: XGBOOST")
print("=" * 78)

use_gpu = gpu_available()

print("NVIDIA GPU detected:", use_gpu)

xgb_params = dict(
    n_estimators=450,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

if use_gpu:
    xgb_params["device"] = "cuda"
else:
    xgb_params["device"] = "cpu"

xgb_model = xgb.XGBClassifier(
    **xgb_params
)

try:
    xgb_model.fit(
        X_train,
        y_train
    )

except Exception as exc:
    print("\nGPU training failed.")
    print("Retrying XGBoost using CPU.")
    print(f"Reason: {exc}")

    xgb_params["device"] = "cpu"

    xgb_model = xgb.XGBClassifier(
        **xgb_params
    )

    xgb_model.fit(
        X_train,
        y_train
    )


xgb_pred = xgb_model.predict(
    X_test
)

xgb_prob = xgb_model.predict_proba(
    X_test
)[:, 1]

xgb_metrics = calculate_metrics(
    "XGBoost",
    y_test,
    xgb_pred,
    xgb_prob
)

print(xgb_metrics)


# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------

comparison = pd.DataFrame(
    [
        lr_metrics,
        rf_metrics,
        xgb_metrics
    ]
)

comparison = comparison.sort_values(
    "roc_auc",
    ascending=False
).reset_index(drop=True)

comparison.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    ),
    index=False
)

print("\n" + "=" * 78)
print("MODEL COMPARISON")
print("=" * 78)

print(
    comparison.to_string(
        index=False
    )
)

best_model = comparison.iloc[0]["model"]

print(
    f"\nBEST MODEL BY ROC-AUC: "
    f"{best_model}"
)


# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------

report = classification_report(
    y_test,
    xgb_pred,
    digits=4
)

with open(
    os.path.join(
        RESULTS_DIR,
        "xgboost_classification_report.txt"
    ),
    "w"
) as file:
    file.write(report)

print("\nXGBoost Classification Report:")
print(report)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    xgb_pred,
    values_format="d"
)

plt.title(
    "XGBoost Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# ROC CURVES
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    lr_prob,
    name="Logistic Regression",
    ax=ax
)

RocCurveDisplay.from_predictions(
    y_test,
    rf_prob,
    name="Random Forest",
    ax=ax
)

RocCurveDisplay.from_predictions(
    y_test,
    xgb_prob,
    name="XGBoost",
    ax=ax
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

ax.set_title(
    "ROC Curves: Semiconductor Disruption Prediction"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# XGBOOST FEATURE IMPORTANCE
# ---------------------------------------------------------

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": xgb_model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    os.path.join(
        RESULTS_DIR,
        "feature_importance.csv"
    ),
    index=False
)

top_features = importance.head(15)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Supply-Chain Risk Factor"
)

plt.title(
    "Top Drivers of Semiconductor Supply-Chain Disruption"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "feature_importance.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# NATIVE XGBOOST SHAP CONTRIBUTIONS
# ---------------------------------------------------------

print("\nCalculating explainability contributions...")

sample_size = min(
    2000,
    len(X_test)
)

X_explain = X_test.sample(
    sample_size,
    random_state=RANDOM_STATE
)

booster = xgb_model.get_booster()

dmatrix = xgb.DMatrix(
    X_explain,
    feature_names=feature_names
)

shap_values = booster.predict(
    dmatrix,
    pred_contribs=True
)

shap_features = shap_values[:, :-1]

mean_abs_shap = np.abs(
    shap_features
).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": mean_abs_shap
})

shap_df = shap_df.sort_values(
    "mean_absolute_shap",
    ascending=False
)

shap_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "shap_feature_importance.csv"
    ),
    index=False
)

top_shap = shap_df.head(15)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_shap["feature"][::-1],
    top_shap["mean_absolute_shap"][::-1]
)

plt.xlabel(
    "Mean Absolute SHAP Contribution"
)

plt.ylabel(
    "Risk Factor"
)

plt.title(
    "Explainable AI: Main Drivers of Disruption Risk"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "shap_importance.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# SUPPLIER RISK SCORING
# ---------------------------------------------------------

all_probabilities = xgb_model.predict_proba(
    X
)[:, 1]

risk_results = df[
    [
        "supplier_id",
        "country",
        "product_type"
    ]
].copy()

risk_results[
    "predicted_disruption_probability"
] = all_probabilities


def risk_category(probability):
    if probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Medium"
    else:
        return "High"


risk_results[
    "risk_category"
] = risk_results[
    "predicted_disruption_probability"
].apply(risk_category)


def resilience_action(category):
    if category == "High":
        return (
            "Diversify suppliers; increase safety stock; "
            "activate alternate sourcing; intensify monitoring"
        )

    if category == "Medium":
        return (
            "Increase monitoring; review inventory buffer; "
            "evaluate secondary suppliers"
        )

    return (
        "Maintain current controls and periodic monitoring"
    )


risk_results[
    "recommended_action"
] = risk_results[
    "risk_category"
].apply(resilience_action)

risk_results = risk_results.sort_values(
    "predicted_disruption_probability",
    ascending=False
)

risk_results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "supplier_risk_predictions.csv"
    ),
    index=False
)


# ---------------------------------------------------------
# COUNTRY RISK SUMMARY
# ---------------------------------------------------------

country_risk = (
    risk_results
    .groupby("country")
    .agg(
        mean_disruption_probability=(
            "predicted_disruption_probability",
            "mean"
        ),
        high_risk_suppliers=(
            "risk_category",
            lambda x: (x == "High").sum()
        ),
        total_suppliers=(
            "supplier_id",
            "count"
        )
    )
    .reset_index()
)

country_risk[
    "high_risk_percentage"
] = (
    country_risk["high_risk_suppliers"]
    / country_risk["total_suppliers"]
    * 100
)

country_risk = country_risk.sort_values(
    "mean_disruption_probability",
    ascending=False
)

country_risk.to_csv(
    os.path.join(
        RESULTS_DIR,
        "country_risk_summary.csv"
    ),
    index=False
)


# ---------------------------------------------------------
# RISK DISTRIBUTION
# ---------------------------------------------------------

risk_counts = (
    risk_results["risk_category"]
    .value_counts()
    .reindex(
        ["Low", "Medium", "High"],
        fill_value=0
    )
)

plt.figure(
    figsize=(8, 6)
)

risk_counts.plot(
    kind="bar"
)

plt.ylabel(
    "Number of Suppliers"
)

plt.xlabel(
    "Predicted Risk Category"
)

plt.title(
    "Semiconductor Supplier Disruption Risk Distribution"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "risk_distribution.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

joblib.dump(
    xgb_model,
    os.path.join(
        MODELS_DIR,
        "xgboost_semiconductor_risk_model.pkl"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        MODELS_DIR,
        "logistic_scaler.pkl"
    )
)


# ---------------------------------------------------------
# EXPERIMENT SUMMARY
# ---------------------------------------------------------

summary = {
    "dataset_rows": int(len(df)),
    "dataset_columns": int(len(df.columns)),
    "disruption_rate": float(
        df["disrupted"].mean()
    ),
    "best_model": str(best_model),
    "logistic_regression": lr_metrics,
    "random_forest": rf_metrics,
    "xgboost": xgb_metrics,
    "high_risk_suppliers": int(
        (risk_results["risk_category"] == "High").sum()
    ),
    "medium_risk_suppliers": int(
        (risk_results["risk_category"] == "Medium").sum()
    ),
    "low_risk_suppliers": int(
        (risk_results["risk_category"] == "Low").sum()
    )
}

with open(
    os.path.join(
        RESULTS_DIR,
        "experiment_summary.json"
    ),
    "w"
) as file:
    json.dump(
        summary,
        file,
        indent=4
    )


print("\n" + "=" * 78)
print("TOP EXPLAINABLE-AI RISK DRIVERS")
print("=" * 78)

print(
    shap_df.head(10).to_string(
        index=False
    )
)

print("\n" + "=" * 78)
print("COUNTRY RISK SUMMARY")
print("=" * 78)

print(
    country_risk.to_string(
        index=False
    )
)

print("\n" + "=" * 78)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 78)

print("\nGenerated outputs:")

for filename in sorted(
    os.listdir(RESULTS_DIR)
):
    print(
        "  results/" + filename
    )

print("\nSaved model:")
print(
    "  models/xgboost_semiconductor_risk_model.pkl"
)
