import pickle
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score


def load_labels(name):
    y_train = pd.read_csv(f"data/{name}_y_train.csv").squeeze()
    y_test  = pd.read_csv(f"data/{name}_y_test.csv").squeeze()

    # normalize string labels (strip trailing dots/spaces from UCI test-set variants)
    if not pd.api.types.is_numeric_dtype(y_train):
        y_train = y_train.str.strip().str.rstrip(".")
        y_test  = y_test.str.strip().str.rstrip(".")

    # remap to 0-based integers
    classes = sorted(y_train.unique())
    remap = {v: i for i, v in enumerate(classes)}
    y_train = y_train.map(remap).astype(int)
    y_test  = y_test.map(remap).astype(int)

    return y_train, y_test


def train_dataset(name):
    X_train = pd.read_csv(f"data/{name}_X_train.csv")
    X_test  = pd.read_csv(f"data/{name}_X_test.csv")
    y_train, y_test = load_labels(name)

    param_grid = {
        "max_depth":     [3, 5, 7],
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
    }
    base = XGBClassifier(random_state=42, eval_metric="logloss")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    model = XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    with open(f"data/{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"\n=== {name} ===")
    print(f"  best params : {best_params}")
    print(f"  accuracy    : {acc:.4f}")
    print(f"  AUC         : {auc:.4f}")
    print(f"  saved to    : data/{name}_model.pkl")


def extract_shap(name):
    with open(f"data/{name}_model.pkl", "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv(f"data/{name}_X_test.csv")

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_test)

    # TreeExplainer returns (n_samples, n_features) for binary XGBoost
    # but may return a list of arrays for some versions — take index 1 if so
    if isinstance(raw, list):
        values = raw[1]
    else:
        values = raw

    shap_df = pd.DataFrame(values, columns=X_test.columns)
    shap_df.to_csv(f"data/{name}_shap_values.csv", index=False)

    print(f"\n=== {name} SHAP ===")
    print(f"  shape : {shap_df.shape}")
    print(shap_df.head(3).to_string())


def main():
    for name in ["german", "adult"]:
        train_dataset(name)

    print("\n--- SHAP extraction ---")
    for name in ["german", "adult"]:
        extract_shap(name)

    print("\nSHAP extraction complete")


if __name__ == "__main__":
    main()
