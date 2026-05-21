import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_dataset(X_path, y_path, replace_question_marks=False):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()

    if replace_question_marks:
        X = X.replace("?", pd.NA)

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(include=["object", "str"]).columns

    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def main():
    datasets = [
        ("german", "data/german_credit_X.csv", "data/german_credit_y.csv", False),
        ("adult",  "data/adult_X.csv",          "data/adult_y.csv",          True),
    ]

    for name, X_path, y_path, qmarks in datasets:
        X_train, X_test, y_train, y_test = preprocess_dataset(X_path, y_path, qmarks)

        X_train.to_csv(f"data/{name}_X_train.csv", index=False)
        X_test.to_csv(f"data/{name}_X_test.csv",   index=False)
        y_train.to_csv(f"data/{name}_y_train.csv", index=False)
        y_test.to_csv(f"data/{name}_y_test.csv",   index=False)

        print(f"{name}_X_train: {X_train.shape}")
        print(f"{name}_X_test:  {X_test.shape}")
        print(f"{name}_y_train: {y_train.shape}")
        print(f"{name}_y_test:  {y_test.shape}")


if __name__ == "__main__":
    main()
