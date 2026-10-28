import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low = np.percentile(X, 1, axis=0)
        self.high = np.percentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.low, self.high)


def add_features(data):
    data = data.copy()
    data["capital_net"] = data["capital-gain"] - data["capital-loss"]
    age = (data["age"] - 16).replace(0, np.nan)
    data["hours_per_age"] = data["hours-per-week"] / age
    return data


def make_pipeline(data):
    numbers = data.select_dtypes(include="number").columns.tolist()
    categories = data.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outliers", OutlierClipper()),
        ("scaler", StandardScaler())
    ])

    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numbers", numeric_pipeline, numbers),
        ("categories", category_pipeline, categories)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    return model


def evaluate(train_x, test_x, train_y, test_y):
    model = make_pipeline(train_x)
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)

    accuracy = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)

    return model, accuracy, f1


def main():
    Path("results").mkdir(exist_ok=True)

    with open("config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    columns = [
        "age", "workclass", "fnlwgt", "education",
        "education-num", "marital-status", "occupation",
        "relationship", "race", "sex", "capital-gain",
        "capital-loss", "hours-per-week", "native-country", "income"
    ]

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"

    train_data = pd.read_csv(
        url + "adult.data",
        names=columns,
        skipinitialspace=True,
        na_values="?"
    )

    test_data = pd.read_csv(
        url + "adult.test",
        names=columns,
        skiprows=1,
        skipinitialspace=True,
        na_values="?"
    )

    data = pd.concat([train_data, test_data], ignore_index=True)
    data["income"] = data["income"].str.strip().str.rstrip(".")

    raw_csv = data.to_csv(index=False).encode("utf-8")
    data_hash = hashlib.sha256(raw_csv).hexdigest()

    Path("results/data_hash.txt").write_text(data_hash)

    missing = data.isna().sum()
    duplicates = data.duplicated().sum()

    valid_income = data["income"].isin(["<=50K", ">50K"])
    valid_age = data["age"].between(17, 90)
    valid_hours = data["hours-per-week"].between(1, 99)

    features = data.drop(columns="income")

    grouped = data.groupby(
        list(features.columns),
        dropna=False
    )["income"].nunique()

    conflicts = (grouped > 1).sum()

    sample = data.sample(
        n=min(50, len(data)),
        random_state=config["random_state"]
    )

    sample.to_csv("results/label_check.csv", index=False)

    validation = [
        "Lab 4 (71616)",
        "",
        f"Rows: {len(data)}",
        f"Columns: {len(data.columns)}",
        f"Duplicates: {duplicates}",
        f"Conflicting duplicate groups: {conflicts}",
        f"Invalid income values: {(~valid_income).sum()}",
        f"Invalid age values: {(~valid_age).sum()}",
        f"Invalid working hours: {(~valid_hours).sum()}",
        "",
        "Column types:",
        str(data.dtypes),
        "",
        "Missing values:",
        str(missing),
        "",
        "50 random rows saved to label_check.csv"
    ]

    Path("results/validation_report.txt").write_text(
        "\n".join(validation),
        encoding="utf-8"
    )

    x = data.drop(columns="income")
    y = (data["income"] == ">50K").astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )

    base_model, base_accuracy, base_f1 = evaluate(
        x_train, x_test, y_train, y_test
    )

    new_train = add_features(x_train)
    new_test = add_features(x_test)

    final_model, new_accuracy, new_f1 = evaluate(
        new_train, new_test, y_train, y_test
    )

    joblib.dump(
        final_model.named_steps["preprocessor"],
        "results/preprocessor.joblib"
    )

    comparison = [
        "Lab 4 (71616)",
        "",
        "Before feature engineering:",
        f"Accuracy: {base_accuracy:.4f}",
        f"F1-score: {base_f1:.4f}",
        "",
        "After feature engineering:",
        f"Accuracy: {new_accuracy:.4f}",
        f"F1-score: {new_f1:.4f}",
        "",
        f"Accuracy difference: {new_accuracy - base_accuracy:+.4f}",
        f"F1 difference: {new_f1 - base_f1:+.4f}"
    ]

    result = "\n".join(comparison)

    print(result)

    Path("results/comparison.txt").write_text(
        result,
        encoding="utf-8"
    )

    data_card = f"""# Lab 4 (71616)

## Dataset
UCI Adult Census Income

## Description
Predict whether annual income exceeds $50,000.

## Size
- Rows: {len(data)}
- Columns: {len(data.columns)}

## Target
income

## Model
Logistic Regression

## Processing
- Missing value imputation
- Outlier clipping
- Numerical scaling
- Categorical encoding
- Feature engineering

## New features
- capital_net
- hours_per_age

## Evaluation
Accuracy and F1-score on a 30% test set.

## Limitations
The dataset contains historical census information.
Some demographic attributes may introduce fairness concerns.
"""

    Path("data_card.md").write_text(
        data_card,
        encoding="utf-8"
    )

    print("\nFiles saved to results/")


if __name__ == "__main__":
    main()