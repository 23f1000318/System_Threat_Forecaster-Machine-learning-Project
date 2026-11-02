"""Train and save the malware-threat prediction service model.

Usage:
    python train.py --train-csv data/train.csv --output model/model.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


DROP_COLUMNS = {
    "MachineID",
    "IsBetaUser",
    "AutoSampleSubmissionEnabled",
    "IsFlightsDisabled",
    "OEMModelID",
    "EnableLUA",
    "OSUILocaleID",
}
DATE_COLUMNS = ("DateAS", "DateOS")


def prepare_frame(frame: pd.DataFrame, *, training: bool, retained_columns=None):
    """Apply the notebook's raw-data cleaning and feature engineering."""
    data = frame.copy()
    if training:
        data = data.drop_duplicates().reset_index(drop=True)
        keep = [column for column in data.columns if data[column].notna().mean() >= 0.30]
    else:
        keep = list(retained_columns)
    data = data.reindex(columns=keep)
    data = data.drop(columns=[c for c in DROP_COLUMNS if c in data], errors="ignore")

    for column in DATE_COLUMNS:
        if column in data:
            values = pd.to_datetime(data[column], errors="coerce")
            data[f"{column}_year"] = values.dt.year
            data[f"{column}_month"] = values.dt.month
            data[f"{column}_day"] = values.dt.day
            data = data.drop(columns=column)

    if training and "target" in data:
        data = data.drop(columns="target")
    return data, keep


def make_preprocessor(data: pd.DataFrame) -> ColumnTransformer:
    numeric = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    low_cardinality = [c for c in categorical if data[c].nunique(dropna=True) <= 10]
    nominal = [c for c in categorical if c not in low_cardinality]

    transformers = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        ),
        (
            "ordinal",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]
            ),
            low_cardinality,
        ),
        (
            "nominal",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            nominal,
        ),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop")


def train(train_csv: str, output: str) -> None:
    raw = pd.read_csv(train_csv)
    if "target" not in raw:
        raise ValueError("The training CSV must contain a 'target' column.")

    # Keep the target aligned with the duplicate removal performed by the
    # notebook's preprocessing workflow.
    raw = raw.drop_duplicates().reset_index(drop=True)
    y = raw["target"].astype(int)
    features, retained_columns = prepare_frame(raw.drop(columns="target"), training=True)
    preprocessor = make_preprocessor(features)
    transformed = preprocessor.fit_transform(features)

    selector_estimator = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    selector_estimator.fit(transformed, y)
    selector = SelectFromModel(selector_estimator, prefit=True, max_features=200)
    selected = selector.transform(transformed)

    model = lgb.LGBMClassifier(
        num_leaves=400,
        learning_rate=0.01,
        n_estimators=900,
        max_bin=300,
        boosting_type="gbdt",
        random_state=42,
        verbosity=-1,
    )
    model.fit(selected, y)

    bundle = {
        "preprocessor": preprocessor,
        "selector": selector,
        "model": model,
        "retained_columns": retained_columns,
        "model_input_columns": features.columns.tolist(),
        "version": 1,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path, compress=3)
    print(f"Saved model bundle to {output_path}")
    print(f"Training rows: {len(y)}; transformed features: {transformed.shape[1]}; selected: {selected.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--output", default="model/model.joblib")
    args = parser.parse_args()
    train(args.train_csv, args.output)
