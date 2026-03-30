import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, coo_matrix
from joblib import dump
from numpy.typing import ArrayLike
from typing import cast, Any
import optuna
import json
from pathlib import Path
from .paths import (
    training_data_filepath,
    training_params_filepath,
    model_filepath,
    payee_vectorizer_filepath,
    label_encoder_filepath,
)


def _read_csv_training_data() -> pd.DataFrame:
    raw = pd.read_csv(training_data_filepath)
    return raw


def _clean_data_in_place(csvdata) -> pd.DataFrame:
    """
    Cleans the data, modifying the argument.
    Returns a reference to the cleaned data as a convenience
    """
    # find rows where the payee starts with transfer, put Transfer in the category.
    csvdata.loc[
        csvdata["Payee"].str.startswith("Transfer :"), "Category Group/Category"
    ] = "Transfer"

    csvdata["Payee"] = csvdata["Payee"].fillna("")
    csvdata["Category Group/Category"] = csvdata["Category Group/Category"].fillna(
        "Uncategorized"
    )
    counts = csvdata["Category Group/Category"].value_counts()
    keep = counts[counts >= 2].index
    csvdata = csvdata[csvdata["Category Group/Category"].isin(keep)]
    csvdata["Outflow"] = (
        csvdata["Outflow"].replace(r"[\$,]", "", regex=True).astype(float)
    )
    csvdata["Inflow"] = (
        csvdata["Inflow"].replace(r"[\$,]", "", regex=True).astype(float)
    )

    return csvdata


def _get_transformers(
    cleaned_data: pd.DataFrame,
) -> tuple[TfidfVectorizer, LabelEncoder]:
    payee_vectorizer = TfidfVectorizer()
    payee_vectorizer.fit(cleaned_data["Payee"])

    label_encoder = LabelEncoder()
    label_encoder.fit(cleaned_data["Category Group/Category"])

    return payee_vectorizer, label_encoder


def _clean_data_and_get_transformers(
    data,
) -> tuple[coo_matrix, ArrayLike, TfidfVectorizer, LabelEncoder]:
    # find rows where the payee starts with transfer, put Transfer in the category.
    data.loc[data["Payee"].str.startswith("Transfer :"), "Category Group/Category"] = (
        "Transfer"
    )

    data["Payee"] = data["Payee"].fillna("")
    data["Category Group/Category"] = data["Category Group/Category"].fillna(
        "Uncategorized"
    )
    counts = data["Category Group/Category"].value_counts()
    keep = counts[counts >= 2].index
    data = data[data["Category Group/Category"].isin(keep)]

    payee_vectorizer = TfidfVectorizer()
    payee_features = payee_vectorizer.fit_transform(data["Payee"])
    data["Outflow"] = data["Outflow"].replace(r"[\$,]", "", regex=True).astype(float)
    data["Inflow"] = data["Inflow"].replace(r"[\$,]", "", regex=True).astype(float)

    money_features = data[["Outflow", "Inflow"]].values

    features = hstack([payee_features, money_features])
    features_matrix = cast(coo_matrix, features)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["Category Group/Category"])

    return features_matrix, labels, payee_vectorizer, label_encoder


def _transform_data(
    data: pd.DataFrame, transformers: tuple
) -> tuple[coo_matrix, ArrayLike]:
    payee_vectorizer, label_encoder = transformers
    payee_features = payee_vectorizer.transform(data["Payee"])
    money_features = data[["Outflow", "Inflow"]].values

    features = hstack([payee_features, money_features])
    features_matrix = cast(coo_matrix, features)

    labels = label_encoder.transform(data["Category Group/Category"])

    return features_matrix, labels


def train() -> float:
    raw_data = _read_csv_training_data()
    cleaned_data = _clean_data_in_place(raw_data)
    transformers = _get_transformers(cleaned_data)
    features, labels = _transform_data(cleaned_data, transformers)

    params_path = Path(training_params_filepath)
    if params_path.exists():
        params = json.loads(params_path.read_text())
    else:
        params = {}

    traindata, testdata, trainlabels, testlabels = train_test_split(
        features, labels, test_size=0.2, stratify=labels
    )

    model = xgboost.XGBClassifier(**params)
    model.fit(traindata, trainlabels)

    payee_vectorizer, label_encoder = transformers

    model.save_model(model_filepath)
    dump(payee_vectorizer, payee_vectorizer_filepath)
    dump(label_encoder, label_encoder_filepath)

    return float(model.score(testdata, testlabels))


def _tune_specific_params(config: dict, params_to_tune: dict = {}) -> None:
    raw = pd.read_csv(training_data_filepath)
    raw = raw.sample(frac=config["data_sample_fraction"], random_state=42)

    data, labels, payee_vectorizer, label_encoder = _clean_data_and_get_transformers(
        raw
    )

    traindata, testdata, trainlabels, testlabels = train_test_split(
        data, labels, test_size=0.2, stratify=labels
    )

    path = Path(training_params_filepath)
    if path.exists():
        current_params = json.loads(path.read_text())
    else:
        current_params = {}

    def objective(trial) -> float:
        new_params = {k: v(trial) for k, v in params_to_tune.items()}
        study_params = current_params | new_params
        model = xgboost.XGBClassifier(**study_params)
        model.fit(traindata, trainlabels)
        return float(model.score(testdata, testlabels))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config["n_trials"], n_jobs=config["n_jobs"])  # type: ignore

    new_params_dict = current_params | study.best_params

    with open(training_params_filepath, "w") as f:
        json.dump(new_params_dict, f, indent=2)


def tune() -> None:
    # fmt: off
    param_tuning_sequence = [
        {
            "n_estimators": lambda trial:
                trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": lambda trial:
                trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        },
        {
            "min_child_weight": lambda trial:
                trial.suggest_int("min_child_weight", 1, 20),
            "max_depth": lambda trial:
                trial.suggest_int("max_depth", 2, 8),
        },
        {
            "subsample": lambda trial:
                trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": lambda trial:
                trial.suggest_float("colsample_bytree", 0.2, 1.0),
        },
    ]
    # fmt: on

    config = {
        "n_trials": 20,
        "n_jobs": -1,
        "data_sample_fraction": 0.5,
    }

    for group in param_tuning_sequence:
        _tune_specific_params(config, group)
