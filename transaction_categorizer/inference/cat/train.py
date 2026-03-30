import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, coo_matrix
from joblib import dump
from numpy.typing import ArrayLike
from typing import Any, Callable, cast
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


def _read_csv_training_data(sample_frac: float) -> pd.DataFrame:
    raw = pd.read_csv(training_data_filepath)
    raw = raw.sample(frac=sample_frac, random_state=42)
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


def _store_model(
    model: xgboost.XGBModel, transformers: tuple[TfidfVectorizer, LabelEncoder]
) -> None:
    payee_vectorizer, label_encoder = transformers

    model.save_model(model_filepath)
    dump(payee_vectorizer, payee_vectorizer_filepath)
    dump(label_encoder, label_encoder_filepath)


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


def _read_params_from_file() -> dict[str, Callable[[optuna.Trial], Any]]:
    stored_params_path = Path(training_params_filepath)
    if stored_params_path.exists():
        stored_params = json.loads(stored_params_path.read_text())
    else:
        stored_params = {}

    return stored_params


def _write_new_params_to_file(new_params: dict) -> None:
    """
    Writes the new parameters to the file, updating
    existing keys with new and writing new keys fresh.
    """
    stored_params_path = Path(training_params_filepath)
    if stored_params_path.exists():
        stored_params = json.loads(stored_params_path.read_text())
    else:
        stored_params = {}

    all_params = stored_params | new_params

    with open(training_params_filepath, "w") as f:
        json.dump(all_params, f, indent=2)


def train(new_params: dict = {}, sample_frac: float = 1) -> float:
    if new_params == {}:
        production = True
    else:
        production = False

    raw_data = _read_csv_training_data(sample_frac)
    cleaned_data = _clean_data_in_place(raw_data)
    transformers = _get_transformers(cleaned_data)
    features, labels = _transform_data(cleaned_data, transformers)

    traindata, testdata, trainlabels, testlabels = train_test_split(
        features, labels, test_size=0.2, stratify=labels
    )

    stored_params_path = Path(training_params_filepath)
    if stored_params_path.exists():
        stored_params = json.loads(stored_params_path.read_text())
    else:
        stored_params = {}

    if production:
        final_params = stored_params
    else:
        final_params = stored_params | new_params

    model = xgboost.XGBClassifier(**final_params)
    model.fit(traindata, trainlabels)

    if production:
        _store_model(model, transformers)

    return float(model.score(testdata, testlabels))


def _tune_specific_params(config: dict, params_to_tune: dict = {}) -> None:
    frac = config["data_sample_fraction"]

    def objective(trial) -> float:
        new_params = {k: v(trial) for k, v in params_to_tune.items()}
        result = train(new_params, frac)
        return result

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config["n_trials"], n_jobs=config["n_jobs"])  # type: ignore

    _write_new_params_to_file(study.best_params)


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
