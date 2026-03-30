import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, coo_matrix
from joblib import dump
from numpy.typing import ArrayLike
from typing import cast
import optuna
import json


path_to_training_data = (
    "transaction_categorizer/inference/cat/training_data/ynab-rh-txns.csv"
)
path_to_model_state = "transaction_categorizer/inference/cat/state/"


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


def train() -> float:
    raw = pd.read_csv(path_to_training_data)

    data, labels, payee_vectorizer, label_encoder = _clean_data_and_get_transformers(
        raw
    )

    traindata, testdata, trainlabels, testlabels = train_test_split(
        data, labels, test_size=0.2, stratify=labels
    )

    model = xgboost.XGBClassifier()
    model.fit(traindata, trainlabels)

    model.save_model(path_to_model_state + "model.json")
    dump(payee_vectorizer, path_to_model_state + "payee_vectorizer.pkl")
    dump(label_encoder, path_to_model_state + "category_encoder.pkl")

    return float(model.score(testdata, testlabels))


def tune(data_sample_fraction: float = 0.05) -> None:
    raw = pd.read_csv(path_to_training_data)
    raw = raw.sample(frac=data_sample_fraction, random_state=42)

    data, labels, payee_vectorizer, label_encoder = _clean_data_and_get_transformers(
        raw
    )

    traindata, testdata, trainlabels, testlabels = train_test_split(
        data, labels, test_size=0.2, stratify=labels
    )

    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }
        model = xgboost.XGBClassifier(**params)
        model.fit(traindata, trainlabels)
        return float(model.score(testdata, testlabels))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)  # type: ignore

    with open(path_to_model_state + "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
