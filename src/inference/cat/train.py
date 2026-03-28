import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, coo_matrix
from joblib import dump
from numpy.typing import ArrayLike
from typing import cast


path_to_training_data = "src/inference/cat/training_data/ynab-rh-txns.csv"
path_to_model_state = "src/inference/cat/state/"


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
