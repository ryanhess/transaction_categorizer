from categorizer_service.models import TransactionRequest, TransactionResponse
from .train import path_to_model_state
from joblib import load as joblib_load
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
from scipy.sparse import hstack, csr_matrix, coo_array
from typing import cast

# this loading is done at import time. Makes predictions fast and initial load simple.
# Don't want to load for each prediction.
try:
    _MODEL = XGBClassifier()
    _MODEL.load_model(path_to_model_state + "model.json")
    _PAYEE_VECTORIZER = joblib_load(str(path_to_model_state + "payee_vectorizer.pkl"))
    _LABEL_ENCODER = joblib_load(str(path_to_model_state + "category_encoder.pkl"))
    MODEL_IS_TRAINED = True
except (XGBoostError, FileNotFoundError):
    MODEL_IS_TRAINED = False


def _get_features_from_data(txns: list[TransactionRequest]) -> coo_array | None:
    payees = [txn.payee for txn in txns]
    if not MODEL_IS_TRAINED:
        return None
    payee_feature = _PAYEE_VECTORIZER.transform(payees)
    inflow_feature = csr_matrix([[txn.inflow] for txn in txns])
    outflow_feature = csr_matrix([[txn.outflow] for txn in txns])

    features = hstack([payee_feature, inflow_feature, outflow_feature])

    return cast(coo_array, features)


def predict(transactions: list[TransactionRequest]) -> list[TransactionResponse] | None:
    features = _get_features_from_data(transactions)
    if not MODEL_IS_TRAINED:
        print("Error: model state not found. Train model first.")
        return None
    predictions = _MODEL.predict(features)

    category_names: list[str] = _LABEL_ENCODER.inverse_transform(predictions)

    result: list[TransactionResponse] = []
    for i in range(len(transactions)):
        txn = transactions[i]
        category = category_names[i]
        newtxn = TransactionResponse(id=txn.id, category=category)
        result.append(newtxn)

    return result
