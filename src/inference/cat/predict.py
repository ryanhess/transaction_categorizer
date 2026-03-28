from src.models import TransactionRequest, TransactionResponse
from .train import path_to_model_state
from joblib import load as joblib_load
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
from scipy.sparse import hstack, csr_matrix, coo_array
from typing import cast

# this loading is done at import time. Makes predictions fast and initial load simple.
# Don't want to load for each prediction.
try:
    _model = XGBClassifier()
    _model.load_model(path_to_model_state + "model.json")
    _payee_vectorizer = joblib_load(str(path_to_model_state + "payee_vectorizer.pkl"))
    _label_encoder = joblib_load(str(path_to_model_state + "category_encoder.pkl"))
    model_is_trained = True
except (XGBoostError, FileNotFoundError):
    model_is_trained = False


def _get_features_from_data(txns: list[TransactionRequest]) -> coo_array | None:
    payees = [txn.payee for txn in txns]
    if not model_is_trained:
        return None
    payee_feature = _payee_vectorizer.transform(payees)
    inflow_feature = csr_matrix([[txn.inflow] for txn in txns])
    outflow_feature = csr_matrix([[txn.outflow] for txn in txns])

    features = hstack([payee_feature, inflow_feature, outflow_feature])

    return cast(coo_array, features)


def predict(transactions: list[TransactionRequest]) -> list[TransactionResponse]:
    features = _get_features_from_data(transactions)
    if not model_is_trained:
        raise RuntimeError("Model not trained. Run train() first.")
    predictions = _model.predict(features)

    category_names: list[str] = _label_encoder.inverse_transform(predictions)

    result: list[TransactionResponse] = []
    for i in range(len(transactions)):
        txn = transactions[i]
        category = category_names[i]
        newtxn = TransactionResponse(id=txn.id, category=category)
        result.append(newtxn)

    return result
