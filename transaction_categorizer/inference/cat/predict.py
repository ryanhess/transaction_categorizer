from transaction_categorizer.models import TransactionRequest, TransactionResponse
from .paths import model_filepath, payee_vectorizer_filepath, label_encoder_filepath
from joblib import load as joblib_load
from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix, coo_array
from typing import cast

# top level def makes model loading happen at server startup
if (
    model_filepath.exists()
    and payee_vectorizer_filepath.exists()
    and label_encoder_filepath.exists()
):
    MODEL_IS_TRAINED = True
    _MODEL = XGBClassifier()
    _MODEL.load_model(model_filepath)
    _PAYEE_VECTORIZER = joblib_load(payee_vectorizer_filepath)
    _LABEL_ENCODER = joblib_load(label_encoder_filepath)
else:
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
