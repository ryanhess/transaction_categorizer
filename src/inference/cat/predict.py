from models import TransactionRequest, TransactionResponse
from .train import path_to_model_state
import joblib
import xgboost
from scipy.sparse import hstack, csr_matrix


try:
    model = xgboost.XGBClassifier()
    model.load_model(path_to_model_state + "model.json")
    payee_vectorizer = joblib.load(str(path_to_model_state + "payee_vectorizer.pkl"))
    label_encoder = joblib.load(str(path_to_model_state + "category_encoder.pkl"))
except Exception:
    model = None
    payee_vectorizer = None
    label_encoder = None


def _get_features_from_data(txns: list[TransactionRequest]):
    payees = [txn.payee for txn in txns]
    payee_feature = payee_vectorizer.transform(payees)
    inflow_feature = csr_matrix([[txn.inflow] for txn in txns])
    outflow_feature = csr_matrix([[txn.outflow] for txn in txns])

    features = hstack([payee_feature, inflow_feature, outflow_feature])

    return features


def predict(transactions: list[TransactionRequest]) -> list[TransactionResponse]:
    if model is None:
        raise RuntimeError("Model not trained. Run train() first.")
    features = _get_features_from_data(transactions)
    predictions = model.predict(features)

    category_names: list[str] = label_encoder.inverse_transform(predictions)

    result: list[TransactionResponse] = []
    for i in range(len(transactions)):
        txn = transactions[i]
        category = category_names[i]
        newtxn = TransactionResponse(id=txn.id, category=category)
        result.append(newtxn)

    return result
