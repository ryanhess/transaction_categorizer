from pytest import fixture
from fastapi.testclient import TestClient
from transaction_categorizer.main import app, NO_MODEL_EXCEPTION
from transaction_categorizer.models import TransactionRequest


client = TestClient(app)


@fixture()
def mock_predict(mocker) -> None:
    return mocker.patch("transaction_categorizer.main.predict")


class TestCategorizeHandler:
    def test_no_model_returns_503(self, mock_predict) -> None:
        mock_predict.return_value = None

        transactions = [
            TransactionRequest(id=1, payee="Trader Joe's", inflow=0, outflow=100.00)
        ]

        response = client.post(
            "/categorize", json=[txn.model_dump() for txn in transactions]
        )

        assert response.status_code == 503
        assert response.json()["detail"] == NO_MODEL_EXCEPTION["detail"]

    def test_missing_id_returns_422(self) -> None:
        transactions = [{"payee": "Trader Joe's", "inflow": 0, "outflow": 100.00}]

        response = client.post("/categorize", json=transactions)

        assert response.status_code == 422

    def test_missing_payee_returns_422(self) -> None:
        transactions = [{"id": 20, "inflow": 0, "outflow": 100.00}]

        response = client.post("/categorize", json=transactions)

        assert response.status_code == 422

    def test_passing_none_returns_422(self) -> None:
        return

    def test_passing_empty_list_returns_200_and_empty_list(self) -> None:
        return

    def test_passing_list_size_1_returns_200_and_list_size_1(self) -> None:
        return
