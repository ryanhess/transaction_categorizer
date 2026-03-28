from pytest import mark, raises
from transaction_categorizer.inference.cat.predict import (
    predict,
    MODEL_IS_TRAINED,
)
from transaction_categorizer.models import TransactionRequest, TransactionResponse


txns_param = [
    TransactionRequest(id=1, payee="The Home Depot", inflow=0, outflow=10.25),
    TransactionRequest(id=2, payee="Whole Foods", inflow=0, outflow=65.43),
]


class TestGetFeaturesFromData:
    pass


class TestPredict:
    def test_returns_none_when_no_model(self, mocker) -> None:
        mocker.patch(
            "transaction_categorizer.inference.cat.predict.MODEL_IS_TRAINED", new=False
        )

        result = predict(txns_param)

        assert result == None

    @mark.skipif(not MODEL_IS_TRAINED, reason="model not trained")
    def test_returns_list_of_correct_type_and_count(self) -> None:
        result = predict(txns_param)

        assert isinstance(result, list)
        assert len(result) == len(txns_param)
        assert all(isinstance(res, TransactionResponse) for res in result)

    @mark.skipif(not MODEL_IS_TRAINED, reason="model not trained")
    def test_predict_raises_value_error_with_empty_list(self) -> None:
        with raises(ValueError, match="TfidfTransformer"):
            predict([])

    @mark.skipif(not MODEL_IS_TRAINED, reason="model not trained")
    def test_categories_in_result_are_strings(self) -> None:
        result = predict(txns_param)

        assert isinstance(result, list)
        assert all(not res.category.isnumeric() for res in result)
