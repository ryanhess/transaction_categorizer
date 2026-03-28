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

        assert result is None

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

    @mark.skipif(not MODEL_IS_TRAINED, reason="model not trained")
    def test_output_order_and_id(self) -> None:
        txns = [
            TransactionRequest(id=1, payee="HAZUKI SUSHI", outflow=95.86),
            TransactionRequest(
                id=2, payee="Transfer : Main Savings 5886", outflow=300.00
            ),
            TransactionRequest(id=3, payee="CHEWY.COM", outflow=53.03),
            TransactionRequest(id=4, payee="AMO SEAFOOD", outflow=100.41),
            TransactionRequest(id=5, payee="Stephen Pruden D.C.", outflow=220.00),
            TransactionRequest(id=6, payee="ParkWhiz, Inc.", outflow=21.20),
            TransactionRequest(id=7, payee="Amazon", outflow=44.85),
            TransactionRequest(id=8, payee="SAKURA RAMEN HOUSE", outflow=14.50),
            TransactionRequest(id=9, payee="BROOKLYN DINER", outflow=32.00),
            TransactionRequest(id=10, payee="H Mart", outflow=67.80),
            TransactionRequest(id=11, payee="CONTAINERSTOREWESTBURY", outflow=89.50),
            TransactionRequest(id=12, payee="OK PETROLEUM", outflow=45.00),
            TransactionRequest(
                id=13, payee="Transfer : Home Escrow 1597", outflow=157.86
            ),
            TransactionRequest(id=14, payee="The Home Depot #1213", outflow=67.23),
            TransactionRequest(id=15, payee="Solid State Coffee", outflow=6.75),
            TransactionRequest(id=16, payee="STOP 1 BAGEL& DELI", outflow=12.40),
            TransactionRequest(id=17, payee="National Grid", outflow=135.30),
            TransactionRequest(id=18, payee="LIPA", outflow=130.48),
            TransactionRequest(
                id=19, payee="Transfer : Autopay Bills 6671", outflow=620.00
            ),
            TransactionRequest(id=20, payee="TST* GOLDEN OAK BISTRO", outflow=42.50),
            TransactionRequest(id=21, payee="SORENSON LUMBER INC", outflow=234.56),
            TransactionRequest(id=22, payee="MTA*LIRR STATION TIX", outflow=18.75),
            TransactionRequest(id=23, payee="NASSAU MEAT MARKET INC", outflow=24.99),
            TransactionRequest(id=24, payee="Transfer : Marzena 7072", outflow=3500.00),
            TransactionRequest(id=25, payee="NORTH SHORE THAI KITCHEN", outflow=14.20),
            TransactionRequest(id=26, payee="GLEN HEAD HARDWARE", outflow=28.90),
            TransactionRequest(id=27, payee="State Farm", outflow=175.00),
            TransactionRequest(id=28, payee="AMAGANSETT IGA", outflow=112.45),
            TransactionRequest(
                id=29, payee="Transfer : Gabe Savings 5665", outflow=1103.29
            ),
            TransactionRequest(id=30, payee="MANGO GRILL & TACO BAR", outflow=48.30),
        ]

        results = predict(txns)

        assert results is not None
        for i, res in enumerate(results):
            assert txns[i].id == res.id
            assert txns[i].payee == res.payee
