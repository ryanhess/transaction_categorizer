from fastapi import FastAPI, HTTPException
from categorizer_service.models import TransactionRequest, TransactionResponse
from categorizer_service.inference.cat import predict


app = FastAPI()


NO_MODEL_EXCEPTION = {
    "status_code": 500,
    "detail": "No trained model found on server.",
}


@app.post(
    "/categorize",
    responses={
        NO_MODEL_EXCEPTION["status_code"]: {"description": NO_MODEL_EXCEPTION["detail"]}
    },
)
async def categorize_handler(
    txns: list[TransactionRequest],
) -> list[TransactionResponse]:
    result = predict(txns)
    if result is None:
        raise HTTPException(
            status_code=NO_MODEL_EXCEPTION["status_code"],
            detail=NO_MODEL_EXCEPTION["detail"],
        )
    return result
