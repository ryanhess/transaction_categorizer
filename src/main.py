from fastapi import FastAPI
from src.models import TransactionRequest, TransactionResponse
from src.inference.cat import predict

app = FastAPI()


@app.post("/categorize")
async def categorize_handler(
    txns: list[TransactionRequest],
) -> list[TransactionResponse]:
    return predict(txns)
