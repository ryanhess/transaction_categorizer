from fastapi import FastAPI
from src.models import TransactionRequest, TransactionResponse

app = FastAPI()


@app.post("/categorize")
async def categorize_handler(
    txns: list[TransactionRequest],
) -> list[TransactionResponse]:
    return []
