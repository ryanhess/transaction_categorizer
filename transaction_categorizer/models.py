from pydantic import BaseModel


class TransactionRequest(BaseModel):
    id: int
    payee: str
    inflow: float = 0
    outflow: float = 0


class TransactionResponse(BaseModel):
    id: int
    category: str = "Uncategorized"
