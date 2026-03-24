from pydantic import BaseModel


class TransactionRequest(BaseModel):
    id: int
    payee: str
    inflow: float
    outflow: float


class TransactionResponse(BaseModel):
    id: int
    category: str
