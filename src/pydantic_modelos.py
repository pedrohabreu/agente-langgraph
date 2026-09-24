from pydantic import BaseModel, Field
from decimal import Decimal
from typing import Optional


class Produto(BaseModel):
    nome: str = Field(min_length=1, max_length=100)
    preco: Decimal = Field(gt=0, le=10000)
    descricao: Optional[str] = Field(None, max_length=500)
    categoria: str = Field(..., pattern=r'^[A-Za-z\s]+$')
    em_estoque: int = Field(ge=0)
    disponivel: bool = True


try:
    produto_invalido = Produto(
        nome="",
        preco=-5,
        categoria="1234",
        em_estoque=-10
    )

except Exception as e:
    print("Erros de validação:")
    print(e)

produto_valido = Produto(
    nome="Notebook Gamer",
    