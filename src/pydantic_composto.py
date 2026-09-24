from pydantic import BaseModel
from typing import List


class Endereco(BaseModel):
    rua: str
    cidade: str


class Item(BaseModel):
    id_item: int
    descricao: str


class Pedido(BaseModel):
    id_pedido: int
    itens: List[Item]


class Cliente(BaseModel):
    nome: str
    endereco: Endereco
    pedidos: List[Pedido]

cliente = Cliente(
    nome="Pedro",
    endereco={
        "rua": "Rua das Flores",
        "cidade": "Brasilia"
    },
    pedidos=[
        {
            "id_pedido": 1001,
            "itens": [
                {
                    "id_item": 1,
                    "descricao": "Notebook"
                },
                {
                    "id_item": 2,
                    "descricao": "Mouse"
                }
            ]
        }
    ]
)

print(cliente)

print("\nNome:")
print(cliente.nome)

print("\nCidade:")
print(cliente.endereco.cidade)

print("\nPrimeiro item:")
print(cliente.pedidos[0].itens[0].descricao)