from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


class Estado(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def responder_pergunta(estado: Estado):
    resposta = llm.invoke(estado["messages"])

    return {
        "messages": [resposta]
    }


grafo = StateGraph(Estado)

grafo.add_node("responder", responder_pergunta)

grafo.add_edge(START, "responder")
grafo.add_edge("responder", END)

agente = grafo.compile()


estado = {
    "messages": []
}


while True:

    entrada = input("\nVocê: ")

    if entrada.lower() in ["sair", "exit"]:
        print("Encerrando conversa.")
        break

    estado["messages"].append(
        HumanMessage(content=entrada)
    )

    resultado = agente.invoke(estado)

    resposta = resultado["messages"][-1]

    print("\nAgente:")
    print(resposta.content)

    estado = resultado