from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

class Estado(TypedDict):
    messages: Annotated[list, add_messages]



llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

memoria = InMemorySaver()


def responder_pergunta(estado: Estado):
    resposta = llm.invoke(estado["messages"])

    return {
        "messages": [resposta]
    }


grafo = StateGraph(Estado)

grafo.add_node("responder", responder_pergunta)

grafo.add_edge(START, "responder")
grafo.add_edge("responder", END)

agente = grafo.compile(
    checkpointer=memoria
)


config = {
    "configurable": {
        "thread_id": "conversa-pedro"
    }
}


while True:

    entrada = input("\nVocê: ")

    if entrada.lower() in ["sair", "exit"]:
        print("Encerrando conversa.")
        break

    resultado = agente.invoke(
        {
            "messages": [
                HumanMessage(content=entrada)
            ]
        },
        config=config
    )

    resposta = resultado["messages"][-1]

    print("\nAgente:")
    print(resposta.content) 