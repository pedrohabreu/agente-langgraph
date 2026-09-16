from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama


class Estado(TypedDict):
    messages: Annotated[list, add_messages]



llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def entrada_usuario(state: Estado) -> Estado:
    pergunta = input("Pergunta: ")

    if isinstance(pergunta, str) and pergunta.strip():
        return {
            "messages": [
                HumanMessage(content=pergunta)
            ]
        }

    else:
        raise ValueError(
            "A pergunta deve ser uma string não vazia."
        )
def processar_solicitacao(state: Estado) -> Estado:
    last_message = state["messages"][-1]

    if (
        isinstance(last_message, HumanMessage)
        and last_message.content.strip()
    ):
        print(
            f"DEBUG: Processando pergunta: "
            f"{last_message.content}"
        )

        try:
            resposta = llm.invoke(
                [last_message]
            ).content

            print(
                f"DEBUG: Resposta gerada: {resposta}"
            )

            return {
                "messages": [
                    AIMessage(content=resposta)
                ]
            }

        except Exception as e:
            print(
                f"DEBUG: Erro ao chamar o LLM: {e}"
            )

            return {
                "messages": [
                    AIMessage(
                        content=f"Desculpe, ocorreu um erro: {e}"
                    )
                ]
            }

    else:
        raise ValueError(
            "A última mensagem no estado "
            "não é uma HumanMessage válida."
        )

def saida_resposta(state: Estado) -> Estado:
    last_message = state["messages"][-1]

    if hasattr(last_message, "content"):
        print(f"Resposta: {last_message.content}")
    else:
        print(f"Resposta: {last_message}")

    return state

grafo = StateGraph(Estado)

grafo.add_node("entrada", entrada_usuario)
grafo.add_node("processamento", processar_solicitacao)
grafo.add_node("saida", saida_resposta)

grafo.add_edge(START, "entrada")
grafo.add_edge("entrada", "processamento")
grafo.add_edge("processamento", "saida")
grafo.add_edge("saida", END)

agente = grafo.compile()

print("Iniciando a interação...")

while True:
    try:
        agente.invoke({"messages": []})

        continuar = input(
            "Deseja fazer outra pergunta? (sim/não): "
        )

        if continuar.lower() != "sim":
            break

    except ValueError as ve:
        print(f"Erro de entrada: {ve}")

    except KeyboardInterrupt:
        print("\nInteração encerrada pelo usuário.")
        break

    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        break

print("Interação encerrada.")

