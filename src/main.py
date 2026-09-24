from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def analisar_sentimento(comentario: str) -> str:
    prompt = (
        "Classifique o sentimento do seguinte comentário como "
        "POSITIVO, NEGATIVO ou NEUTRO:\n"
        f'Comentário: "{comentario}"'
    )

    resposta = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return resposta.content.strip().upper()


while True:
    comentario = input(
        "\nDigite um comentário ou 'sair' para encerrar: "
    )

    if comentario.lower() == "sair":
        print("Análise encerrada.")
        break

    sentimento = analisar_sentimento(comentario)

    print(
        f"O sentimento do comentário é: {sentimento}"
    )