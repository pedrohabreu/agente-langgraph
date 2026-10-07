import asyncio

from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)


set_tracing_disabled(True)


client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


model = OpenAIChatCompletionsModel(
    model="llama3.2:3b",
    openai_client=client,
)


@function_tool
def consulta_clima(cidade: str) -> dict:
    return {
        "cidade": cidade,
        "temperature": 28.0,
        "condition": "ensolarado",
    }


@function_tool
def sugere_atividade(
    cidade: str,
    temperature: float
) -> str:

    if temperature > 30:
        return (
            f"Em {cidade}, com {temperature:.1f}°C, "
            "piscina ou praia pode ser uma boa opção."
        )

    elif temperature > 20:
        return (
            f"Em {cidade}, com {temperature:.1f}°C, "
            "uma caminhada ao ar livre pode ser uma boa opção."
        )

    else:
        return (
            f"Em {cidade}, com {temperature:.1f}°C, "
            "um café ou museu pode ser uma boa opção."
        )


agent = Agent(
    name="AgenteCulturaClima",
    instructions=(
        "Responda perguntas sobre clima e atividades. "
        "Primeiro use consulta_clima para obter a temperatura. "
        "Depois use sugere_atividade com a temperatura obtida."
    ),
    tools=[
        consulta_clima,
        sugere_atividade,
    ],
    model=model,
)


async def main():

    res = await Runner.run(
        agent,
        "Qual o clima em Recife e que atividade você sugere?"
    )

    print("\nResposta:")
    print(res.final_output)


if __name__ == "__main__":
    asyncio.run(main())