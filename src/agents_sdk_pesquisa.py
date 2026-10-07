import asyncio

from pydantic import BaseModel
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


class Report(BaseModel):
    summary: str
    sources: list[str]


@function_tool
def pesquisar_dados(tema: str) -> str:
    """
    Simula resultados provenientes de uma pesquisa externa.
    """

    return (
        f"Dados coletados sobre {tema}: "
        "há alterações observadas em padrões de precipitação, "
        "temperatura e eventos climáticos extremos. "
        "Fonte simulada: relatório científico."
    )


search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "Colete informações relevantes usando "
        "a ferramenta pesquisar_dados."
    ),
    tools=[pesquisar_dados],
    model=model,
)


analysis_agent = Agent(
    name="AnalysisAgent",
    instructions=(
        "Analise criticamente as informações recebidas, "
        "identifique pontos importantes e possíveis limitações."
    ),
    model=model,
)


synthesis_agent = Agent(
    name="SynthesisAgent",
    instructions=(
        "Produza um relatório curto e estruturado "
        "a partir das informações disponíveis."
    ),
    output_type=Report,
    model=model,
)


main_agent = Agent(
    name="ResearchAssistant",
    instructions=(
        "Você é um assistente de pesquisa. "
        "Quando precisar coletar informações, transfira "
        "para SearchAgent. Para análise, use AnalysisAgent."
    ),
    handoffs=[
        search_agent,
        analysis_agent,
    ],
    model=model,
)


async def main():

    result = await Runner.run(
        main_agent,
        (
            "Explique brevemente os impactos da mudança "
            "climática nos padrões de monções na Ásia."
        ),
    )

    print("\nResultado:")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())