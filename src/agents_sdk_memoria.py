import asyncio

from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    SQLiteSession,
    OpenAIChatCompletionsModel,
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


agent = Agent(
    name="AgenteMemoria",
    instructions=(
        "Você responde perguntas usando o contexto "
        "do diálogo anterior."
    ),
    model=model,
)


session = SQLiteSession(
    "sessao_usuario_1"
)


async def run_interactions():

    res1 = await Runner.run(
        agent,
        "Onde fica o Taj Mahal?",
        session=session,
    )

    print("Resposta 1:")
    print(res1.final_output)

    res2 = await Runner.run(
        agent,
        "E em que país ele está?",
        session=session,
    )

    print("\nResposta 2:")
    print(res2.final_output)


if __name__ == "__main__":
    asyncio.run(run_interactions())