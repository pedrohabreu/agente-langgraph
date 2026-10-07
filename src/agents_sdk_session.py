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
    name="AgenteConversacional",
    instructions=(
        "Você conversa naturalmente e mantém "
        "o contexto das mensagens anteriores."
    ),
    model=model,
)


session = SQLiteSession("sessao_123")


async def run_conversation():

    res1 = await Runner.run(
        agent,
        "Onde fica a Torre Eiffel?",
        session=session,
    )

    print("Resposta 1:")
    print(res1.final_output)


    res2 = await Runner.run(
        agent,
        "E qual é o país?",
        session=session,
    )

    print("\nResposta 2:")
    print(res2.final_output)


if __name__ == "__main__":
    asyncio.run(run_conversation())