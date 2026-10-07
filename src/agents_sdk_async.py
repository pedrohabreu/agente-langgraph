import asyncio

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
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


async def main():

    agent = Agent(
        name="ExemploAsync",
        instructions="Você responde de forma rápida e objetiva.",
        model=model,
    )

    result = await Runner.run(
        agent,
        "Qual é a capital da França?"
    )

    print("Resposta:", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())