import asyncio

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)


# Como estamos usando Ollama local,
# não precisamos do tracing da OpenAI
set_tracing_disabled(True)


# Cliente compatível com a API OpenAI,
# apontando para o Ollama
client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


# Modelo local
model = OpenAIChatCompletionsModel(
    model="llama3.2:3b",
    openai_client=client,
)


# Agente
agent = Agent(
    name="Assistant",
    instructions="Você é um assistente prestativo.",
    model=model,
)


async def main():
    result = await Runner.run(
        agent,
        "Escreva um poema curto sobre recursão em programação."
    )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())