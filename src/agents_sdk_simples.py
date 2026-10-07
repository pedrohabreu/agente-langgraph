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


agent = Agent(
    name="AssistenteSimples",
    instructions=(
        "Você é um assistente que responde "
        "com clareza e simplicidade."
    ),
    model=model,
)


async def main():
    result = await Runner.run(
        agent,
        "Explique o que é recursão."
    )

    print("Resposta:")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())