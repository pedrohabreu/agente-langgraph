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
def get_weather(city: str) -> str:
    return f"O tempo em {city} é ensolarado."


agent = Agent(
    name="WeatherAgent",
    instructions=(
        "Responda perguntas sobre clima usando a ferramenta "
        "get_weather quando apropriado."
    ),
    tools=[get_weather],
    model=model,
)


async def main():
    result = await Runner.run(
        agent,
        "Como está o tempo em São Paulo?"
    )

    print("\nResposta final:")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())