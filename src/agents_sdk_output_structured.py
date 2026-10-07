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


class WeatherResponse(BaseModel):
    temperature: float
    description: str


@function_tool
def get_weather(city: str) -> WeatherResponse:
    return WeatherResponse(
        temperature=25.5,
        description=f"Ensolarado com céu claro em {city}"
    )


agent = Agent(
    name="WeatherAgentStruct",
    instructions=(
        "Você fornece previsões do tempo "
        "em formato estruturado."
    ),
    output_type=WeatherResponse,
    tools=[get_weather],
    model=model,
)


async def main():
    result = await Runner.run(
        agent,
        "Previsão do tempo em São Paulo?"
    )

    print("Saída estruturada:")
    print(result.final_output)

    print("\nTemperatura:")
    print(result.final_output.temperature)

    print("\nDescrição:")
    print(result.final_output.description)


if __name__ == "__main__":
    asyncio.run(main())