from dataclasses import dataclass
from datetime import date

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel


@dataclass
class MockDeps:
    weather_api: object | None = None


weather_agent = Agent(
    TestModel(),
    deps_type=MockDeps,
    system_prompt="Stubbed weather agent for testing",
)


@weather_agent.tool
def run_weather_forecast(
    ctx: RunContext[MockDeps],
    city: str = "São Paulo",
    when: date | None = None
) -> str:

    d = (when or date.today()).isoformat()

    return (
        f"Previsão fake para {city} "
        f"em {d}: céu limpo."
    )


async def test_weather_agent_simple():
    deps = MockDeps()

    with weather_agent.override(
        model=TestModel()
    ):
        result = await weather_agent.run(
            "Como está o tempo em SP?",
            deps=deps
        )

    assert isinstance(result.output, str)
    assert "Previsão fake" in result.output

    print(result.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        test_weather_agent_simple()
    )