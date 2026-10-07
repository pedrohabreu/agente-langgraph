from dataclasses import dataclass
from typing import Optional
import asyncio

from pydantic_graph import (
    GraphBuilder,
    StepContext,
)


# --------- State & Deps ---------

@dataclass
class WeatherState:
    query: str
    location: Optional[str] = None
    units: str = "metric"
    forecast: Optional[str] = None


@dataclass
class Deps:
    def fetch_weather(
        self,
        location: str,
        units: str
    ) -> str:
        return f"{location}: 26° and clear ({units})"


# --------- Graph Builder ---------

g = GraphBuilder(
    state_type=WeatherState,
    deps_type=Deps,
    input_type=None,
    output_type=str,
)


@g.step
async def parse_query(
    ctx: StepContext[WeatherState, Deps, None]
) -> str:

    if "sp" in ctx.state.query.lower():
        ctx.state.location = "São Paulo"

    return "parsed"


@g.step
async def decide_next(
    ctx: StepContext[WeatherState, Deps, str]
) -> str:

    if not ctx.state.location:
        ctx.state.location = "São Paulo"

    return "ready"


@g.step
async def fetch_forecast(
    ctx: StepContext[WeatherState, Deps, str]
) -> str:

    assert ctx.state.location is not None

    ctx.state.forecast = ctx.deps.fetch_weather(
        ctx.state.location,
        ctx.state.units
    )

    return "forecast_ready"


@g.step
async def format_summary(
    ctx: StepContext[WeatherState, Deps, str]
) -> str:

    return (
        f"Previsão para {ctx.state.location}: "
        f"{ctx.state.forecast}"
    )


# --------- Edges ---------

g.add(
    g.edge_from(g.start_node).to(parse_query),
    g.edge_from(parse_query).to(decide_next),
    g.edge_from(decide_next).to(fetch_forecast),
    g.edge_from(fetch_forecast).to(format_summary),
    g.edge_from(format_summary).to(g.end_node),
)


weather_graph = g.build()


async def main():
    state = WeatherState(
        query="Como está o tempo em SP?"
    )

    result = await weather_graph.run(
        state=state,
        deps=Deps()
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())