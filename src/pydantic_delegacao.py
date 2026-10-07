from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


# --- Modelo local ---
model = OllamaModel(
    "llama3.2:3b",
    provider=OllamaProvider(
        base_url="http://localhost:11434/v1"
    ),
)


# --- Agente principal ---
joke_selection_agent = Agent(
    model,
    system_prompt=(
        "Use a `joke_factory` para gerar piadas, "
        "depois escolha a melhor e retorne apenas uma."
    )
)


# --- Agente especialista ---
joke_generation_agent = Agent(
    model,
    output_type=list[str]
)


# --- Tool que delega trabalho ao segundo agente ---
@joke_selection_agent.tool
async def joke_factory(
    ctx: RunContext[None],
    count: int
) -> list[str]:

    r = await joke_generation_agent.run(
        f"Por favor, gere {count} piadas.",
        usage=ctx.usage,
    )

    return r.output


# --- Execução ---
result = joke_selection_agent.run_sync(
    "Conte uma piada.",
    usage_limits=UsageLimits(
        request_limit=5,
        total_tokens_limit=2000
    ),
)

print(result.output)
print(result.usage)