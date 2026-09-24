from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


# Banco de dados simulado
class DatabaseConn:
    async def customer_name(self, id: int) -> str:
        return f"Customer {id}"

    async def customer_balance(
        self,
        id: int,
        include_pending: bool
    ) -> float:
        return 1000.0 + (500.0 if include_pending else 0.0)


@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    support_advice: str = Field(
        description="Conselho ao cliente"
    )
    block_card: bool = Field(
        description="Bloquear cartão?"
    )
    risk: int = Field(ge=0, le=10)


model = OllamaModel(
    "llama3.2:3b",
    provider=OllamaProvider(
        base_url="http://localhost:11434/v1"
    ),
)


support_agent = Agent(
    model=model,
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt="Você é um agente de suporte bancário."
)


@support_agent.system_prompt
async def add_customer_name(
    ctx: RunContext[SupportDependencies]
) -> str:
    customer_name = await ctx.deps.db.customer_name(
        id=ctx.deps.customer_id
    )
    return f"Nome do cliente: {customer_name!r}"


@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies],
    include_pending: bool
) -> float:
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )


deps = SupportDependencies(
    customer_id=123,
    db=DatabaseConn()
)


result = support_agent.run_sync(
    "Qual é meu saldo?",
    deps=deps
)

print(result.output)