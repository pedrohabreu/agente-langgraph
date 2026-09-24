from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


model = OllamaModel(
    "llama3.2:3b",
    provider=OllamaProvider(
        base_url="http://localhost:11434/v1"
    ),
)


class MedicalDatabase(Protocol):
    async def get_history(self, patient_id: int) -> str: ...
    async def get_patient_name(self, patient_id: int) -> str: ...


@dataclass
class RequestContext:
    patient_id: int
    db: MedicalDatabase


class AssistanceRequest(BaseModel):
    symptoms: str = Field(max_length=500)
    urgency: int = Field(ge=1, le=5)


class AssistanceResponse(BaseModel):
    advice: str
    risk_level: int = Field(ge=0, le=10)
    refer_to_specialist: bool


assist_agent = Agent(
    model=model,
    deps_type=RequestContext,
    output_type=AssistanceResponse,
    instructions=(
        "Você é um assistente médico virtual. "
        "Avalie os sintomas de forma prudente, comunique incertezas, "
        "e forneça orientação geral e não-diagnóstica. "
        "Se houver sinais de emergência, incentive procurar "
        "atendimento imediato. "
        "Não prescreva medicamentos."
    ),
)


@assist_agent.tool
async def fetch_patient_history(
    ctx: RunContext[RequestContext]
) -> str:
    return await ctx.deps.db.get_history(
        patient_id=ctx.deps.patient_id
    )


@assist_agent.system_prompt
async def add_patient_context(
    ctx: RunContext[RequestContext]
) -> str:
    name = await ctx.deps.db.get_patient_name(
        ctx.deps.patient_id
    )
    return (
        f"Contexto: paciente '{name}' "
        f"(id={ctx.deps.patient_id})."
    )


class FakeDB:
    async def get_history(self, patient_id: int) -> str:
        return "Hipertensão controlada; sem alergias registradas."

    async def get_patient_name(self, patient_id: int) -> str:
        return "José da Silva"


deps = RequestContext(
    patient_id=42,
    db=FakeDB()
)

question = AssistanceRequest(
    symptoms="Dor no peito e falta de ar ao esforço",
    urgency=5
)

result = assist_agent.run_sync(
    f"Avalie: {question.model_dump_json()}",
    deps=deps
)

print(result.output)