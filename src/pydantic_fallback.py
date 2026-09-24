from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


# --- Modelo local ---
model = OllamaModel(
    "llama3.2:3b",
    provider=OllamaProvider(
        base_url="http://localhost:11434/v1"
    ),
)


# --- Contrato para o banco de dados ---
class MedicalDatabase(Protocol):
    async def get_history(self, patient_id: int) -> str:
        ...

    async def get_patient_name(self, patient_id: int) -> str:
        ...


# --- Dependências do agente ---
@dataclass
class RequestContext:
    patient_id: int
    db: MedicalDatabase


# --- Saída principal ---
class AssistanceResponse(BaseModel):
    advice: str
    risk_level: int = Field(ge=0, le=10)
    refer_to_specialist: bool


# --- Saída alternativa (fallback) ---
class EmergencyResponse(BaseModel):
    emergency_advice: str


# --- Definição do agente ---
assist_agent = Agent(
    model=model,
    deps_type=RequestContext,
    output_type=[
        AssistanceResponse,
        EmergencyResponse
    ],
    system_prompt=(
        "Se não tiver certeza, recomende que "
        "busque atendimento de emergência."
    ),
    instructions=(
        "Você é um assistente médico virtual. "
        "Avalie os sintomas de forma prudente, comunique incertezas, "
        "e forneça orientação geral e não-diagnóstica. "
        "Se houver sinais de emergência, incentive procurar "
        "atendimento imediato. "
        "Não prescreva medicamentos."
    ),
)


# --- Tool para consultar o histórico ---
@assist_agent.tool
async def fetch_patient_history(
    ctx: RunContext[RequestContext]
) -> str:
    return await ctx.deps.db.get_history(
        patient_id=ctx.deps.patient_id
    )


# --- Contexto dinâmico do paciente ---
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


# --- Banco de dados fictício ---
class FakeDB:
    async def get_history(self, patient_id: int) -> str:
        return (
            "Hipertensão controlada; "
            "sem alergias registradas."
        )

    async def get_patient_name(self, patient_id: int) -> str:
        return "José da Silva"


# --- Dependências ---
deps = RequestContext(
    patient_id=42,
    db=FakeDB()
)


# --- Execução ---
result = assist_agent.run_sync(
    (
        "Avalie o paciente com dor no peito e "
        "falta de ar ao esforço."
    ),
    deps=deps
)


print(result.output)