import asyncio

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    handoff,
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


billing_agent = Agent(
    name="BillingAgent",
    instructions=(
        "Você é especialista em cobrança. "
        "Responda apenas questões relacionadas "
        "a cobranças, pagamentos e faturas."
    ),
    model=model,
)


refund_agent = Agent(
    name="RefundAgent",
    instructions=(
        "Você é especialista em reembolsos. "
        "Ajude o usuário com solicitações de devolução "
        "e reembolso."
    ),
    model=model,
)


triage_agent = Agent(
    name="TriageAgent",
    instructions=(
        "Você é responsável pela triagem. "
        "Analise a solicitação e encaminhe para "
        "o agente especialista adequado."
    ),
    handoffs=[
        billing_agent,
        handoff(refund_agent),
    ],
    model=model,
)


async def main():
    result = await Runner.run(
        triage_agent,
        "Quero solicitar o reembolso de uma compra."
    )

    print("\nResposta final:")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())