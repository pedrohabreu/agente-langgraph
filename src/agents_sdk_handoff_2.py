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
        "Responda exclusivamente dúvidas de cobrança, "
        "pagamentos e faturas."
    ),
    model=model,
)


refund_agent = Agent(
    name="RefundAgent",
    instructions=(
        "Cuide exclusivamente de pedidos de reembolso "
        "e devolução."
    ),
    model=model,
)


triage_agent = Agent(
    name="TriageAgent",
    instructions=(
        "Direcione questões de cobrança ou reembolso "
        "ao agente especialista apropriado."
    ),
    handoffs=[
        billing_agent,
        handoff(refund_agent),
    ],
    model=model,
)


async def main():
    resultado = await Runner.run(
        triage_agent,
        "Fui cobrado duas vezes pela mesma compra."
    )

    print("\nResposta final:")
    print(resultado.final_output)


if __name__ == "__main__":
    asyncio.run(main())