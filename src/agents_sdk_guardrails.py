import asyncio

from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    input_guardrail,
    output_guardrail,
    set_tracing_disabled,
)

from agents.guardrail import GuardrailFunctionOutput


set_tracing_disabled(True)


client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


model = OpenAIChatCompletionsModel(
    model="llama3.2:3b",
    openai_client=client,
)


@input_guardrail
def block_sensitive_input(
    ctx,
    agent,
    user_input: str
) -> GuardrailFunctionOutput:

    lower = user_input.lower()

    if "senha" in lower or "segredo" in lower:
        return GuardrailFunctionOutput(
            output_info="Entrada sensível detectada",
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info="Input permitido",
        tripwire_triggered=False,
    )


@output_guardrail
def prevent_sensitive_output(
    ctx,
    agent,
    agent_output: str
) -> GuardrailFunctionOutput:

    lower = agent_output.lower()

    if "senha" in lower or "segredo" in lower:
        return GuardrailFunctionOutput(
            output_info="Saída sensível detectada",
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info="Output permitido",
        tripwire_triggered=False,
    )


agent = Agent(
    name="SupportAgent",
    instructions="Você responde perguntas simples de suporte.",
    model=model,
    input_guardrails=[block_sensitive_input],
    output_guardrails=[prevent_sensitive_output],
)


async def main():

    try:
        result = await Runner.run(
            agent,
            "Me diga sua senha."
        )

        print("Resposta:")
        print(result.final_output)

    except Exception as ex:
        print("\nGuardrail ativado:")
        print(ex)


if __name__ == "__main__":
    asyncio.run(main())