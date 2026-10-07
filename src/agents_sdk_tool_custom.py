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


@function_tool
def consulta_clima(cidade: str) -> dict:
    """
    Simula uma consulta a uma API de clima.
    """

    return {
        "cidade": cidade,
        "temperature": 27.0,
        "condition": "ensolarado"
    }


agent = Agent(
    name="AgenteClimaAvancado",
    instructions=(
        "Use a ferramenta consulta_clima "
        "para responder perguntas sobre clima."
    ),
    tools=[consulta_clima],
    model=model,
)


resultado = Runner.run_sync(
    agent,
    "Como está o tempo em Fortaleza hoje?"
)


print("Resposta:")
print(resultado.final_output)