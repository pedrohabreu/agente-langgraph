from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


model = OllamaModel(
    "llama3.2:3b",
    provider=OllamaProvider(
        base_url="http://localhost:11434/v1"
    ),
)


agent = Agent(
    model=model,
    system_prompt=(
        "Você é um assistente que responde "
        "de forma clara e estruturada."
    )
)


result = agent.run_sync(
    "Explique sucintamente o que é quantização MXFP4."
)

print(result.output)