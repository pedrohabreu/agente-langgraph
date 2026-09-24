import random

from pydantic_ai import Agent, RunContext
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
    deps_type=str,
    system_prompt=(
        "Você é um jogo de dados: "
        "role o dado e veja se acerta o palpite."
    )
)

@agent.tool_plain
def roll_dice() -> str:
    return str(random.randint(1, 6))

dice_result = agent.run_sync(
    "Meu palpite é 4",
    deps="Carlos"
)

print(dice_result.output) 