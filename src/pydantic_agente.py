from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


class CityLocation(BaseModel):
    city: str
    country: str


model = OllamaModel(
    "llama3.2:3b",
    provider=OllamaProvider(
        base_url="http://localhost:11434/v1"
    ),
)


agent = Agent(
    model=model,
    output_type=CityLocation
)


result = agent.run_sync(
    "Where were the 2012 Olympics held?"
)

print(result.output)

print(result.output)

print("\nTipo do resultado:")
print(type(result.output))

print("\nCidade:")
print(result.output.city)

print("\nPaís:")
print(result.output.country)