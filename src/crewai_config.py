from pathlib import Path

import yaml
from crewai import Agent, Task, Crew, LLM


llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434"
)


def load_yaml(path: str):
    return yaml.safe_load(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


agents_conf = load_yaml(
    "config/agents.yaml"
)

tasks_conf = load_yaml(
    "config/tasks.yaml"
)


# --- Agentes ---

agents = {}

for key, conf in agents_conf.items():
    agents[key] = Agent(
        role=conf["role"],
        goal=conf["goal"],
        backstory=conf["backstory"],
        llm=llm
    )


# --- Tarefas ---

tasks = {}

for key, conf in tasks_conf.items():

    agente = agents[
        conf["agent"]
    ]

    task = Task(
        description=conf["description"],
        expected_output=conf["expected_output"],
        agent=agente
    )

    tasks[key] = task


# --- Contexto entre tarefas ---

for key, conf in tasks_conf.items():

    if "context" in conf:
        tasks[key].context = [
            tasks[task_name]
            for task_name in conf["context"]
        ]


# --- Crew ---

crew = Crew(
    agents=list(
        agents.values()
    ),
    tasks=list(
        tasks.values()
    ),
    verbose=True
)


if __name__ == "__main__":

    resultado = crew.kickoff(
        inputs={
            "mensagem_cliente":
                "Não consigo fazer login, aparece erro 500",
            "logs":
                "stacktrace ...",
            "tipo_dispositivo":
                "Web"
        }
    )

    print("\nResultado:")
    print(resultado)