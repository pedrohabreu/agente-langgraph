from crewai import Agent, Task, Crew, Process, LLM


llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434"
)


# --- Agentes ---

agent_researcher = Agent(
    role="Pesquisador Técnico",
    goal=(
        "Coletar informações detalhadas e confiáveis "
        "sobre temas técnicos"
    ),
    backstory=(
        "Você é um pesquisador acadêmico com atenção "
        "a detalhes e experiência com documentação técnica."
    ),
    allow_delegation=True,
    verbose=True,
    llm=llm
)


agent_writer = Agent(
    role="Redator Técnico",
    goal=(
        "Produzir conteúdo claro baseado em pesquisa, "
        "com exemplos práticos"
    ),
    backstory=(
        "Você escreve guias, tutoriais e documentação "
        "técnica para desenvolvedores."
    ),
    allow_delegation=True,
    verbose=True,
    llm=llm
)


agent_reviewer = Agent(
    role="Revisor Técnico",
    goal=(
        "Garantir precisão técnica e clareza, "
        "evitando erros ou ambiguidades"
    ),
    backstory=(
        "Você trabalha revisando artigos científicos "
        "e documentação técnica, com atenção a "
        "terminologia e consistência."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)


# --- Tarefa ---

task_write_guide = Task(
    description=(
        "Produza um guia detalhado sobre segurança em APIs REST, "
        "incluindo autenticação, autorização, validação de dados, "
        "proteção contra injeção, rate limiting e boas práticas. "
        "Se necessário, pergunte ao pesquisador sobre conceitos "
        "ou exemplos avançados."
    ),
    expected_output=(
        "Documento em Markdown com seções: introdução, "
        "autenticação, autorização, validação, rate limiting "
        "e exemplos de código."
    ),
    agent=agent_writer
)


# --- Crew ---

crew_content = Crew(
    agents=[
        agent_researcher,
        agent_writer,
        agent_reviewer
    ],
    tasks=[
        task_write_guide
    ],
    process=Process.sequential,
    verbose=True
)


if __name__ == "__main__":
    result = crew_content.kickoff()

    print("\nResultado do guia:")
    print(result)