from crewai import Agent, Task, Crew, Process, LLM


llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434"
)


manager = Agent(
    role="Gerente de Conteúdo",
    goal=(
        "Supervisionar a criação de conteúdo técnico "
        "com qualidade e profundidade"
    ),
    backstory=(
        "Você coordena equipes de documentação, "
        "guias de estilo e padronização."
    ),
    allow_delegation=False,
    verbose=True,
    max_iter=2,
    llm=llm
)


researcher = Agent(
    role="Pesquisador de Domínio",
    goal=(
        "Buscar informações técnicas precisas "
        "e confiáveis"
    ),
    backstory=(
        "Você tem experiência acadêmica e técnica, "
        "conhece documentação oficial, artigos e RFCs."
    ),
    allow_delegation=False,
    verbose=True,
    max_iter=2,
    llm=llm
)


writer = Agent(
    role="Redator",
    goal=(
        "Produzir texto claro e acessível "
        "a partir dos resultados da pesquisa"
    ),
    backstory=(
        "Você escreve guias e tutoriais com "
        "exemplos práticos e boas práticas."
    ),
    allow_delegation=False,
    verbose=True,
    max_iter=2,
    llm=llm
)


reviewer = Agent(
    role="Revisor",
    goal=(
        "Garantir consistência técnica e estilo "
        "no conteúdo final"
    ),
    backstory=(
        "Você revisa documentação técnica e verifica "
        "terminologia, clareza e coesão."
    ),
    allow_delegation=False,
    verbose=True,
    max_iter=2,
    llm=llm
)


task_plan = Task(
    description=(
        "Gere um plano curto de tópicos para um guia "
        "sobre segurança em APIs REST."
    ),
    expected_output=(
        "Estrutura com introdução, autenticação, "
        "autorização e boas práticas."
    ),
    agent=manager
)


task_research = Task(
    description=(
        "Com base no plano, reúna os principais pontos "
        "técnicos necessários para cada seção."
    ),
    expected_output=(
        "Resumo técnico organizado por tópico."
    ),
    agent=researcher,
    context=[task_plan]
)


task_write = Task(
    description=(
        "Escreva um guia técnico curto com base "
        "na pesquisa recebida."
    ),
    expected_output=(
        "Texto em Markdown, claro e objetivo."
    ),
    agent=writer,
    context=[task_research]
)


task_review = Task(
    description=(
        "Revise o guia garantindo clareza, "
        "consistência e ausência de redundâncias."
    ),
    expected_output=(
        "Versão final pronta para publicação."
    ),
    agent=reviewer,
    context=[task_write]
)


crew_docs = Crew(
    agents=[
        manager,
        researcher,
        writer,
        reviewer
    ],
    tasks=[
        task_plan,
        task_research,
        task_write,
        task_review
    ],
    process=Process.sequential,
    verbose=True
)


if __name__ == "__main__":
    resultado = crew_docs.kickoff()

    print("\nGuia finalizado:")
    print(resultado)