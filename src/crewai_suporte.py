from crewai import Agent, Task, Crew, LLM


# --- LLM local via Ollama ---
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434"
)


# --- Agentes ---

agent_triagem = Agent(
    role="Especialista de Triagem de Suporte",
    goal=(
        "Classificar o tipo de solicitação do cliente "
        "e encaminhar para o agente adequado"
    ),
    backstory=(
        "Você é um agente experiente em atendimento técnico. "
        "Já trabalhou muitos anos triando solicitações de software, "
        "hardware e conta de usuário. Seu foco é determinar "
        "rapidamente a categoria e a urgência do chamado."
    ),
    llm=llm
)


agent_tecnico = Agent(
    role="Técnico de Suporte",
    goal=(
        "Resolver problemas técnicos de software "
        "e configuração de aplicativos"
    ),
    backstory=(
        "Você é especializado em sistemas operacionais, "
        "software instalado, configurações e diagnósticos. "
        "Lida frequentemente com logs, mensagens de erro "
        "e procedimentos de depuração."
    ),
    llm=llm
)


agent_escalonamento = Agent(
    role="Especialista em Escalonamento",
    goal=(
        "Atuar nos casos mais complexos não resolvidos "
        "pelos agentes técnicos"
    ),
    backstory=(
        "Você atua em casos críticos, problemas de infraestrutura "
        "e situações que envolvem múltiplos sistemas. "
        "Você revisa diagnósticos anteriores e decide próximos passos."
    ),
    llm=llm
)


# --- Tarefas ---

task_triagem = Task(
    description=(
        "Receba a solicitação do cliente. "
        "Classifique como 'software', 'hardware', "
        "'conta de usuário' ou 'outro'. "
        "Determine a urgência: baixa, média ou alta. "
        "Indique para qual agente o caso deve seguir."
    ),
    expected_output=(
        "Classificação contendo categoria, urgência "
        "e encaminhamento recomendado."
    ),
    agent=agent_triagem
)


task_resolucao_tecnica = Task(
    description=(
        "Analise o problema técnico apresentado. "
        "Para solicitações de software ou hardware, "
        "proponha um plano de resolução com passos claros."
    ),
    expected_output=(
        "Plano técnico de resolução com passos, "
        "riscos e orientação final."
    ),
    agent=agent_tecnico
)


task_escalonamento = Task(
    description=(
        "Analise o caso considerando o diagnóstico anterior. "
        "Decida se é possível resolver, se deve envolver "
        "um especialista humano ou se precisa de outro recurso."
    ),
    expected_output=(
        "Recomendação final sobre resolução ou escalonamento."
    ),
    agent=agent_escalonamento
)


# --- Crew ---

crew = Crew(
    agents=[
        agent_triagem,
        agent_tecnico,
        agent_escalonamento
    ],
    tasks=[
        task_triagem,
        task_resolucao_tecnica,
        task_escalonamento
    ],
    verbose=True
)


# --- Execução ---

if __name__ == "__main__":
    mensagem = (
        "Meu aplicativo está travando depois da última atualização. "
        "Aparece uma mensagem de erro ao abrir. "
        "O dispositivo é Android."
    )

    resultado = crew.kickoff(
        inputs={
            "mensagem_cliente": mensagem
        }
    )

    print("\nResposta final:")
    print(resultado)