import time


class TimeoutError(Exception):
    pass


def monitor_execution(task_fn, timeout_secs=2):
    inicio = time.time()

    resultado = task_fn()

    tempo = time.time() - inicio

    if tempo > timeout_secs:
        raise TimeoutError(
            f"Tarefa excedeu {timeout_secs}s. "
            f"Tempo total: {tempo:.2f}s"
        )

    return resultado


def tarefa_principal():
    print("Executando tarefa principal...")

    # Simula uma tarefa muito lenta
    time.sleep(3)

    return "Relatório detalhado concluído."


def tarefa_fallback():
    print("Executando fallback...")

    return (
        "Resumo executivo produzido pelo fluxo alternativo."
    )


def executar():
    try:
        return monitor_execution(
            tarefa_principal,
            timeout_secs=2
        )

    except TimeoutError as erro:
        print(f"\nTimeout detectado: {erro}")
        return tarefa_fallback()


if __name__ == "__main__":
    resultado = executar()

    print("\nResultado final:")
    print(resultado)