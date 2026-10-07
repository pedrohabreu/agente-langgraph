from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.common import IsInstance


case1 = Case(
    name="capital_franca",
    inputs="Qual é a capital da França?",
    expected_output="Paris",
    metadata={"tipo": "fácil"},
)

dataset = Dataset(
    name="teste_capitais",
    cases=[case1]
)

dataset.add_evaluator(
    IsInstance(type_name="str")
)

print(dataset)