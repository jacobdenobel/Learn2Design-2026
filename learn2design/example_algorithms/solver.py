# Reference: https://github.com/artificial-scientist-lab/Differometor-Benchmark/blob/main/src/dfbench/algorithms/evolutionary/random_search.py
from jaxtyping import Array, Float

from dfbench import Objective, OptimizationAlgorithm


class Solver(OptimizationAlgorithm):
    """Uniform random sampling within the problem bounds (batched evaluation)."""

    algorithm_str = "solver"

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
    ) -> None:
        obj = objective
        # Random search operates directly in bounded physical space.
        self.prepare(obj, unbounded=False, random_seed=random_seed)

        obj.warmup_value()
        obj.start_logging()

        # Repeatedly draw random params from the active (bounded) space and
        # evaluate them until the budget is exhausted. Each `obj.value()` call
        # is automatically logged.
        while not obj.budget_exceeded:
            obj.value(obj.random_params())