"""Solver registry: category policy (config/categories.yaml `solver:`) -> implementation."""

from __future__ import annotations

from src.core.schemas import Category
from src.solvers.base import BaseSolver, SolveContext
from src.solvers.exec_solvers import CodeDebugSolver, CodeGenSolver, MathSolver
from src.solvers.reasoning_solvers import FactualSolver, LogicSolver
from src.solvers.text_solvers import NERSolver, SentimentSolver, SummarizeSolver

SOLVER_CLASSES: dict[str, type[BaseSolver]] = {
    "math": MathSolver,
    "code_gen": CodeGenSolver,
    "code_debug": CodeDebugSolver,
    "ner": NERSolver,
    "summarize": SummarizeSolver,
    "sentiment": SentimentSolver,
    "logic": LogicSolver,
    "factual": FactualSolver,
}


def make_solvers(ctx: SolveContext) -> dict[Category, BaseSolver]:
    solvers: dict[Category, BaseSolver] = {}
    for category, policy in ctx.config.categories.items():
        cls = SOLVER_CLASSES.get(policy.solver)
        if cls is None:
            raise ValueError(f"category {category.value!r} names unknown solver {policy.solver!r}")
        solvers[category] = cls(policy, ctx)
    return solvers
