"""Classification accuracy benchmarks — Slices 1.5.1 & 1.5.2.

These tests hit real LLM APIs and are excluded from the default pytest run.
Run explicitly with: pytest tests/benchmarks/ -m benchmark
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from cbs.llm.provider import get_llm

from .harness import load_labeled_dataset, run_classification_benchmark


@pytest.fixture(scope="module")
def dataset() -> list:
    return load_labeled_dataset()


@pytest.fixture(scope="module")
def anthropic_llm() -> BaseChatModel:
    return get_llm("anthropic", "claude-sonnet-4-20250514")


@pytest.fixture(scope="module")
def anthropic_result(anthropic_llm: BaseChatModel, dataset: list) -> object:
    return run_classification_benchmark(anthropic_llm, dataset)


@pytest.mark.benchmark
class TestPrecisionAbove95Percent:
    def test_precision_above_95_percent(self, anthropic_result: object) -> None:
        assert anthropic_result.precision >= 0.95, (
            f"Precision {anthropic_result.precision:.2%} below 95% threshold. "
            f"TP={anthropic_result.tp}, FP={anthropic_result.fp}"
        )


@pytest.mark.benchmark
class TestRecallAbove90Percent:
    def test_recall_above_90_percent(self, anthropic_result: object) -> None:
        assert anthropic_result.recall >= 0.90, (
            f"Recall {anthropic_result.recall:.2%} below 90% threshold. "
            f"TP={anthropic_result.tp}, FN={anthropic_result.fn}"
        )


@pytest.mark.benchmark
class TestBenchmarkRunsAcrossTwoProviders:
    @pytest.mark.parametrize(
        "provider,model",
        [
            ("anthropic", "claude-sonnet-4-20250514"),
            ("openai", "gpt-4o-mini"),
        ],
    )
    def test_benchmark_produces_valid_result(
        self, provider: str, model: str, dataset: list
    ) -> None:
        llm = get_llm(provider, model)
        result = run_classification_benchmark(llm, dataset)

        assert result.tp + result.fp + result.tn + result.fn == len(dataset)
        assert result.precision >= 0.0
        assert result.recall >= 0.0


@pytest.mark.benchmark
class TestCheapClassifierAccuracy:
    """Verify accuracy with a cheaper classification model (1.5.2)."""

    @pytest.fixture(scope="class")
    def cheap_result(self, dataset: list) -> object:
        llm = get_llm("anthropic", "claude-haiku-3-5")
        return run_classification_benchmark(llm, dataset)

    def test_cheap_classifier_precision_above_90(self, cheap_result: object) -> None:
        assert cheap_result.precision >= 0.90, (
            f"Cheap model precision {cheap_result.precision:.2%} below 90%. "
            f"TP={cheap_result.tp}, FP={cheap_result.fp}"
        )

    def test_cheap_classifier_recall_above_85(self, cheap_result: object) -> None:
        assert cheap_result.recall >= 0.85, (
            f"Cheap model recall {cheap_result.recall:.2%} below 85%. "
            f"TP={cheap_result.tp}, FN={cheap_result.fn}"
        )
