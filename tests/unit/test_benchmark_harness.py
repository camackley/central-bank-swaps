"""Unit tests for the benchmark harness — Slice 1.5.1."""

from __future__ import annotations

from tests.benchmarks.harness import (
    BenchmarkResult,
    LabeledSample,
    load_labeled_dataset,
)


class TestBenchmarkResultMath:
    def test_precision_all_correct(self) -> None:
        result = BenchmarkResult(tp=10, fp=0, tn=20, fn=0)
        assert result.precision == 1.0

    def test_recall_all_correct(self) -> None:
        result = BenchmarkResult(tp=10, fp=0, tn=20, fn=0)
        assert result.recall == 1.0

    def test_precision_with_false_positives(self) -> None:
        result = BenchmarkResult(tp=8, fp=2, tn=18, fn=0)
        assert result.precision == 0.8

    def test_recall_with_false_negatives(self) -> None:
        result = BenchmarkResult(tp=9, fp=0, tn=20, fn=1)
        assert result.recall == 0.9

    def test_precision_zero_when_no_positives(self) -> None:
        result = BenchmarkResult(tp=0, fp=0, tn=20, fn=5)
        assert result.precision == 0.0

    def test_recall_zero_when_no_true_positives(self) -> None:
        result = BenchmarkResult(tp=0, fp=5, tn=20, fn=10)
        assert result.recall == 0.0


class TestLoadLabeledDataset:
    def test_loads_nonzero_samples(self) -> None:
        dataset = load_labeled_dataset()
        assert len(dataset) > 50

    def test_samples_have_required_fields(self) -> None:
        dataset = load_labeled_dataset()
        sample = dataset[0]
        assert isinstance(sample, LabeledSample)
        assert isinstance(sample.id, str)
        assert isinstance(sample.text, str)
        assert len(sample.text) > 0
        assert isinstance(sample.is_swap_related, bool)
        assert sample.source in ("fed", "ecb", "pboc")

    def test_dataset_has_both_classes(self) -> None:
        dataset = load_labeled_dataset()
        swap_count = sum(1 for s in dataset if s.is_swap_related)
        non_swap_count = sum(1 for s in dataset if not s.is_swap_related)
        assert swap_count > 0
        assert non_swap_count > 0

    def test_no_duplicate_ids(self) -> None:
        dataset = load_labeled_dataset()
        ids = [s.id for s in dataset]
        assert len(ids) == len(set(ids))
