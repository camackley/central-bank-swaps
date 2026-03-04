"""Benchmark harness — load labeled data and measure classification accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openpyxl
from langchain_core.language_models.chat_models import BaseChatModel

from cbs.pipeline.classifier import classify_press_release

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# (filename, sheet, swap_col_index, text_col_index, id_col_index, source_label)
_EXCEL_SOURCES: list[tuple[str, str, int, int, int, str]] = [
    ("fed_events_review.xlsx", "process_press", 4, 2, 0, "fed"),
    ("ecb_review.xlsx", "CORRECT", 5, 4, 0, "ecb"),
    ("pboc_review.xlsx", "swaps", 5, 4, 0, "pboc"),
]


@dataclass
class LabeledSample:
    """A labeled press release for benchmarking."""

    id: str
    text: str
    is_swap_related: bool
    source: str  # "fed" | "ecb" | "pboc"


@dataclass
class BenchmarkResult:
    """Confusion matrix from a classification benchmark run."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0


def load_labeled_dataset() -> list[LabeledSample]:
    """Load labeled press releases from data/*.xlsx, deduplicated by row_id."""
    samples: dict[str, LabeledSample] = {}

    for filename, sheet, swap_col, text_col, id_col, source in _EXCEL_SOURCES:
        path = _DATA_DIR / filename
        if not path.exists():
            continue

        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb[sheet]

        for row in ws.iter_rows(min_row=2, values_only=True):
            row_id = row[id_col]
            if row_id is None or row_id in samples:
                continue

            text = row[text_col]
            is_swap = row[swap_col]

            if text is None or is_swap is None:
                continue

            # Normalize is_swap_related to bool
            if isinstance(is_swap, bool):
                swap_bool = is_swap
            else:
                swap_bool = str(is_swap).strip().lower() == "true"

            samples[str(row_id)] = LabeledSample(
                id=str(row_id),
                text=str(text),
                is_swap_related=swap_bool,
                source=source,
            )

        wb.close()

    return list(samples.values())


def run_classification_benchmark(
    llm: BaseChatModel,
    dataset: list[LabeledSample],
) -> BenchmarkResult:
    """Run the classifier on each sample and compute confusion matrix."""
    tp = fp = tn = fn = 0

    for sample in dataset:
        result = classify_press_release(llm, sample.text)
        predicted = result.is_swap_related
        actual = sample.is_swap_related

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    return BenchmarkResult(tp=tp, fp=fp, tn=tn, fn=fn)
