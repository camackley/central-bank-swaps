"""Tests for the swap classifier pipeline — Slice 1.5 (FR-004)."""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

from cbs.pipeline.classifier import ClassificationResult, classify_press_release

# ---------------------------------------------------------------------------
# Sample press-release texts used across tests
# ---------------------------------------------------------------------------

SWAP_TEXT = (
    "The Federal Reserve and the European Central Bank today announced "
    "the establishment of a bilateral currency swap arrangement. Under "
    "this agreement, the Fed will provide U.S. dollars to the ECB in "
    "exchange for euros, with the aim of improving liquidity conditions "
    "in financial markets. The swap line will be available for up to "
    "$50 billion and will remain in effect until further notice."
)

NON_SWAP_TEXT = (
    "The Federal Reserve decided today to raise the target range for "
    "the federal funds rate by 25 basis points, to a range of 5.25 to "
    "5.50 percent. The Committee continues to assess additional "
    "information and its implications for monetary policy."
)

AMBIGUOUS_TEXT = (
    "The Bank of Japan today released its quarterly economic outlook. "
    "The report notes that global financial conditions have remained "
    "stable, with existing swap lines between major central banks "
    "continuing to provide a backstop for liquidity. The BOJ's primary "
    "focus remains on yield curve control and inflation targeting."
)


def _make_mock_llm(*, is_swap_related: bool, reason: str) -> MagicMock:
    """Return a mock LLM with canned structured output."""
    result = ClassificationResult(
        is_swap_related=is_swap_related,
        reason=reason,
    )

    mock_structured = MagicMock()
    mock_structured.invoke.return_value = result

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


class TestSwapPressReleaseClassifiedTrue:
    """A press release about a swap agreement must be classified as swap-related."""

    def test_swap_text_returns_true(self) -> None:
        llm = _make_mock_llm(
            is_swap_related=True,
            reason=(
                "Announces a bilateral currency swap "
                "arrangement between the Fed and ECB."
            ),
        )
        result = classify_press_release(llm, SWAP_TEXT)
        assert result.is_swap_related is True


class TestNonSwapPressReleaseClassifiedFalse:
    """A press release about monetary policy (not swaps) must be classified False."""

    def test_non_swap_text_returns_false(self) -> None:
        llm = _make_mock_llm(
            is_swap_related=False,
            reason=(
                "This press release discusses a federal funds "
                "rate decision, not a swap agreement."
            ),
        )
        result = classify_press_release(llm, NON_SWAP_TEXT)
        assert result.is_swap_related is False


class TestClassificationIncludesReason:
    """Every classification result must include a human-readable reason."""

    def test_true_classification_has_reason(self) -> None:
        reason = "Announces a new bilateral swap line."
        llm = _make_mock_llm(is_swap_related=True, reason=reason)
        result = classify_press_release(llm, SWAP_TEXT)
        assert result.reason == reason

    def test_false_classification_has_reason(self) -> None:
        reason = "Discusses interest rate policy, not swap agreements."
        llm = _make_mock_llm(is_swap_related=False, reason=reason)
        result = classify_press_release(llm, NON_SWAP_TEXT)
        assert result.reason == reason


class TestClassificationOutputSchemaValid:
    """ClassificationResult must be a Pydantic model with the correct fields."""

    def test_is_pydantic_base_model(self) -> None:
        assert issubclass(ClassificationResult, BaseModel)

    def test_has_is_swap_related_bool(self) -> None:
        result = ClassificationResult(is_swap_related=True, reason="test")
        assert isinstance(result.is_swap_related, bool)

    def test_has_reason_str(self) -> None:
        result = ClassificationResult(is_swap_related=False, reason="test reason")
        assert isinstance(result.reason, str)

    def test_schema_fields(self) -> None:
        fields = set(ClassificationResult.model_fields.keys())
        assert fields == {"is_swap_related", "reason"}


class TestAmbiguousMentionClassifiedFalse:
    """A passing mention of swaps (not the primary topic) must be classified False."""

    def test_ambiguous_swap_mention_returns_false(self) -> None:
        llm = _make_mock_llm(
            is_swap_related=False,
            reason=(
                "The press release primarily discusses the BOJ's economic outlook "
                "and monetary policy. Swap lines are mentioned only in passing as "
                "background context, not as the primary topic."
            ),
        )
        result = classify_press_release(llm, AMBIGUOUS_TEXT)
        assert result.is_swap_related is False
