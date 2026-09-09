"""Coverage tests for money list-wrapper helpers."""

from decimal import Decimal

import pytest

from lexnlp.extract.common.annotations.money_annotation import MoneyAnnotation
from lexnlp.extract.en.amounts import get_amount_list
from lexnlp.extract.en.money import (
    get_money,
    get_money_annotation_list,
    get_money_annotations,
    get_money_list,
)


def test_get_money_list_returns_amounts():
    text = "The cost is $25.50."
    assert get_money_list(text) == [(Decimal("25.50"), "USD")]
    assert get_money_list(text) == list(get_money(text))


def test_get_money_list_with_sources():
    actual = get_money_list("The cost is $25.50.", return_sources=True)
    assert actual == [(Decimal("25.50"), "USD", "cost is $25.50")]
    assert actual == list(get_money("The cost is $25.50.", return_sources=True))


def test_get_money_list_empty_without_match():
    assert get_money_list("no money here xyz") == []


def test_get_money_annotation_list_returns_annotations():
    text = "The cost is $25.50."
    expected = list(get_money_annotations(text))
    actual = get_money_annotation_list(text)
    assert isinstance(actual, list)
    assert len(actual) == len(expected) == 1
    assert all(isinstance(item, MoneyAnnotation) for item in actual)
    annotation = actual[0]
    assert annotation.amount == Decimal("25.50")
    assert annotation.currency == "USD"
    # (3, 19) includes the trailing "." exactly as the ``$``-prefix branch does
    # for "The fee is $25.50." -- ``.text`` strips it either way. The old (3, 18)
    # was an artifact of the trigger branch's narrow number pattern, which could
    # not consume the trailing \W and truncated grouped amounts as a result.
    assert annotation.coords == (3, 19)
    assert annotation.text == "cost is $25.50"


def test_get_money_annotation_list_empty_without_match():
    assert get_money_annotation_list("no money here xyz") == []


class TestTriggerWordDoesNotTruncateGroupedAmounts:
    """The trigger-word branch of the currency pattern carried its own number
    sub-pattern, ``\\d+(?:\\.\\d{1,16})?``, which cannot span a thousands
    separator. Because "price"/"cost" occur BEFORE the currency symbol, that
    branch matched further left than the ``$`` branch and won the leftmost
    match, so "The price is $1,500,000.00" reported 1.0 -- silently, and with
    the right currency, which is the worst way to be wrong.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The price is $1,500,000.00", Decimal("1500000.00")),
            ("The price is $250,000.00", Decimal("250000.00")),
            ("The price is $1,234.56", Decimal("1234.56")),
            ("The cost is $1,500,000.00", Decimal("1500000.00")),
            ("The purchase price of $12,345.67 is due", Decimal("12345.67")),
            ("The cost of $9,000 shall be borne by Buyer", Decimal("9000")),
        ],
    )
    def test_grouped_amount_survives_a_trigger_word(self, text: str, expected: Decimal) -> None:
        amounts = [amount for amount, _currency in get_money_list(text)]
        assert amounts, f"no money extracted from {text!r}"
        assert amounts[0] == expected

    def test_amounts_and_money_agree_on_the_same_text(self) -> None:
        """get_amounts never had the defect, so the two extractors disagreeing
        on one string is itself the signal."""
        text = "The price is $1,500,000.00"
        assert [a for a, _ in get_money_list(text)] == get_amount_list(text)

    def test_trigger_word_without_a_currency_symbol_still_works(self) -> None:
        """The branch exists so a bare "price is 25.50" is still money; that
        behaviour must survive the fix."""
        assert get_money_list("The price is 25.50") == [(Decimal("25.50"), "USD")]
