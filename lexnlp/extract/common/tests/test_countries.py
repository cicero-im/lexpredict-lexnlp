"""Tests for :mod:`lexnlp.extract.common.countries`."""

from __future__ import annotations

from typing import Any, cast

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.common.countries import (
    CountryInfo,
    currency_codes,
    fuzzy_country,
    is_currency_code,
    is_language_code,
    language_codes,
    lookup_country,
)


class TestLookupCountry:
    def test_alpha_2(self) -> None:
        info = lookup_country("US")
        assert isinstance(info, CountryInfo)
        assert info.alpha_3 == "USA"
        assert info.name == "United States"

    def test_alpha_3(self) -> None:
        info = lookup_country("FRA")
        assert info is not None
        assert info.alpha_2 == "FR"

    def test_name_case_insensitive(self) -> None:
        info = lookup_country("germany")
        assert info is not None
        assert info.alpha_2 == "DE"

    def test_unknown_returns_none(self) -> None:
        assert lookup_country("Narnia") is None

    def test_empty_returns_none(self) -> None:
        assert lookup_country("") is None


class TestFuzzyCountry:
    def test_multiple_matches_when_requested(self) -> None:
        matches = fuzzy_country("United", max_results=3)
        names = [m.name for m in matches]
        # Some variation of "United" exists in multiple names.
        assert any("United" in n for n in names)
        # Honour the explicit cap so noisy fuzzy backends don't surface
        # more candidates than the caller asked for.
        assert len(matches) <= 3

    def test_empty_returns_empty_tuple(self) -> None:
        assert fuzzy_country("") == ()

    def test_no_match_returns_empty_tuple(self) -> None:
        assert fuzzy_country("Valyria") == ()

    def test_invalid_max_results_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="max_results must be a positive integer"):
            fuzzy_country("United", max_results=0)
        with pytest.raises(ValueError, match="max_results must be a positive integer"):
            fuzzy_country("United", max_results=-1)


class TestCurrencyCodes:
    def test_has_usd(self) -> None:
        assert "USD" in currency_codes()
        assert is_currency_code("usd")

    def test_rejects_unknown(self) -> None:
        assert not is_currency_code("XYZ")

    def test_empty_string(self) -> None:
        assert not is_currency_code("")


class TestLanguageCodes:
    def test_has_en(self) -> None:
        assert "en" in language_codes()
        assert is_language_code("EN")

    def test_rejects_unknown(self) -> None:
        assert not is_language_code("zz")


# ---------------------------------------------------------------------------
# Additional tests for PR changes: max_results validation & lru_cache
# ---------------------------------------------------------------------------


class TestFuzzyCountryMaxResultsBoundary:
    """Pin boundary behaviour of the new max_results validation."""

    def test_max_results_one_returns_at_most_one(self) -> None:
        """Default max_results=1 must cap results to a single entry."""
        matches = fuzzy_country("Germany")
        assert len(matches) <= 1

    def test_max_results_two_returns_at_most_two(self) -> None:
        """max_results=2 must cap results to at most two entries."""
        matches = fuzzy_country("United", max_results=2)
        assert len(matches) <= 2

    def test_max_results_one_is_explicit_same_as_default(self) -> None:
        """Explicitly passing max_results=1 is equivalent to the default."""
        default = fuzzy_country("France")
        explicit = fuzzy_country("France", max_results=1)
        assert default == explicit

    def test_max_results_large_does_not_raise(self) -> None:
        """A large max_results value is valid even if fewer results exist."""
        matches = fuzzy_country("France", max_results=1000)
        assert isinstance(matches, tuple)

    def test_max_results_zero_message_includes_value(self) -> None:
        """The ValueError message must include the invalid value for debugging."""
        import pytest

        with pytest.raises(ValueError, match="0"):
            fuzzy_country("United", max_results=0)

    def test_max_results_negative_large_raises(self) -> None:
        """Any negative integer must raise regardless of magnitude."""
        import pytest

        with pytest.raises(ValueError):
            fuzzy_country("United", max_results=-100)

    def test_max_results_non_int_raises_type_error(self) -> None:
        """Non-int values for max_results must raise TypeError, not silently slice."""
        import pytest

        with pytest.raises(TypeError, match="max_results must be an int"):
            fuzzy_country("United", max_results=cast(Any, 1.5))
        with pytest.raises(TypeError, match="max_results must be an int"):
            fuzzy_country("United", max_results=cast(Any, "1"))
        with pytest.raises(TypeError, match="max_results must be an int"):
            # ``bool`` is a subclass of ``int``; reject it explicitly so callers
            # don't accidentally pass ``True`` and silently slice to 1.
            fuzzy_country("United", max_results=cast(Any, True))

    def test_returns_tuple_not_list(self) -> None:
        """Return type must be tuple, not list, to satisfy the type signature."""
        result = fuzzy_country("France")
        assert isinstance(result, tuple)

    def test_each_match_is_country_info(self) -> None:
        """Every element returned must be a CountryInfo dataclass instance."""
        matches = fuzzy_country("Germany", max_results=2)
        for m in matches:
            assert isinstance(m, CountryInfo)
            assert m.alpha_2  # non-empty


class TestLookupCountryAdditional:
    """Additional lookup_country tests for lru_cache switch."""

    def test_returns_country_info_dataclass(self) -> None:
        """lookup_country result must be a CountryInfo with populated fields."""
        info = lookup_country("DE")
        assert info is not None
        assert isinstance(info, CountryInfo)
        assert info.alpha_2 == "DE"
        assert info.alpha_3 == "DEU"
        assert info.name  # non-empty

    def test_whitespace_stripped_before_lookup(self) -> None:
        """Leading/trailing whitespace around the key is normalised."""
        info_plain = lookup_country("US")
        info_padded = lookup_country("  US  ")
        assert info_plain is not None
        assert info_padded is not None
        assert info_padded.alpha_2 == info_plain.alpha_2

    def test_official_name_field_exists(self) -> None:
        """CountryInfo dataclass must expose official_name (str | None)."""
        info = lookup_country("IR")
        assert info is not None
        assert hasattr(info, "official_name")

    def test_result_is_frozen(self) -> None:
        """CountryInfo is a frozen dataclass — mutation must raise."""
        import pytest

        info = lookup_country("GB")
        assert info is not None
        with pytest.raises((AttributeError, TypeError)):
            # setattr() avoids needing a type-checker suppression while still
            # exercising the runtime FrozenInstanceError path.
            setattr(info, "alpha_2", "XX")  # noqa: B010 - intentional dynamic set
