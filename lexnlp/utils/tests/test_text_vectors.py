__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"

from unittest import TestCase

import numpy as np

from lexnlp.utils.text_vectors import (
    vectorized_lower,
    vectorized_slice,
    vectorized_startswith,
    vectorized_strip,
    vectorized_substring_count,
)


class TestVectorizedLower(TestCase):
    def test_returns_ndarray(self):
        result = vectorized_lower(["ABC", "Def"])
        self.assertIsInstance(result, np.ndarray)

    def test_lowercases_ascii(self):
        result = vectorized_lower(["ABC", "Def", "gHI"])
        self.assertEqual(list(result), ["abc", "def", "ghi"])

    def test_preserves_unicode(self):
        result = vectorized_lower(["ÁRVORE", "Über", "NAÏVE"])
        self.assertEqual(list(result), ["árvore", "über", "naïve"])

    def test_accepts_generator(self):
        result = vectorized_lower(iter(["A", "B"]))
        self.assertEqual(list(result), ["a", "b"])

    def test_empty_input(self):
        result = vectorized_lower([])
        self.assertEqual(result.shape, (0,))


class TestVectorizedStrip(TestCase):
    def test_strips_whitespace(self):
        result = vectorized_strip(["  abc  ", "\tdef\n", "ghi"])
        self.assertEqual(list(result), ["abc", "def", "ghi"])

    def test_returns_ndarray(self):
        result = vectorized_strip(["a "])
        self.assertIsInstance(result, np.ndarray)

    def test_empty_strings(self):
        result = vectorized_strip(["", "   "])
        self.assertEqual(list(result), ["", ""])


class TestVectorizedStartswith(TestCase):
    def test_returns_boolean_ndarray(self):
        result = vectorized_startswith(["art. 1", "decreto 42", "art. 99"], "art.")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.bool_)

    def test_detects_prefixes(self):
        result = vectorized_startswith(["art. 1", "decreto 42", "art. 99"], "art.")
        self.assertEqual(list(result), [True, False, True])

    def test_unicode_prefix(self):
        result = vectorized_startswith(["nº 10", "No 10", "n. 10"], "nº")
        self.assertEqual(list(result), [True, False, False])

    def test_empty_input(self):
        result = vectorized_startswith([], "anything")
        self.assertEqual(result.shape, (0,))


class TestVectorizedSubstringCount(TestCase):
    def test_counts_substrings(self):
        result = vectorized_substring_count(["abab", "aaa", "bcd"], "a")
        self.assertEqual(list(result), [2, 3, 0])

    def test_returns_integer_ndarray(self):
        result = vectorized_substring_count(["a"], "a")
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

    def test_multi_character_substring(self):
        result = vectorized_substring_count(
            ["art. 5 e art. 7", "sem referências", "art."],
            "art.",
        )
        self.assertEqual(list(result), [2, 0, 1])


class TestVectorizedSlice(TestCase):
    def test_slices_prefix(self):
        result = vectorized_slice(["hello", "world", "xy"], 0, 3)
        self.assertEqual(list(result), ["hel", "wor", "xy"])

    def test_slices_middle(self):
        result = vectorized_slice(["abcdef", "ghijkl"], 2, 4)
        self.assertEqual(list(result), ["cd", "ij"])

    def test_returns_ndarray(self):
        result = vectorized_slice(["abc"], 0, 1)
        self.assertIsInstance(result, np.ndarray)

    def test_stop_past_end_is_safe(self):
        result = vectorized_slice(["abc", "d"], 0, 10)
        self.assertEqual(list(result), ["abc", "d"])

    def test_empty_input(self):
        result = vectorized_slice([], 0, 1)
        self.assertEqual(result.shape, (0,))


# ---------------------------------------------------------------------------
# Additional tests for PR change: StringDType and __array__ protocol
# ---------------------------------------------------------------------------


class TestAsStringArrayInternalBehavior(TestCase):
    """Tests that exercise the PR-specific _as_string_array behaviour."""

    def test_ndarray_input_uses_string_dtype(self):
        """When given an ndarray, _as_string_array must return an array with
        StringDType (not the legacy fixed-width np.str_)."""
        from lexnlp.utils.text_vectors import _as_string_array

        inp = np.array(["hello", "world"])
        out = _as_string_array(inp)
        self.assertIsInstance(out, np.ndarray)
        # StringDType should compare equal to an instance of StringDType
        dtype = out.dtype
        self.assertEqual(dtype.__class__.__name__, "StringDType")
        self.assertTrue("numpy" in dtype.__class__.__module__)

    def test_list_input_uses_string_dtype(self):
        """Plain list input must also yield a StringDType array."""
        from lexnlp.utils.text_vectors import _as_string_array

        out = _as_string_array(["a", "b", "c"])
        dtype = out.dtype
        self.assertEqual(dtype.__class__.__name__, "StringDType")
        self.assertTrue("numpy" in dtype.__class__.__module__)

    def test_object_with_array_protocol_is_accepted(self):
        """Objects that implement __array__ (e.g. pandas Series) must not
        be converted via list() — they should go through np.asarray()."""
        from lexnlp.utils.text_vectors import _as_string_array

        class FakeArrayLike:
            def __array__(self, dtype=None, copy=None):
                return np.array(["x", "y", "z"])

        out = _as_string_array(FakeArrayLike())
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(list(out), ["x", "y", "z"])

    def test_generator_input_materialised_correctly(self):
        """Generator inputs (no __array__) must be materialised via list()."""
        from lexnlp.utils.text_vectors import _as_string_array

        gen = (c for c in ["p", "q", "r"])
        out = _as_string_array(gen)
        self.assertEqual(list(out), ["p", "q", "r"])


class TestVectorizedLowerStringDType(TestCase):
    """Verify vectorized_lower with the new StringDType backend."""

    def test_lowercases_variable_length_unicode(self):
        """StringDType handles variable-length strings without truncation."""
        long_upper = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        short_upper = "Z"
        result = vectorized_lower([long_upper, short_upper])
        self.assertEqual(list(result), [long_upper.lower(), short_upper.lower()])

    def test_accepts_numpy_string_array(self):
        """An np.ndarray of strings (old np.str_ dtype) must be accepted."""
        arr = np.array(["HELLO", "WORLD"], dtype=np.str_)
        result = vectorized_lower(arr)
        self.assertEqual(list(result), ["hello", "world"])


class TestVectorizedStripStringDType(TestCase):
    def test_preserves_length_no_crash_on_unicode_whitespace(self):
        """Vectorized strip must accept Unicode whitespace without crashing.

        Whether ``\\u3000`` (ideographic space) is stripped depends on the
        underlying NumPy ``vectorize_strip`` implementation, which has changed
        between releases. We therefore only assert shape preservation here;
        ASCII whitespace stripping is exercised by ``test_mixed_length_strings``
        below.
        """
        result = vectorized_strip(["\u3000foo\u3000", "bar"])
        # \u3000 is an ideographic space — strip should handle it
        # Result may or may not strip non-ASCII whitespace depending on
        # the NumPy version; just verify no crash and correct length.
        self.assertEqual(len(result), 2)

    def test_mixed_length_strings(self):
        """Strings with very different lengths must all be stripped correctly."""
        data = ["  a  ", "  " + "x" * 100 + "  ", "  b  "]
        result = vectorized_strip(data)
        self.assertEqual(result[0], "a")
        self.assertEqual(result[1], "x" * 100)
        self.assertEqual(result[2], "b")
