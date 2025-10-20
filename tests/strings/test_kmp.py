"""
Unit tests for KMP (Knuth-Morris-Pratt) Pattern Matching Algorithm.

Test coverage:
- LPS array computation
- Pattern searching (all occurrences)
- First occurrence search
- Pattern counting
- Edge cases and error handling
"""

import pytest
from algorithms.strings.kmp import (
    compute_lps_array,
    kmp_search,
    kmp_search_first,
    kmp_count
)


class TestComputeLPS:
    """Test LPS (Longest Proper Prefix which is also Suffix) array computation."""

    def test_basic_lps(self) -> None:
        """Test basic LPS computation."""
        lps = compute_lps_array("ABABCABAB")
        assert lps == [0, 0, 1, 2, 0, 1, 2, 3, 4]

    def test_no_prefix_suffix_match(self) -> None:
        """Test pattern with no prefix-suffix matches."""
        lps = compute_lps_array("ABCDE")
        assert lps == [0, 0, 0, 0, 0]

    def test_all_same_characters(self) -> None:
        """Test pattern with all same characters."""
        lps = compute_lps_array("AAAA")
        assert lps == [0, 1, 2, 3]

    def test_repeating_pattern(self) -> None:
        """Test repeating pattern."""
        lps = compute_lps_array("ABCABC")
        assert lps == [0, 0, 0, 1, 2, 3]

    def test_single_character(self) -> None:
        """Test single character pattern."""
        lps = compute_lps_array("A")
        assert lps == [0]

    def test_two_characters_same(self) -> None:
        """Test two same characters."""
        lps = compute_lps_array("AA")
        assert lps == [0, 1]

    def test_two_characters_different(self) -> None:
        """Test two different characters."""
        lps = compute_lps_array("AB")
        assert lps == [0, 0]

    def test_empty_pattern(self) -> None:
        """Test empty pattern."""
        lps = compute_lps_array("")
        assert lps == []


class TestKMPSearch:
    """Test KMP pattern searching."""

    def test_single_match(self) -> None:
        """Test finding single occurrence."""
        text = "ABABDABACDABABCABAB"
        pattern = "ABABCABAB"
        matches = kmp_search(text, pattern)

        assert matches == [10]

    def test_multiple_matches(self) -> None:
        """Test finding multiple occurrences."""
        text = "AAAAAAA"
        pattern = "AAA"
        matches = kmp_search(text, pattern)

        assert matches == [0, 1, 2, 3, 4]

    def test_no_match(self) -> None:
        """Test when pattern not found."""
        text = "HELLO WORLD"
        pattern = "XYZ"
        matches = kmp_search(text, pattern)

        assert matches == []

    def test_pattern_at_start(self) -> None:
        """Test pattern at beginning of text."""
        text = "ABCDEFGH"
        pattern = "ABC"
        matches = kmp_search(text, pattern)

        assert matches == [0]

    def test_pattern_at_end(self) -> None:
        """Test pattern at end of text."""
        text = "ABCDEFGH"
        pattern = "FGH"
        matches = kmp_search(text, pattern)

        assert matches == [5]

    def test_pattern_equals_text(self) -> None:
        """Test when pattern equals entire text."""
        text = "HELLO"
        pattern = "HELLO"
        matches = kmp_search(text, pattern)

        assert matches == [0]

    def test_pattern_longer_than_text(self) -> None:
        """Test when pattern is longer than text."""
        text = "HI"
        pattern = "HELLO"
        matches = kmp_search(text, pattern)

        assert matches == []

    def test_empty_text(self) -> None:
        """Test with empty text."""
        text = ""
        pattern = "ABC"
        matches = kmp_search(text, pattern)

        assert matches == []

    def test_empty_pattern_raises_error(self) -> None:
        """Test that empty pattern raises ValueError."""
        text = "HELLO"
        pattern = ""

        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            kmp_search(text, pattern)

    def test_overlapping_matches(self) -> None:
        """Test finding overlapping matches."""
        text = "AAAA"
        pattern = "AA"
        matches = kmp_search(text, pattern)

        assert matches == [0, 1, 2]

    def test_case_sensitive(self) -> None:
        """Test that search is case-sensitive."""
        text = "Hello World"
        pattern = "hello"
        matches = kmp_search(text, pattern)

        assert matches == []

    def test_special_characters(self) -> None:
        """Test with special characters."""
        text = "abc@123#def@123#ghi"
        pattern = "@123#"
        matches = kmp_search(text, pattern)

        assert matches == [3, 11]

    def test_numbers(self) -> None:
        """Test with numeric strings."""
        text = "123451234567890"
        pattern = "12345"
        matches = kmp_search(text, pattern)

        assert matches == [0, 5]

    def test_unicode_characters(self) -> None:
        """Test with Unicode characters."""
        text = "hello🌍world🌍test"
        pattern = "🌍"
        matches = kmp_search(text, pattern)

        assert len(matches) == 2


class TestKMPSearchFirst:
    """Test KMP first occurrence search."""

    def test_find_first_occurrence(self) -> None:
        """Test finding first occurrence."""
        text = "ABABDABACDABABCABAB"
        pattern = "ABAB"
        index = kmp_search_first(text, pattern)

        assert index == 0

    def test_first_not_at_start(self) -> None:
        """Test when first occurrence is not at start."""
        text = "XYZABCABC"
        pattern = "ABC"
        index = kmp_search_first(text, pattern)

        assert index == 3

    def test_no_match_returns_minus_one(self) -> None:
        """Test that no match returns -1."""
        text = "HELLO WORLD"
        pattern = "XYZ"
        index = kmp_search_first(text, pattern)

        assert index == -1

    def test_empty_text_returns_minus_one(self) -> None:
        """Test empty text returns -1."""
        text = ""
        pattern = "ABC"
        index = kmp_search_first(text, pattern)

        assert index == -1

    def test_empty_pattern_raises_error(self) -> None:
        """Test that empty pattern raises ValueError."""
        text = "HELLO"
        pattern = ""

        with pytest.raises(ValueError):
            kmp_search_first(text, pattern)


class TestKMPCount:
    """Test KMP pattern counting."""

    def test_count_multiple_occurrences(self) -> None:
        """Test counting multiple occurrences."""
        text = "AAAAAAA"
        pattern = "AAA"
        count = kmp_count(text, pattern)

        assert count == 5

    def test_count_single_occurrence(self) -> None:
        """Test counting single occurrence."""
        text = "HELLO WORLD"
        pattern = "WORLD"
        count = kmp_count(text, pattern)

        assert count == 1

    def test_count_no_occurrences(self) -> None:
        """Test counting with no occurrences."""
        text = "HELLO WORLD"
        pattern = "XYZ"
        count = kmp_count(text, pattern)

        assert count == 0

    def test_count_empty_text(self) -> None:
        """Test counting in empty text."""
        text = ""
        pattern = "ABC"
        count = kmp_count(text, pattern)

        assert count == 0

    def test_count_empty_pattern_raises_error(self) -> None:
        """Test that empty pattern raises ValueError."""
        text = "HELLO"
        pattern = ""

        with pytest.raises(ValueError):
            kmp_count(text, pattern)


class TestKMPApplications:
    """Test real-world applications of KMP."""

    def test_dna_sequence_search(self) -> None:
        """Test DNA sequence searching."""
        dna = "ATCGATCGATCGTAGCTAGCTAGC"
        motif = "ATCG"
        matches = kmp_search(dna, motif)

        assert len(matches) == 3
        assert matches[0] == 0

    def test_word_search_in_sentence(self) -> None:
        """Test word searching in text."""
        text = "the quick brown fox jumps over the lazy dog"
        pattern = "the"
        matches = kmp_search(text, pattern)

        assert len(matches) == 2
        assert matches == [0, 31]

    def test_log_file_pattern_search(self) -> None:
        """Test log file pattern matching."""
        log = "ERROR: Failed\nINFO: Success\nERROR: Timeout\nWARNING: Slow"
        pattern = "ERROR"
        matches = kmp_search(log, pattern)

        assert len(matches) == 2

    def test_repeated_substring_detection(self) -> None:
        """Test detecting repeated substrings."""
        text = "abcabcabcabc"
        pattern = "abc"
        count = kmp_count(text, pattern)

        assert count == 4


class TestKMPPerformance:
    """Test performance characteristics of KMP."""

    def test_long_text_short_pattern(self) -> None:
        """Test with long text and short pattern."""
        text = "A" * 10000 + "B"
        pattern = "AB"
        matches = kmp_search(text, pattern)

        assert len(matches) == 1
        assert matches[0] == 9999

    def test_long_repeating_pattern(self) -> None:
        """Test with long repeating pattern."""
        text = "ABC" * 100
        pattern = "ABCABC"
        matches = kmp_search(text, pattern)

        # Should find many overlapping matches
        assert len(matches) > 0

    def test_worst_case_mismatch(self) -> None:
        """Test worst case with many mismatches."""
        text = "A" * 1000 + "B"
        pattern = "AAAB"
        matches = kmp_search(text, pattern)

        assert len(matches) == 1

    def test_many_partial_matches(self) -> None:
        """Test with many partial matches."""
        text = "ABABABABABABABABCABAB"
        pattern = "ABABCABAB"
        matches = kmp_search(text, pattern)

        assert len(matches) == 1


class TestKMPEdgeCases:
    """Test edge cases for KMP."""

    def test_single_char_text_and_pattern(self) -> None:
        """Test with single character text and pattern."""
        text = "A"
        pattern = "A"
        matches = kmp_search(text, pattern)

        assert matches == [0]

    def test_single_char_text_no_match(self) -> None:
        """Test with single character text, no match."""
        text = "A"
        pattern = "B"
        matches = kmp_search(text, pattern)

        assert matches == []

    def test_whitespace_pattern(self) -> None:
        """Test searching for whitespace."""
        text = "hello world test"
        pattern = " "
        matches = kmp_search(text, pattern)

        assert matches == [5, 11]

    def test_newline_pattern(self) -> None:
        """Test searching for newlines."""
        text = "line1\nline2\nline3"
        pattern = "\n"
        matches = kmp_search(text, pattern)

        assert matches == [5, 11]

    def test_pattern_with_repeated_prefix(self) -> None:
        """Test pattern with repeated prefix."""
        text = "ABABABABABCABABABAB"
        pattern = "ABABABAB"
        matches = kmp_search(text, pattern)

        assert len(matches) >= 1


class TestKMPComparison:
    """Test KMP against expected naive approach results."""

    def test_matches_naive_search(self) -> None:
        """Test that KMP gives same results as naive search."""
        text = "ABABDABACDABABCABAB"
        pattern = "ABAB"

        # KMP result
        kmp_matches = kmp_search(text, pattern)

        # Naive search result
        naive_matches = []
        for i in range(len(text) - len(pattern) + 1):
            if text[i:i + len(pattern)] == pattern:
                naive_matches.append(i)

        assert kmp_matches == naive_matches

    def test_first_match_consistency(self) -> None:
        """Test that kmp_search_first matches first from kmp_search."""
        text = "ABABDABACDABABCABAB"
        pattern = "ABAB"

        all_matches = kmp_search(text, pattern)
        first_match = kmp_search_first(text, pattern)

        if all_matches:
            assert first_match == all_matches[0]
        else:
            assert first_match == -1

    def test_count_matches_length(self) -> None:
        """Test that kmp_count matches length of kmp_search."""
        text = "AAAAAAA"
        pattern = "AAA"

        matches = kmp_search(text, pattern)
        count = kmp_count(text, pattern)

        assert count == len(matches)
