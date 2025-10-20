"""
Unit tests for Longest Common Subsequence (LCS) algorithm.

Test coverage:
- LCS length calculation
- LCS string extraction
- Finding all LCS
- Space-optimized version
- Generic sequence support
- Diff operations
"""

import pytest
from algorithms.dynamic_programming.lcs import (
    lcs_length,
    lcs_string,
    lcs_all,
    lcs_length_space_optimized,
    lcs_generic,
    diff_strings
)


class TestLCSLength:
    """Test LCS length calculation."""

    def test_identical_strings(self) -> None:
        """Test that identical strings have LCS equal to their length."""
        assert lcs_length("hello", "hello") == 5
        assert lcs_length("test", "test") == 4
        assert lcs_length("", "") == 0

    def test_no_common_subsequence(self) -> None:
        """Test strings with no common characters."""
        assert lcs_length("abc", "xyz") == 0
        assert lcs_length("123", "abc") == 0

    def test_classic_examples(self) -> None:
        """Test classic LCS examples."""
        assert lcs_length("ABCDGH", "AEDFHR") == 3  # "ADH"
        assert lcs_length("AGGTAB", "GXTXAYB") == 4  # "GTAB"

    def test_one_substring_of_other(self) -> None:
        """Test when one string is substring of another."""
        assert lcs_length("abc", "aabbcc") == 3
        assert lcs_length("test", "tteesstt") == 4

    def test_partial_overlap(self) -> None:
        """Test strings with partial overlap."""
        assert lcs_length("abc", "bcd") == 2  # "bc"
        assert lcs_length("hello", "yellow") == 4  # "ello"


class TestLCSString:
    """Test LCS string extraction."""

    def test_returns_correct_lcs(self) -> None:
        """Test that correct LCS string is returned."""
        assert lcs_string("ABCDGH", "AEDFHR") == "ADH"
        assert lcs_string("AGGTAB", "GXTXAYB") == "GTAB"

    def test_identical_strings(self) -> None:
        """Test LCS of identical strings."""
        assert lcs_string("hello", "hello") == "hello"
        assert lcs_string("test", "test") == "test"

    def test_no_common_characters(self) -> None:
        """Test LCS when no common characters exist."""
        assert lcs_string("abc", "xyz") == ""
        assert lcs_string("123", "456") == ""

    def test_single_character_lcs(self) -> None:
        """Test LCS that is a single character."""
        assert lcs_string("a", "a") == "a"
        result = lcs_string("abc", "def")
        assert result == ""  # No common chars

    def test_empty_strings(self) -> None:
        """Test LCS with empty strings."""
        assert lcs_string("", "hello") == ""
        assert lcs_string("hello", "") == ""
        assert lcs_string("", "") == ""


class TestLCSAll:
    """Test finding all possible LCS."""

    def test_single_lcs(self) -> None:
        """Test when there's only one LCS."""
        result = lcs_all("ABC", "AC")
        assert result == ["AC"]

    def test_multiple_lcs(self) -> None:
        """Test when multiple LCS exist."""
        result = lcs_all("AGTGATG", "GTTAG")
        # Can be "GTAG" or "GTTG"
        assert len(result) == 2
        assert set(result) == {"GTAG", "GTTG"}

    def test_identical_strings_one_lcs(self) -> None:
        """Test identical strings have one LCS."""
        result = lcs_all("ABC", "ABC")
        assert result == ["ABC"]

    def test_no_lcs(self) -> None:
        """Test when there's no LCS."""
        result = lcs_all("abc", "xyz")
        assert result == [""]


class TestLCSSpaceOptimized:
    """Test space-optimized LCS length calculation."""

    def test_same_as_standard(self) -> None:
        """Test that space-optimized gives same results."""
        test_pairs = [
            ("ABCDGH", "AEDFHR"),
            ("AGGTAB", "GXTXAYB"),
            ("hello", "yellow"),
            ("", "test"),
            ("test", ""),
        ]

        for str1, str2 in test_pairs:
            standard = lcs_length(str1, str2)
            optimized = lcs_length_space_optimized(str1, str2)
            assert standard == optimized, f"Failed for {str1}, {str2}"

    def test_long_strings(self) -> None:
        """Test with longer strings."""
        str1 = "a" * 100 + "b" * 100
        str2 = "a" * 100 + "c" * 100

        result = lcs_length_space_optimized(str1, str2)
        assert result == 100  # The "a"s in common


class TestLCSGeneric:
    """Test LCS with generic sequence types."""

    def test_list_sequences(self) -> None:
        """Test LCS with lists."""
        list1 = [1, 2, 3, 4, 5]
        list2 = [2, 4, 5, 6]
        result = lcs_generic(list1, list2)

        assert result == [2, 4, 5]

    def test_string_lists(self) -> None:
        """Test LCS with lists of strings."""
        list1 = ["a", "b", "c", "d"]
        list2 = ["b", "d", "e"]
        result = lcs_generic(list1, list2)

        assert result == ["b", "d"]

    def test_tuple_sequences(self) -> None:
        """Test LCS with tuples."""
        tuple1 = (1, 2, 3)
        tuple2 = (2, 3, 4)
        result = lcs_generic(tuple1, tuple2)

        assert result == [2, 3]

    def test_empty_sequences(self) -> None:
        """Test LCS with empty sequences."""
        assert lcs_generic([], [1, 2, 3]) == []
        assert lcs_generic([1, 2, 3], []) == []
        assert lcs_generic([], []) == []


class TestDiffStrings:
    """Test diff generation."""

    def test_identical_strings(self) -> None:
        """Test diff of identical strings."""
        diff = diff_strings("abc", "abc")

        assert len(diff) == 3
        for op, char in diff:
            assert op == "keep"

    def test_single_substitution(self) -> None:
        """Test diff with one substitution."""
        diff = diff_strings("abc", "adc")

        # Should have: keep a, delete b, insert d, keep c
        ops = [op for op, char in diff]
        assert "keep" in ops
        assert "delete" in ops or "insert" in ops

    def test_insertion_only(self) -> None:
        """Test diff with only insertions."""
        diff = diff_strings("ac", "abc")

        # Should insert 'b' between 'a' and 'c'
        keep_count = sum(1 for op, _ in diff if op == "keep")
        insert_count = sum(1 for op, _ in diff if op == "insert")

        assert keep_count == 2  # 'a' and 'c'
        assert insert_count == 1  # 'b'

    def test_deletion_only(self) -> None:
        """Test diff with only deletions."""
        diff = diff_strings("abc", "ac")

        # Should delete 'b'
        keep_count = sum(1 for op, _ in diff if op == "keep")
        delete_count = sum(1 for op, _ in diff if op == "delete")

        assert keep_count == 2  # 'a' and 'c'
        assert delete_count == 1  # 'b'

    def test_complex_diff(self) -> None:
        """Test diff with multiple operations."""
        diff = diff_strings("ABCDEFG", "ABDXEFG")

        # Should have mix of keep, delete, and insert operations
        ops = set(op for op, _ in diff)
        assert "keep" in ops


class TestLCSEdgeCases:
    """Test edge cases."""

    def test_single_character_strings(self) -> None:
        """Test with single character strings."""
        assert lcs_length("a", "a") == 1
        assert lcs_length("a", "b") == 0
        assert lcs_string("x", "x") == "x"

    def test_very_long_strings(self) -> None:
        """Test with very long strings."""
        str1 = "a" * 1000
        str2 = "a" * 1000

        result = lcs_length_space_optimized(str1, str2)
        assert result == 1000

    def test_all_different_characters(self) -> None:
        """Test strings with all different characters."""
        str1 = "abcdefgh"
        str2 = "ijklmnop"

        assert lcs_length(str1, str2) == 0
        assert lcs_string(str1, str2) == ""

    def test_one_char_repeated(self) -> None:
        """Test strings with repeated single character."""
        str1 = "aaaa"
        str2 = "aa"

        assert lcs_length(str1, str2) == 2
        assert lcs_string(str1, str2) == "aa"


class TestLCSApplications:
    """Test real-world applications."""

    def test_dna_sequence_comparison(self) -> None:
        """Test DNA sequence comparison."""
        seq1 = "ACCGGTCGAGTGCGCGGAAGCCGGCCGAA"
        seq2 = "GTCGTTCGGAATGCCGTTGCTCTGTAAA"

        lcs_len = lcs_length(seq1, seq2)

        # Should find some common subsequence
        assert lcs_len > 0
        assert lcs_len <= min(len(seq1), len(seq2))

    def test_file_line_comparison(self) -> None:
        """Test file line comparison (like diff utility)."""
        file1_lines = ["line1", "line2", "line3", "line4"]
        file2_lines = ["line1", "line2", "newline", "line4"]

        common_lines = lcs_generic(file1_lines, file2_lines)

        # Should find: line1, line2, line4
        assert common_lines == ["line1", "line2", "line4"]

    def test_version_control_diff(self) -> None:
        """Test use case similar to version control."""
        original = "functionfoobar"
        modified = "functionbazbar"

        diff = diff_strings(original, modified)

        # Should show what changed
        assert len(diff) > 0
        operations = [op for op, _ in diff]
        # Should have keep, delete, and/or insert operations
        assert len(set(operations)) > 1


class TestLCSPerformance:
    """Test performance characteristics."""

    def test_long_similar_strings(self) -> None:
        """Test with long similar strings."""
        str1 = "a" * 500 + "x"
        str2 = "a" * 500 + "y"

        # Should find the 500 'a's in common
        result = lcs_length_space_optimized(str1, str2)
        assert result == 500

    def test_long_completely_different(self) -> None:
        """Test with long completely different strings."""
        str1 = "a" * 500
        str2 = "b" * 500

        result = lcs_length_space_optimized(str1, str2)
        assert result == 0

    def test_many_short_sequences(self) -> None:
        """Test multiple LCS calculations."""
        sequences = [
            ("abc", "ac"),
            ("xyz", "xz"),
            ("123", "13"),
        ]

        for str1, str2 in sequences:
            lcs_length(str1, str2)  # Should complete quickly
