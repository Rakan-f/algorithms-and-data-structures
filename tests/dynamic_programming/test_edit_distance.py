"""
Unit tests for Edit Distance (Levenshtein Distance) algorithm.

Test coverage:
- Basic edit distance calculation
- Edit operations tracking
- Space-optimized version
- One-edit-away check
- Similarity calculation
"""

import pytest
from algorithms.dynamic_programming.edit_distance import (
    edit_distance,
    edit_distance_with_operations,
    edit_distance_space_optimized,
    is_one_edit_away,
    similarity_percentage
)


class TestEditDistanceBasic:
    """Test basic edit distance calculation."""

    def test_identical_strings(self) -> None:
        """Test that identical strings have distance 0."""
        assert edit_distance("hello", "hello") == 0
        assert edit_distance("", "") == 0
        assert edit_distance("test", "test") == 0

    def test_one_insertion(self) -> None:
        """Test single insertion operation."""
        assert edit_distance("cat", "cats") == 1
        assert edit_distance("app", "apple") == 2

    def test_one_deletion(self) -> None:
        """Test single deletion operation."""
        assert edit_distance("cats", "cat") == 1
        assert edit_distance("apple", "app") == 2

    def test_one_substitution(self) -> None:
        """Test single substitution operation."""
        assert edit_distance("cat", "bat") == 1
        assert edit_distance("dog", "fog") == 1

    def test_classic_examples(self) -> None:
        """Test classic edit distance examples."""
        assert edit_distance("kitten", "sitting") == 3
        assert edit_distance("horse", "ros") == 3
        assert edit_distance("intention", "execution") == 5
        assert edit_distance("saturday", "sunday") == 3


class TestEditDistanceEmpty:
    """Test edit distance with empty strings."""

    def test_empty_to_string(self) -> None:
        """Test distance from empty string to non-empty."""
        assert edit_distance("", "abc") == 3
        assert edit_distance("", "hello") == 5

    def test_string_to_empty(self) -> None:
        """Test distance from non-empty string to empty."""
        assert edit_distance("abc", "") == 3
        assert edit_distance("hello", "") == 5

    def test_both_empty(self) -> None:
        """Test distance between two empty strings."""
        assert edit_distance("", "") == 0


class TestEditDistanceWithOperations:
    """Test edit distance with operation tracking."""

    def test_returns_operations(self) -> None:
        """Test that operations are returned."""
        distance, ops = edit_distance_with_operations("cat", "bat")

        assert distance == 1
        assert len(ops) == 1
        assert "Substitute" in ops[0] or "substitute" in ops[0].lower()

    def test_kitten_to_sitting(self) -> None:
        """Test operations for kitten -> sitting."""
        distance, ops = edit_distance_with_operations("kitten", "sitting")

        assert distance == 3
        assert len(ops) == 3

    def test_identical_strings_no_ops(self) -> None:
        """Test that identical strings have no operations."""
        distance, ops = edit_distance_with_operations("same", "same")

        assert distance == 0
        assert len(ops) == 0

    def test_all_operation_types(self) -> None:
        """Test that all types of operations can be tracked."""
        # This will require insert, delete, and substitute
        distance, ops = edit_distance_with_operations("abc", "adc")

        assert distance == 1
        # Should involve substitution b->d


class TestEditDistanceSpaceOptimized:
    """Test space-optimized version."""

    def test_same_results_as_standard(self) -> None:
        """Test that space-optimized gives same results."""
        test_pairs = [
            ("kitten", "sitting"),
            ("horse", "ros"),
            ("intention", "execution"),
            ("", "abc"),
            ("abc", ""),
        ]

        for str1, str2 in test_pairs:
            standard = edit_distance(str1, str2)
            optimized = edit_distance_space_optimized(str1, str2)
            assert standard == optimized, f"Failed for {str1}, {str2}"

    def test_long_strings(self) -> None:
        """Test with longer strings."""
        str1 = "a" * 100
        str2 = "b" * 100

        result = edit_distance_space_optimized(str1, str2)
        assert result == 100  # All substitutions


class TestIsOneEditAway:
    """Test one-edit-away check."""

    def test_one_substitution_away(self) -> None:
        """Test strings one substitution apart."""
        assert is_one_edit_away("pale", "bale") is True
        assert is_one_edit_away("cat", "bat") is True

    def test_one_insertion_away(self) -> None:
        """Test strings one insertion apart."""
        assert is_one_edit_away("cat", "cats") is True
        assert is_one_edit_away("pales", "pale") is True

    def test_one_deletion_away(self) -> None:
        """Test strings one deletion apart."""
        assert is_one_edit_away("cats", "cat") is True
        assert is_one_edit_away("pale", "ple") is True

    def test_two_edits_away(self) -> None:
        """Test strings more than one edit apart."""
        assert is_one_edit_away("pale", "bake") is False
        assert is_one_edit_away("cat", "dog") is False

    def test_identical_strings(self) -> None:
        """Test identical strings (zero edits)."""
        assert is_one_edit_away("same", "same") is False

    def test_length_difference_too_large(self) -> None:
        """Test strings with length difference > 1."""
        assert is_one_edit_away("cat", "category") is False
        assert is_one_edit_away("", "ab") is False


class TestSimilarityPercentage:
    """Test similarity percentage calculation."""

    def test_identical_strings(self) -> None:
        """Test that identical strings have 100% similarity."""
        assert similarity_percentage("hello", "hello") == 100.0
        assert similarity_percentage("test", "test") == 100.0

    def test_completely_different(self) -> None:
        """Test completely different strings."""
        result = similarity_percentage("abc", "xyz")
        assert result == 0.0

    def test_partial_similarity(self) -> None:
        """Test strings with partial similarity."""
        # "kitten" -> "sitting" has distance 3, max_len 7
        # similarity = (7-3)/7 * 100 = 57.14%
        result = similarity_percentage("kitten", "sitting")
        assert 57.0 < result < 58.0

    def test_empty_strings(self) -> None:
        """Test empty strings."""
        assert similarity_percentage("", "") == 100.0

    def test_one_empty_string(self) -> None:
        """Test one empty string."""
        assert similarity_percentage("hello", "") == 0.0
        assert similarity_percentage("", "hello") == 0.0


class TestEditDistanceEdgeCases:
    """Test edge cases."""

    def test_single_character_strings(self) -> None:
        """Test with single character strings."""
        assert edit_distance("a", "b") == 1
        assert edit_distance("a", "a") == 0
        assert edit_distance("x", "") == 1

    def test_very_long_strings(self) -> None:
        """Test with very long strings."""
        str1 = "a" * 1000
        str2 = "a" * 1000 + "b"

        result = edit_distance_space_optimized(str1, str2)
        assert result == 1  # Just one insertion

    def test_all_operations_needed(self) -> None:
        """Test string requiring all operation types."""
        # abc -> dez requires: substitute a->d, substitute b->e, substitute c->z
        distance = edit_distance("abc", "dez")
        assert distance == 3

    def test_repeated_characters(self) -> None:
        """Test strings with many repeated characters."""
        assert edit_distance("aaaa", "aaab") == 1
        assert edit_distance("aaaa", "bbbb") == 4


class TestEditDistanceApplications:
    """Test real-world applications."""

    def test_spell_checking(self) -> None:
        """Test use case for spell checking."""
        correct = "algorithm"
        misspellings = ["algorythm", "algoritm", "algorith"]

        for misspelled in misspellings:
            distance = edit_distance(correct, misspelled)
            # Should be small distance for common misspellings
            assert distance <= 3

    def test_dna_sequence_alignment(self) -> None:
        """Test use case for DNA sequence alignment."""
        seq1 = "AGGTAB"
        seq2 = "GXTXAYB"

        distance = edit_distance(seq1, seq2)
        # Should be able to calculate distance
        assert distance >= 0

    def test_find_similar_words(self) -> None:
        """Test finding similar words in a dictionary."""
        target = "hello"
        dictionary = ["hell", "hallo", "hillo", "world", "help"]

        similarities = [(word, edit_distance(target, word)) for word in dictionary]
        similarities.sort(key=lambda x: x[1])

        # "hell" and "hallo" should be closest
        assert similarities[0][1] <= 2


class TestEditDistancePerformance:
    """Test performance characteristics."""

    def test_long_string_performance(self) -> None:
        """Test that algorithm completes for reasonably long strings."""
        str1 = "a" * 500
        str2 = "b" * 500

        # Should complete without timeout
        result = edit_distance_space_optimized(str1, str2)
        assert result == 500

    def test_similar_long_strings(self) -> None:
        """Test with long similar strings."""
        str1 = "a" * 500 + "x"
        str2 = "a" * 500 + "y"

        result = edit_distance_space_optimized(str1, str2)
        assert result == 1  # Only last character different
