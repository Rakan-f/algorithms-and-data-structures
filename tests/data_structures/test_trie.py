"""
Unit tests for Trie (Prefix Tree) data structure.

Test coverage:
- Insert and search operations
- Prefix matching
- Word deletion
- Autocomplete functionality
- Edge cases
"""

import pytest
from algorithms.data_structures.trie import Trie


class TestTrieBasic:
    """Test basic Trie operations."""

    def test_insert_and_search(self) -> None:
        """Test inserting and searching for words."""
        trie = Trie()

        trie.insert("hello")
        assert trie.search("hello") is True
        assert trie.search("hell") is False
        assert trie.search("hello world") is False

    def test_empty_trie(self) -> None:
        """Test operations on empty trie."""
        trie = Trie()

        assert trie.search("anything") is False
        assert trie.starts_with("any") is False
        assert len(trie) == 0
        assert trie.get_all_words() == []

    def test_single_character(self) -> None:
        """Test with single character words."""
        trie = Trie()

        trie.insert("a")
        trie.insert("b")

        assert trie.search("a") is True
        assert trie.search("b") is True
        assert len(trie) == 2

    def test_multiple_inserts_same_word(self) -> None:
        """Test inserting the same word multiple times."""
        trie = Trie()

        trie.insert("test")
        trie.insert("test")
        trie.insert("test")

        assert trie.search("test") is True
        assert len(trie) == 1  # Should count unique words


class TestTriePrefix:
    """Test prefix-related operations."""

    def test_starts_with(self) -> None:
        """Test prefix checking."""
        trie = Trie()

        trie.insert("apple")
        trie.insert("app")
        trie.insert("application")

        assert trie.starts_with("app") is True
        assert trie.starts_with("appl") is True
        assert trie.starts_with("apple") is True
        assert trie.starts_with("ban") is False

    def test_words_with_prefix(self) -> None:
        """Test getting all words with a given prefix."""
        trie = Trie()

        words = ["apple", "app", "application", "apply", "banana"]
        for word in words:
            trie.insert(word)

        app_words = trie.words_with_prefix("app")
        assert set(app_words) == {"apple", "app", "application", "apply"}

        ban_words = trie.words_with_prefix("ban")
        assert ban_words == ["banana"]

        no_words = trie.words_with_prefix("xyz")
        assert no_words == []

    def test_count_words_with_prefix(self) -> None:
        """Test counting words with prefix."""
        trie = Trie()

        for word in ["cat", "cats", "catastrophe", "dog"]:
            trie.insert(word)

        assert trie.count_words_with_prefix("cat") == 3
        assert trie.count_words_with_prefix("cats") == 1
        assert trie.count_words_with_prefix("dog") == 1
        assert trie.count_words_with_prefix("bird") == 0

    def test_longest_common_prefix(self) -> None:
        """Test finding longest common prefix."""
        trie = Trie()

        trie.insert("flower")
        trie.insert("flow")
        trie.insert("flight")

        lcp = trie.longest_common_prefix()
        assert lcp == "fl"

        # Test with no common prefix
        trie2 = Trie()
        trie2.insert("dog")
        trie2.insert("cat")
        assert trie2.longest_common_prefix() == ""


class TestTrieDelete:
    """Test word deletion operations."""

    def test_delete_existing_word(self) -> None:
        """Test deleting an existing word."""
        trie = Trie()

        trie.insert("hello")
        trie.insert("hell")

        assert trie.delete("hello") is True
        assert trie.search("hello") is False
        assert trie.search("hell") is True  # Should not affect prefix

    def test_delete_nonexistent_word(self) -> None:
        """Test deleting a word that doesn't exist."""
        trie = Trie()

        trie.insert("hello")
        assert trie.delete("goodbye") is False
        assert trie.delete("hel") is False

    def test_delete_with_shared_prefix(self) -> None:
        """Test deletion when words share prefixes."""
        trie = Trie()

        trie.insert("app")
        trie.insert("apple")
        trie.insert("application")

        trie.delete("apple")

        assert trie.search("apple") is False
        assert trie.search("app") is True
        assert trie.search("application") is True

    def test_delete_all_words(self) -> None:
        """Test deleting all words from trie."""
        trie = Trie()

        words = ["cat", "dog", "bird"]
        for word in words:
            trie.insert(word)

        for word in words:
            trie.delete(word)

        assert len(trie) == 0
        for word in words:
            assert trie.search(word) is False


class TestTrieAutocomplete:
    """Test autocomplete-like functionality."""

    def test_autocomplete_suggestions(self) -> None:
        """Test getting autocomplete suggestions."""
        trie = Trie()

        dictionary = [
            "apple", "application", "apply", "ape",
            "banana", "band", "bandana",
            "cat", "category"
        ]

        for word in dictionary:
            trie.insert(word)

        # Test suggestions for "app"
        suggestions = trie.words_with_prefix("app")
        assert set(suggestions) == {"apple", "application", "apply"}

        # Test suggestions for "ban"
        suggestions = trie.words_with_prefix("ban")
        assert set(suggestions) == {"banana", "band", "bandana"}

        # Test suggestions for "ca"
        suggestions = trie.words_with_prefix("ca")
        assert set(suggestions) == {"cat", "category"}

    def test_empty_prefix_returns_all(self) -> None:
        """Test that empty prefix returns all words."""
        trie = Trie()

        words = ["apple", "banana", "cherry"]
        for word in words:
            trie.insert(word)

        all_words = trie.words_with_prefix("")
        assert set(all_words) == set(words)


class TestTrieEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string_insert(self) -> None:
        """Test inserting empty string."""
        trie = Trie()

        trie.insert("")
        # Empty string insertion should be ignored
        assert len(trie) == 0

    def test_special_characters(self) -> None:
        """Test with special characters."""
        trie = Trie()

        trie.insert("hello-world")
        trie.insert("test@123")
        trie.insert("a.b.c")

        assert trie.search("hello-world") is True
        assert trie.search("test@123") is True
        assert trie.search("a.b.c") is True

    def test_numbers_in_words(self) -> None:
        """Test with numbers in words."""
        trie = Trie()

        trie.insert("test123")
        trie.insert("abc456")

        assert trie.search("test123") is True
        assert trie.search("abc456") is True
        assert trie.starts_with("test") is True

    def test_case_sensitivity(self) -> None:
        """Test that Trie is case-sensitive."""
        trie = Trie()

        trie.insert("Hello")
        trie.insert("hello")

        assert trie.search("Hello") is True
        assert trie.search("hello") is True
        assert trie.search("HELLO") is False
        assert len(trie) == 2


class TestTrieDunderMethods:
    """Test special Python methods."""

    def test_len(self) -> None:
        """Test __len__ method."""
        trie = Trie()

        assert len(trie) == 0

        trie.insert("a")
        assert len(trie) == 1

        trie.insert("b")
        trie.insert("c")
        assert len(trie) == 3

        trie.delete("b")
        assert len(trie) == 2

    def test_contains(self) -> None:
        """Test __contains__ method (in operator)."""
        trie = Trie()

        trie.insert("hello")

        assert "hello" in trie
        assert "hell" not in trie
        assert "goodbye" not in trie

    def test_repr(self) -> None:
        """Test __repr__ method."""
        trie = Trie()

        trie.insert("apple")
        trie.insert("banana")

        repr_str = repr(trie)
        assert "Trie" in repr_str
        assert "apple" in repr_str or "banana" in repr_str


class TestTrieGetAllWords:
    """Test getting all words from Trie."""

    def test_get_all_words(self) -> None:
        """Test retrieving all words."""
        trie = Trie()

        words = ["cat", "cats", "dog", "dogs", "bird"]
        for word in words:
            trie.insert(word)

        all_words = trie.get_all_words()
        assert set(all_words) == set(words)

    def test_get_all_words_alphabetical_order(self) -> None:
        """Test that get_all_words returns alphabetically sorted."""
        trie = Trie()

        words = ["zebra", "apple", "mango", "banana"]
        for word in words:
            trie.insert(word)

        all_words = trie.get_all_words()
        # Words should be in alphabetical order due to sorted iteration
        assert all_words == sorted(words)


class TestTriePerformance:
    """Test performance characteristics."""

    def test_large_dictionary(self) -> None:
        """Test with a large number of words."""
        trie = Trie()

        # Insert many words
        for i in range(1000):
            trie.insert(f"word{i}")

        assert len(trie) == 1000

        # Search should still be fast
        assert trie.search("word500") is True
        assert trie.search("word999") is True
        assert trie.search("word1000") is False

    def test_long_words(self) -> None:
        """Test with very long words."""
        trie = Trie()

        long_word = "a" * 1000
        trie.insert(long_word)

        assert trie.search(long_word) is True
        assert trie.starts_with("a" * 500) is True

    def test_many_similar_words(self) -> None:
        """Test with many words sharing common prefix."""
        trie = Trie()

        prefix = "test"
        for i in range(100):
            trie.insert(f"{prefix}{i}")

        assert len(trie) == 100
        assert trie.count_words_with_prefix(prefix) == 100
