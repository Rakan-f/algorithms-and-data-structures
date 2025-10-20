"""
Trie (Prefix Tree) Data Structure.

A Trie is a tree-like data structure used for efficient string storage and retrieval.
It's particularly useful for autocomplete, spell checking, and prefix matching operations.

Time Complexity:
    - Insert: O(m) where m is the length of the string
    - Search: O(m)
    - StartsWith (prefix search): O(m)
    - Delete: O(m)

Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of keys, M is avg length

Example:
    >>> trie = Trie()
    >>> trie.insert("apple")
    >>> trie.search("apple")
    True
    >>> trie.search("app")
    False
    >>> trie.starts_with("app")
    True
"""

from typing import Optional, List, Dict


class TrieNode:
    """Node in a Trie data structure."""

    def __init__(self) -> None:
        """
        Initialize a Trie node.

        Attributes:
            children: Dictionary mapping characters to child nodes
            is_end_of_word: Flag indicating if this node represents end of a word
            word_count: Number of words ending at this node (for handling duplicates)
        """
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False
        self.word_count: int = 0


class Trie:
    """
    Trie (Prefix Tree) implementation for efficient string operations.

    A Trie stores strings in a tree structure where each node represents a character.
    Strings that share common prefixes share the same path in the tree.

    Time Complexity:
        - insert(word): O(m) where m is length of word
        - search(word): O(m)
        - starts_with(prefix): O(m)
        - delete(word): O(m)

    Space Complexity: O(ALPHABET_SIZE * N * M)
        - ALPHABET_SIZE: typically 26 for lowercase English
        - N: number of words
        - M: average word length

    Example:
        >>> trie = Trie()
        >>> trie.insert("hello")
        >>> trie.insert("hell")
        >>> trie.insert("heaven")
        >>> trie.search("hello")
        True
        >>> trie.starts_with("hea")
        True
        >>> list(trie.words_with_prefix("hel"))
        ['hell', 'hello']
    """

    def __init__(self) -> None:
        """Initialize an empty Trie with a root node."""
        self.root = TrieNode()
        self.size = 0

    def insert(self, word: str) -> None:
        """
        Insert a word into the Trie.

        Args:
            word: Word to insert

        Time Complexity: O(m) where m is length of word
        Space Complexity: O(m) in worst case (if no prefix exists)

        Example:
            >>> trie = Trie()
            >>> trie.insert("apple")
            >>> trie.search("apple")
            True
        """
        if not word:
            return

        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        if not node.is_end_of_word:
            self.size += 1

        node.is_end_of_word = True
        node.word_count += 1

    def search(self, word: str) -> bool:
        """
        Search for an exact word in the Trie.

        Args:
            word: Word to search for

        Returns:
            True if word exists in Trie, False otherwise

        Time Complexity: O(m)

        Example:
            >>> trie = Trie()
            >>> trie.insert("apple")
            >>> trie.search("apple")
            True
            >>> trie.search("app")
            False
        """
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word in the Trie starts with the given prefix.

        Args:
            prefix: Prefix to search for

        Returns:
            True if at least one word starts with prefix

        Time Complexity: O(m)

        Example:
            >>> trie = Trie()
            >>> trie.insert("apple")
            >>> trie.starts_with("app")
            True
            >>> trie.starts_with("ban")
            False
        """
        return self._find_node(prefix) is not None

    def delete(self, word: str) -> bool:
        """
        Delete a word from the Trie.

        Args:
            word: Word to delete

        Returns:
            True if word was found and deleted, False otherwise

        Time Complexity: O(m)

        Example:
            >>> trie = Trie()
            >>> trie.insert("apple")
            >>> trie.delete("apple")
            True
            >>> trie.search("apple")
            False
        """
        def _delete_helper(node: TrieNode, word: str, depth: int) -> bool:
            """
            Recursively delete a word from the Trie.

            Returns True if the current node should be deleted.
            """
            if depth == len(word):
                if not node.is_end_of_word:
                    return False

                node.word_count -= 1
                if node.word_count == 0:
                    node.is_end_of_word = False
                    self.size -= 1

                # Delete node if it has no children
                return len(node.children) == 0

            char = word[depth]
            if char not in node.children:
                return False

            child_node = node.children[char]
            should_delete_child = _delete_helper(child_node, word, depth + 1)

            if should_delete_child:
                del node.children[char]
                # Delete current node if it's not end of another word and has no children
                return not node.is_end_of_word and len(node.children) == 0

            return False

        if not word:
            return False

        return _delete_helper(self.root, word, 0)

    def words_with_prefix(self, prefix: str) -> List[str]:
        """
        Get all words in the Trie that start with the given prefix.

        Args:
            prefix: Prefix to search for

        Returns:
            List of all words starting with prefix

        Time Complexity: O(p + n) where p is prefix length, n is number of nodes in subtree
        Space Complexity: O(n) for storing results

        Example:
            >>> trie = Trie()
            >>> trie.insert("apple")
            >>> trie.insert("app")
            >>> trie.insert("application")
            >>> list(trie.words_with_prefix("app"))
            ['app', 'apple', 'application']
        """
        node = self._find_node(prefix)
        if node is None:
            return []

        words = []
        self._collect_words(node, prefix, words)
        return words

    def get_all_words(self) -> List[str]:
        """
        Get all words stored in the Trie.

        Returns:
            List of all words

        Time Complexity: O(n) where n is total number of characters in all words
        Space Complexity: O(n) for storing results

        Example:
            >>> trie = Trie()
            >>> trie.insert("cat")
            >>> trie.insert("dog")
            >>> sorted(trie.get_all_words())
            ['cat', 'dog']
        """
        words = []
        self._collect_words(self.root, "", words)
        return words

    def count_words_with_prefix(self, prefix: str) -> int:
        """
        Count how many words start with the given prefix.

        Args:
            prefix: Prefix to count

        Returns:
            Number of words starting with prefix

        Time Complexity: O(p + n) where p is prefix length, n is nodes in subtree

        Example:
            >>> trie = Trie()
            >>> trie.insert("app")
            >>> trie.insert("apple")
            >>> trie.insert("application")
            >>> trie.count_words_with_prefix("app")
            3
        """
        return len(self.words_with_prefix(prefix))

    def longest_common_prefix(self) -> str:
        """
        Find the longest common prefix of all words in the Trie.

        Returns:
            Longest common prefix string

        Time Complexity: O(m) where m is length of shortest word
        Space Complexity: O(m)

        Example:
            >>> trie = Trie()
            >>> trie.insert("flower")
            >>> trie.insert("flow")
            >>> trie.insert("flight")
            >>> trie.longest_common_prefix()
            'fl'
        """
        if self.size == 0:
            return ""

        prefix = []
        node = self.root

        while len(node.children) == 1 and not node.is_end_of_word:
            char = next(iter(node.children))
            prefix.append(char)
            node = node.children[char]

        return "".join(prefix)

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """
        Find the node corresponding to the given prefix.

        Args:
            prefix: Prefix to find

        Returns:
            Node if prefix exists, None otherwise
        """
        if not prefix:
            return self.root

        node = self.root

        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]

        return node

    def _collect_words(self, node: TrieNode, prefix: str, words: List[str]) -> None:
        """
        Recursively collect all words from a given node.

        Args:
            node: Starting node
            prefix: Prefix accumulated so far
            words: List to accumulate words into
        """
        if node.is_end_of_word:
            words.append(prefix)

        for char, child_node in sorted(node.children.items()):
            self._collect_words(child_node, prefix + char, words)

    def __len__(self) -> int:
        """Return number of unique words in the Trie."""
        return self.size

    def __contains__(self, word: str) -> bool:
        """Check if word exists in Trie (enables 'word in trie' syntax)."""
        return self.search(word)

    def __repr__(self) -> str:
        """String representation of Trie."""
        words = self.get_all_words()
        if len(words) <= 5:
            return f"Trie({words})"
        else:
            return f"Trie([{', '.join(words[:5])}, ... ({len(words)} total)])"


if __name__ == "__main__":
    print("Trie (Prefix Tree) Demonstration\n")
    print("=" * 70)

    # Create a Trie
    trie = Trie()

    # Insert words
    words = ["apple", "app", "apricot", "application", "apply", "banana", "band", "bandana"]
    print("Inserting words:")
    for word in words:
        trie.insert(word)
        print(f"  Inserted: {word}")

    print(f"\nTrie now contains {len(trie)} unique words")

    # Search operations
    print("\n" + "=" * 70)
    print("Search operations:")
    test_words = ["apple", "app", "appl", "ban", "bandana", "orange"]

    for word in test_words:
        exists = trie.search(word)
        print(f"  search('{word}'): {exists}")

    # Prefix operations
    print("\n" + "=" * 70)
    print("Prefix operations:")
    prefixes = ["app", "ban", "or"]

    for prefix in prefixes:
        has_prefix = trie.starts_with(prefix)
        count = trie.count_words_with_prefix(prefix)
        words_list = trie.words_with_prefix(prefix)

        print(f"\n  Prefix: '{prefix}'")
        print(f"    Exists: {has_prefix}")
        print(f"    Count: {count}")
        print(f"    Words: {words_list}")

    # Longest common prefix
    print("\n" + "=" * 70)
    print("Longest common prefix:")
    trie2 = Trie()
    for word in ["flower", "flow", "flight"]:
        trie2.insert(word)
    print(f"  Words: {trie2.get_all_words()}")
    print(f"  Longest common prefix: '{trie2.longest_common_prefix()}'")

    # Delete operations
    print("\n" + "=" * 70)
    print("Delete operations:")
    print(f"  Before deletion: {len(trie)} words")
    print(f"  All words: {trie.get_all_words()}")

    trie.delete("app")
    print(f"\n  Deleted 'app'")
    print(f"  After deletion: {len(trie)} words")
    print(f"  search('app'): {trie.search('app')}")
    print(f"  search('apple'): {trie.search('apple')} (should still exist)")

    # Autocomplete demo
    print("\n" + "=" * 70)
    print("Autocomplete Demo:")
    print("  Typing 'app'...")
    suggestions = trie.words_with_prefix("app")
    print(f"  Suggestions: {suggestions}")

    print("\n  Typing 'ban'...")
    suggestions = trie.words_with_prefix("ban")
    print(f"  Suggestions: {suggestions}")
