"""
KMP (Knuth-Morris-Pratt) Pattern Matching Algorithm.

KMP is an efficient string matching algorithm that finds all occurrences of a
pattern in a text in linear time. It avoids unnecessary comparisons by using
information from previous comparisons to skip ahead.

Time Complexity: O(n + m) where n is text length, m is pattern length
Space Complexity: O(m) for the LPS (Longest Proper Prefix which is also Suffix) array

Example:
    >>> text = "ABABDABACDABABCABAB"
    >>> pattern = "ABABCABAB"
    >>> matches = kmp_search(text, pattern)
    >>> matches
    [10]  # Pattern found at index 10
"""

from typing import List


def compute_lps_array(pattern: str) -> List[int]:
    """
    Compute the Longest Proper Prefix which is also Suffix (LPS) array.

    The LPS array is used by KMP to determine how many characters to skip
    when a mismatch occurs. LPS[i] contains the length of the longest proper
    prefix of pattern[0..i] which is also a suffix of pattern[0..i].

    Args:
        pattern: Pattern string to compute LPS for

    Returns:
        LPS array of same length as pattern

    Time Complexity: O(m) where m is pattern length
    Space Complexity: O(m) for the LPS array

    Example:
        >>> compute_lps_array("ABABCABAB")
        [0, 0, 1, 2, 0, 1, 2, 3, 4]

        Explanation for "ABABCABAB":
        - Index 0: No proper prefix, LPS[0] = 0
        - Index 1: "AB" - no match, LPS[1] = 0
        - Index 2: "ABA" - "A" matches, LPS[2] = 1
        - Index 3: "ABAB" - "AB" matches, LPS[3] = 2
        - Index 4: "ABABC" - no match after "AB", LPS[4] = 0
        - Index 5: "ABABCA" - "A" matches, LPS[5] = 1
        - Index 6: "ABABCAB" - "AB" matches, LPS[6] = 2
        - Index 7: "ABABCABA" - "ABA" matches, LPS[7] = 3
        - Index 8: "ABABCABAB" - "ABAB" matches, LPS[8] = 4
    """
    m = len(pattern)
    lps = [0] * m

    # Length of previous longest prefix suffix
    length = 0
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            # Characters match, extend the current prefix suffix
            length += 1
            lps[i] = length
            i += 1
        else:
            # Mismatch after length matches
            if length != 0:
                # Use previous LPS value to avoid redundant comparisons
                # This is the key insight of KMP
                length = lps[length - 1]
            else:
                # No prefix suffix match
                lps[i] = 0
                i += 1

    return lps


def kmp_search(text: str, pattern: str) -> List[int]:
    """
    Find all occurrences of pattern in text using KMP algorithm.

    Args:
        text: Text string to search in
        pattern: Pattern string to search for

    Returns:
        List of starting indices where pattern is found in text

    Raises:
        ValueError: If pattern is empty

    Time Complexity: O(n + m) where n is text length, m is pattern length
    Space Complexity: O(m) for the LPS array

    Example:
        >>> kmp_search("ABABDABACDABABCABAB", "ABABCABAB")
        [10]
        >>> kmp_search("AAAAAAA", "AAA")
        [0, 1, 2, 3, 4]
        >>> kmp_search("HELLO WORLD", "XYZ")
        []
    """
    if not pattern:
        raise ValueError("Pattern cannot be empty")

    if not text:
        return []

    n = len(text)
    m = len(pattern)

    # Preprocess pattern to build LPS array
    lps = compute_lps_array(pattern)

    matches = []
    i = 0  # Index for text
    j = 0  # Index for pattern

    while i < n:
        if pattern[j] == text[i]:
            # Characters match
            i += 1
            j += 1

        if j == m:
            # Found complete match
            matches.append(i - j)
            # Use LPS to find next potential match
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            # Mismatch after j matches
            if j != 0:
                # Use LPS to skip ahead
                # This is where KMP saves time compared to naive approach
                j = lps[j - 1]
            else:
                # No match at all, move to next character in text
                i += 1

    return matches


def kmp_search_first(text: str, pattern: str) -> int:
    """
    Find first occurrence of pattern in text using KMP.

    This is an optimized version that returns immediately upon finding
    the first match, rather than finding all matches.

    Args:
        text: Text string to search in
        pattern: Pattern string to search for

    Returns:
        Index of first occurrence, or -1 if not found

    Raises:
        ValueError: If pattern is empty

    Time Complexity: O(n + m) worst case, often better with early termination
    Space Complexity: O(m)

    Example:
        >>> kmp_search_first("HELLO WORLD", "WORLD")
        6
        >>> kmp_search_first("HELLO WORLD", "XYZ")
        -1
    """
    if not pattern:
        raise ValueError("Pattern cannot be empty")

    if not text:
        return -1

    n = len(text)
    m = len(pattern)

    lps = compute_lps_array(pattern)

    i = 0
    j = 0

    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == m:
            # Found match!
            return i - j

        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return -1


def kmp_count(text: str, pattern: str) -> int:
    """
    Count number of occurrences of pattern in text using KMP.

    Args:
        text: Text string to search in
        pattern: Pattern string to search for

    Returns:
        Number of occurrences of pattern in text

    Raises:
        ValueError: If pattern is empty

    Time Complexity: O(n + m)
    Space Complexity: O(m)

    Example:
        >>> kmp_count("AAAAAAA", "AAA")
        5
        >>> kmp_count("ABCABCABC", "ABC")
        3
    """
    return len(kmp_search(text, pattern))


if __name__ == "__main__":
    print("KMP Pattern Matching Algorithm\n")

    # Example 1: Basic search
    print("Example 1: Basic Pattern Search")
    text1 = "ABABDABACDABABCABAB"
    pattern1 = "ABABCABAB"

    print(f"Text:    {text1}")
    print(f"Pattern: {pattern1}")

    matches = kmp_search(text1, pattern1)
    print(f"\nMatches found at indices: {matches}")

    for idx in matches:
        print(f"  Position {idx}: {text1[idx:idx+len(pattern1)]}")

    # Example 2: Multiple overlapping matches
    print("\n" + "="*60)
    print("Example 2: Overlapping Matches")
    text2 = "AAAAAAA"
    pattern2 = "AAA"

    print(f"Text:    {text2}")
    print(f"Pattern: {pattern2}")

    matches2 = kmp_search(text2, pattern2)
    print(f"\nMatches found at indices: {matches2}")
    print(f"Total count: {kmp_count(text2, pattern2)}")

    # Example 3: LPS Array computation
    print("\n" + "="*60)
    print("Example 3: LPS Array Computation")
    patterns = ["ABABCABAB", "AAAA", "ABCDE", "ABABAB"]

    for pattern in patterns:
        lps = compute_lps_array(pattern)
        print(f"\nPattern: {pattern}")
        print(f"LPS:     {lps}")
        print(f"Visual:  ", end="")
        for i, char in enumerate(pattern):
            print(f"{char}({lps[i]})", end=" ")
        print()

    # Example 4: Performance comparison (conceptual)
    print("\n" + "="*60)
    print("Example 4: Why KMP is Efficient")
    text3 = "AAAAAAAAAB"
    pattern3 = "AAAAB"

    print(f"Text:    {text3}")
    print(f"Pattern: {pattern3}")
    print(f"\nNaive approach: Would do many redundant comparisons")
    print(f"KMP approach:   Uses LPS to skip redundant comparisons")

    lps3 = compute_lps_array(pattern3)
    print(f"\nLPS array: {lps3}")
    print(f"When mismatch at last char, KMP knows first {lps3[-1]} chars already match")

    first_match = kmp_search_first(text3, pattern3)
    print(f"\nFirst match at index: {first_match}")

    # Example 5: Real-world use case
    print("\n" + "="*60)
    print("Example 5: Real-World Use Case - Log Analysis")
    log_text = "ERROR: Connection failed. INFO: Retrying... ERROR: Connection failed. SUCCESS: Connected."
    error_pattern = "ERROR"

    error_positions = kmp_search(log_text, error_pattern)
    print(f"Log: {log_text}")
    print(f"\nSearching for '{error_pattern}':")
    print(f"Found {len(error_positions)} errors at positions: {error_positions}")

    for pos in error_positions:
        # Extract context around error
        start = max(0, pos - 5)
        end = min(len(log_text), pos + 30)
        print(f"  ...{log_text[start:end]}...")
