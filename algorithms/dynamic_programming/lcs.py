"""
Longest Common Subsequence (LCS) Algorithm.

LCS finds the longest subsequence common to two sequences. A subsequence is a
sequence that appears in the same relative order, but not necessarily contiguous.
Used in diff utilities, bioinformatics, and file comparison tools.

Time Complexity: O(mn) where m and n are lengths of the two sequences
Space Complexity: O(mn) standard, O(min(m,n)) space-optimized

Example:
    >>> lcs_length("ABCDGH", "AEDFHR")
    3  # "ADH"
    >>> lcs_string("ABCDGH", "AEDFHR")
    'ADH'
"""

from typing import List, Tuple, TypeVar, Sequence

T = TypeVar('T')


def lcs_length(str1: str, str2: str) -> int:
    """
    Calculate the length of the longest common subsequence.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Length of LCS

    Time Complexity: O(mn)
    Space Complexity: O(mn)

    Algorithm:
        dp[i][j] = LCS length of str1[0:i] and str2[0:j]

        dp[i][j] = {
            dp[i-1][j-1] + 1           if str1[i-1] == str2[j-1]
            max(dp[i-1][j], dp[i][j-1]) otherwise
        }

    Example:
        >>> lcs_length("AGGTAB", "GXTXAYB")
        4  # "GTAB"
    """
    m, n = len(str1), len(str2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match, extend LCS
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # Take maximum of excluding either character
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def lcs_string(str1: str, str2: str) -> str:
    """
    Find the actual longest common subsequence string.

    Args:
        str1: First string
        str2: Second string

    Returns:
        One possible LCS string (there may be multiple)

    Time Complexity: O(mn)
    Space Complexity: O(mn)

    Example:
        >>> lcs_string("ABCDGH", "AEDFHR")
        'ADH'
        >>> lcs_string("AGGTAB", "GXTXAYB")
        'GTAB'
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the actual LCS
    lcs = []
    i, j = m, n

    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            # Part of LCS
            lcs.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            # Move up
            i -= 1
        else:
            # Move left
            j -= 1

    return ''.join(reversed(lcs))


def lcs_all(str1: str, str2: str) -> List[str]:
    """
    Find all possible longest common subsequences.

    When multiple LCS exist with the same length, this finds all of them.

    Args:
        str1: First string
        str2: Second string

    Returns:
        List of all LCS strings

    Time Complexity: O(mn + k) where k is number of LCS
    Space Complexity: O(mn + k)

    Example:
        >>> sorted(lcs_all("AGTGATG", "GTTAG"))
        ['GTAG', 'GTTG']
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Find all LCS using backtracking
    def backtrack(i: int, j: int, current: str) -> None:
        """Recursively find all LCS."""
        if i == 0 or j == 0:
            result = current[::-1]
            if result not in all_lcs:
                all_lcs.add(result)
            return

        if str1[i - 1] == str2[j - 1]:
            # Part of LCS, must include it
            backtrack(i - 1, j - 1, current + str1[i - 1])
        else:
            # Explore both directions if they have same LCS length
            if dp[i - 1][j] == dp[i][j]:
                backtrack(i - 1, j, current)
            if dp[i][j - 1] == dp[i][j]:
                backtrack(i, j - 1, current)

    all_lcs = set()
    backtrack(m, n, "")
    return sorted(all_lcs)


def lcs_length_space_optimized(str1: str, str2: str) -> int:
    """
    Calculate LCS length with O(min(m,n)) space complexity.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Length of LCS

    Time Complexity: O(mn)
    Space Complexity: O(min(m,n))

    Example:
        >>> lcs_length_space_optimized("AGGTAB", "GXTXAYB")
        4
    """
    # Ensure str2 is the shorter string
    if len(str1) < len(str2):
        str1, str2 = str2, str1

    m, n = len(str1), len(str2)

    # Only need two rows
    prev_row = [0] * (n + 1)
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                curr_row[j] = prev_row[j - 1] + 1
            else:
                curr_row[j] = max(prev_row[j], curr_row[j - 1])

        prev_row, curr_row = curr_row, prev_row

    return prev_row[n]


def lcs_generic(seq1: Sequence[T], seq2: Sequence[T]) -> List[T]:
    """
    Find LCS for any sequence type (lists, tuples, etc.).

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        LCS as a list

    Time Complexity: O(mn)
    Space Complexity: O(mn)

    Example:
        >>> lcs_generic([1, 2, 3, 4], [2, 4, 3])
        [2, 3]
        >>> lcs_generic(['a', 'b', 'c'], ['b', 'a', 'c'])
        ['b', 'c']
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    lcs = []
    i, j = m, n

    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return list(reversed(lcs))


def diff_strings(str1: str, str2: str) -> List[Tuple[str, str]]:
    """
    Generate a diff between two strings using LCS.

    Returns operations to transform str1 into str2, similar to diff utility.

    Args:
        str1: Original string
        str2: Modified string

    Returns:
        List of (operation, character) tuples where operation is:
        - 'keep': character is in both strings
        - 'delete': character only in str1
        - 'insert': character only in str2

    Time Complexity: O(mn)
    Space Complexity: O(mn)

    Example:
        >>> diff_strings("ABCD", "ACBD")
        [('keep', 'A'), ('delete', 'B'), ('keep', 'C'), ('insert', 'B'), ('keep', 'D')]
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Generate diff
    diff = []
    i, j = m, n

    while i > 0 or j > 0:
        if i == 0:
            diff.append(('insert', str2[j - 1]))
            j -= 1
        elif j == 0:
            diff.append(('delete', str1[i - 1]))
            i -= 1
        elif str1[i - 1] == str2[j - 1]:
            diff.append(('keep', str1[i - 1]))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            diff.append(('delete', str1[i - 1]))
            i -= 1
        else:
            diff.append(('insert', str2[j - 1]))
            j -= 1

    return list(reversed(diff))


if __name__ == "__main__":
    print("Longest Common Subsequence (LCS) Demonstration\n")
    print("=" * 70)

    # Example 1: Basic LCS
    print("Example 1: Basic LCS Calculation")
    print("-" * 70)

    test_pairs = [
        ("ABCDGH", "AEDFHR"),
        ("AGGTAB", "GXTXAYB"),
        ("HUMAN", "CHIMPANZEE"),
    ]

    for str1, str2 in test_pairs:
        length = lcs_length(str1, str2)
        lcs_str = lcs_string(str1, str2)

        print(f"String 1: {str1}")
        print(f"String 2: {str2}")
        print(f"LCS length: {length}")
        print(f"LCS: {lcs_str}")
        print()

    # Example 2: All possible LCS
    print("=" * 70)
    print("Example 2: Finding All Possible LCS")
    print("-" * 70)

    str1, str2 = "AGTGATG", "GTTAG"
    all_lcs_list = lcs_all(str1, str2)

    print(f"String 1: {str1}")
    print(f"String 2: {str2}")
    print(f"All LCS (length {lcs_length(str1, str2)}):")
    for lcs in all_lcs_list:
        print(f"  - {lcs}")

    # Example 3: LCS with different data types
    print("\n" + "=" * 70)
    print("Example 3: LCS with Lists")
    print("-" * 70)

    list1 = [1, 2, 3, 4, 5]
    list2 = [2, 4, 5, 6]
    lcs_list = lcs_generic(list1, list2)

    print(f"List 1: {list1}")
    print(f"List 2: {list2}")
    print(f"LCS: {lcs_list}")

    # Example 4: Diff utility
    print("\n" + "=" * 70)
    print("Example 4: String Diff (like diff utility)")
    print("-" * 70)

    original = "ABCDEFG"
    modified = "ABDXEFG"
    diff = diff_strings(original, modified)

    print(f"Original: {original}")
    print(f"Modified: {modified}")
    print(f"\nDiff:")

    for op, char in diff:
        if op == 'keep':
            print(f"  {char}")
        elif op == 'delete':
            print(f"- {char}")
        elif op == 'insert':
            print(f"+ {char}")

    # Example 5: DNA Sequence Analysis
    print("\n" + "=" * 70)
    print("Example 5: DNA Sequence Comparison")
    print("-" * 70)

    dna1 = "ACCGGTCGAGTGCGCGGAAGCCGGCCGAA"
    dna2 = "GTCGTTCGGAATGCCGTTGCTCTGTAAA"

    lcs_len = lcs_length(dna1, dna2)
    lcs_seq = lcs_string(dna1, dna2)
    similarity = (lcs_len / max(len(dna1), len(dna2))) * 100

    print(f"Sequence 1: {dna1}")
    print(f"Sequence 2: {dna2}")
    print(f"\nLCS length: {lcs_len}")
    print(f"LCS: {lcs_seq}")
    print(f"Similarity: {similarity:.2f}%")

    # Example 6: File Comparison Demo
    print("\n" + "=" * 70)
    print("Example 6: File Line Comparison")
    print("-" * 70)

    file1_lines = ["line1", "line2", "line3", "line4"]
    file2_lines = ["line1", "line2", "newline", "line4"]

    common_lines = lcs_generic(file1_lines, file2_lines)

    print("File 1 lines:", file1_lines)
    print("File 2 lines:", file2_lines)
    print(f"\nCommon lines: {common_lines}")
    print(f"Lines in common: {len(common_lines)}/{max(len(file1_lines), len(file2_lines))}")
