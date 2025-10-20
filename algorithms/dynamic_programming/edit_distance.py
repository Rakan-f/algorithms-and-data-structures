"""
Edit Distance (Levenshtein Distance) Algorithm.

Edit distance measures the minimum number of single-character edits (insertions,
deletions, or substitutions) required to transform one string into another.
It's used in spell checking, DNA sequence alignment, and diff algorithms.

Time Complexity: O(mn) where m and n are lengths of the two strings
Space Complexity: O(mn) standard, O(min(m,n)) space-optimized

Example:
    >>> edit_distance("kitten", "sitting")
    3
    # kitten -> sitten (substitute k->s)
    # sitten -> sittin (substitute e->i)
    # sittin -> sitting (insert g)
"""

from typing import List, Tuple, Optional


def edit_distance(str1: str, str2: str) -> int:
    """
    Calculate the minimum edit distance between two strings.

    Uses dynamic programming to find the minimum number of operations
    (insert, delete, substitute) to transform str1 into str2.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Minimum number of edit operations needed

    Time Complexity: O(mn) where m, n are string lengths
    Space Complexity: O(mn) for the DP table

    Algorithm:
        dp[i][j] = minimum edits to transform str1[0:i] to str2[0:j]

        dp[i][j] = min(
            dp[i-1][j] + 1,      # Delete from str1
            dp[i][j-1] + 1,      # Insert into str1
            dp[i-1][j-1] + cost  # Substitute (cost=0 if chars match, 1 otherwise)
        )

    Example:
        >>> edit_distance("horse", "ros")
        3
        >>> edit_distance("intention", "execution")
        5
    """
    m, n = len(str1), len(str2)

    # Create DP table
    # dp[i][j] = edit distance between str1[0:i] and str2[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: converting empty string to/from str1/str2
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters from str1

    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters from str2

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match, no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Take minimum of three operations
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete from str1
                    dp[i][j - 1],      # Insert into str1
                    dp[i - 1][j - 1]   # Substitute
                )

    return dp[m][n]


def edit_distance_with_operations(str1: str, str2: str) -> Tuple[int, List[str]]:
    """
    Calculate edit distance and return the sequence of operations.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Tuple of (distance, operations) where operations is a list of strings
        describing each edit operation

    Time Complexity: O(mn)
    Space Complexity: O(mn)

    Example:
        >>> distance, ops = edit_distance_with_operations("kitten", "sitting")
        >>> distance
        3
        >>> ops
        ['Substitute k->s at position 0', 'Substitute e->i at position 4',
         'Insert g at position 6']
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    # Backtrack to find operations
    operations = []
    i, j = m, n

    while i > 0 or j > 0:
        if i == 0:
            # Only insertions left
            operations.append(f"Insert '{str2[j-1]}' at position {i}")
            j -= 1
        elif j == 0:
            # Only deletions left
            operations.append(f"Delete '{str1[i-1]}' at position {i-1}")
            i -= 1
        elif str1[i - 1] == str2[j - 1]:
            # Characters match, no operation
            i -= 1
            j -= 1
        else:
            # Find which operation was used
            delete_cost = dp[i - 1][j]
            insert_cost = dp[i][j - 1]
            substitute_cost = dp[i - 1][j - 1]

            min_cost = min(delete_cost, insert_cost, substitute_cost)

            if min_cost == substitute_cost:
                operations.append(
                    f"Substitute '{str1[i-1]}' -> '{str2[j-1]}' at position {i-1}"
                )
                i -= 1
                j -= 1
            elif min_cost == delete_cost:
                operations.append(f"Delete '{str1[i-1]}' at position {i-1}")
                i -= 1
            else:  # insert
                operations.append(f"Insert '{str2[j-1]}' at position {i}")
                j -= 1

    operations.reverse()
    return dp[m][n], operations


def edit_distance_space_optimized(str1: str, str2: str) -> int:
    """
    Calculate edit distance with O(min(m,n)) space complexity.

    Uses only two rows of the DP table instead of the full table.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Minimum number of edit operations

    Time Complexity: O(mn)
    Space Complexity: O(min(m,n))

    Example:
        >>> edit_distance_space_optimized("horse", "ros")
        3
    """
    # Ensure str2 is the shorter string (for space optimization)
    if len(str1) < len(str2):
        str1, str2 = str2, str1

    m, n = len(str1), len(str2)

    # Only need two rows: previous and current
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row[0] = i

        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(
                    prev_row[j],      # Delete
                    curr_row[j - 1],  # Insert
                    prev_row[j - 1]   # Substitute
                )

        prev_row, curr_row = curr_row, prev_row

    return prev_row[n]


def is_one_edit_away(str1: str, str2: str) -> bool:
    """
    Check if two strings are one edit operation away from each other.

    This is an optimized function for the specific case of checking
    if edit distance is exactly 1.

    Args:
        str1: First string
        str2: Second string

    Returns:
        True if strings are one edit away, False otherwise

    Time Complexity: O(n) where n is length of shorter string
    Space Complexity: O(1)

    Example:
        >>> is_one_edit_away("pale", "ple")
        True
        >>> is_one_edit_away("pales", "pale")
        True
        >>> is_one_edit_away("pale", "bale")
        True
        >>> is_one_edit_away("pale", "bake")
        False
    """
    len1, len2 = len(str1), len(str2)

    # Length difference greater than 1 means more than one edit
    if abs(len1 - len2) > 1:
        return False

    # Get shorter and longer strings
    if len1 < len2:
        shorter, longer = str1, str2
    else:
        shorter, longer = str2, str1

    i = j = 0
    found_difference = False

    while i < len(shorter) and j < len(longer):
        if shorter[i] != longer[j]:
            if found_difference:
                return False  # Second difference found

            found_difference = True

            # If same length, move both pointers (substitution)
            if len(shorter) == len(longer):
                i += 1

            # If different length, only move longer pointer (insertion/deletion)
            j += 1
        else:
            i += 1
            j += 1

    return True


def similarity_percentage(str1: str, str2: str) -> float:
    """
    Calculate similarity percentage between two strings based on edit distance.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity as percentage (0-100)

    Time Complexity: O(mn)
    Space Complexity: O(min(m,n))

    Example:
        >>> similarity_percentage("kitten", "sitting")
        57.14
        >>> similarity_percentage("hello", "hello")
        100.0
    """
    if not str1 and not str2:
        return 100.0

    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 100.0

    distance = edit_distance_space_optimized(str1, str2)
    similarity = ((max_len - distance) / max_len) * 100

    return round(similarity, 2)


if __name__ == "__main__":
    print("Edit Distance (Levenshtein Distance) Demonstration\n")
    print("=" * 70)

    # Example 1: Classic edit distance
    examples = [
        ("kitten", "sitting"),
        ("horse", "ros"),
        ("intention", "execution"),
        ("saturday", "sunday"),
        ("", "abc"),
        ("abc", ""),
        ("same", "same")
    ]

    print("Example 1: Edit Distance Calculation")
    print("-" * 70)

    for str1, str2 in examples:
        dist = edit_distance(str1, str2)
        similarity = similarity_percentage(str1, str2)

        print(f"'{str1}' -> '{str2}'")
        print(f"  Edit distance: {dist}")
        print(f"  Similarity: {similarity}%")
        print()

    # Example 2: With operations
    print("=" * 70)
    print("Example 2: Edit Distance with Operations")
    print("-" * 70)

    test_pairs = [
        ("kitten", "sitting"),
        ("horse", "ros")
    ]

    for str1, str2 in test_pairs:
        distance, operations = edit_distance_with_operations(str1, str2)

        print(f"Transform '{str1}' -> '{str2}'")
        print(f"Edit distance: {distance}")
        print(f"Operations:")
        for i, op in enumerate(operations, 1):
            print(f"  {i}. {op}")
        print()

    # Example 3: One edit away check
    print("=" * 70)
    print("Example 3: One Edit Away Check")
    print("-" * 70)

    one_edit_tests = [
        ("pale", "ple"),    # Delete
        ("pales", "pale"),  # Insert
        ("pale", "bale"),   # Substitute
        ("pale", "bake"),   # Two edits
    ]

    for str1, str2 in one_edit_tests:
        result = is_one_edit_away(str1, str2)
        print(f"'{str1}' and '{str2}': {result}")

    # Example 4: Spell checker demo
    print("\n" + "=" * 70)
    print("Example 4: Spell Checker Demo")
    print("-" * 70)

    dictionary = ["hello", "world", "python", "algorithm", "programming"]
    misspelled = "progaming"

    print(f"Misspelled word: '{misspelled}'")
    print(f"Dictionary: {dictionary}")
    print(f"\nSuggestions (sorted by similarity):")

    suggestions = []
    for word in dictionary:
        dist = edit_distance(misspelled, word)
        sim = similarity_percentage(misspelled, word)
        suggestions.append((word, dist, sim))

    suggestions.sort(key=lambda x: x[1])

    for word, dist, sim in suggestions:
        print(f"  {word}: distance={dist}, similarity={sim}%")

    # Example 5: DNA sequence alignment demo
    print("\n" + "=" * 70)
    print("Example 5: DNA Sequence Alignment")
    print("-" * 70)

    seq1 = "AGGTAB"
    seq2 = "GXTXAYB"

    dist = edit_distance(seq1, seq2)
    print(f"Sequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")
    print(f"Edit distance: {dist}")
    print(f"Similarity: {similarity_percentage(seq1, seq2)}%")
