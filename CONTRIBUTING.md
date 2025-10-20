# Contributing to Algorithms and Data Structures

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

This project follows a simple code of conduct:
- Be respectful and constructive
- Focus on the technical merits of contributions
- Help create a welcoming environment for all contributors

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- **Clear title** describing the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Environment details** (Python version, OS, etc.)
- **Code sample** if applicable

### Suggesting Enhancements

For feature requests or enhancements:
- **Check existing issues** to avoid duplicates
- **Describe the use case** and why it's valuable
- **Propose an implementation** if you have ideas
- **Consider trade-offs** (complexity, performance, etc.)

### Contributing Code

#### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/algorithms-and-data-structures.git
cd algorithms-and-data-structures

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

#### Development Workflow

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Write your code** following our [style guidelines](#code-style)

3. **Write tests** for your implementation:
   - Add tests in the appropriate `tests/` subdirectory
   - Aim for high coverage (80%+)
   - Test edge cases and error conditions

4. **Run quality checks**:
   ```bash
   # Format code
   black algorithms/ tests/

   # Lint
   ruff check algorithms/ tests/

   # Type check
   mypy algorithms/

   # Run tests
   pytest

   # Check coverage
   pytest --cov=algorithms --cov-report=term-missing
   ```

5. **Commit your changes** with [conventional commits](#commit-messages):
   ```bash
   git add .
   git commit -m "feat: add Bellman-Ford algorithm implementation"
   ```

6. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

#### Code Style

We follow these conventions:

**Python Style (PEP 8)**
- Line length: 100 characters
- Indentation: 4 spaces
- Use Black for formatting: `black algorithms/ tests/`

**Type Hints**
```python
def dijkstra(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """All public functions must have type hints."""
    pass
```

**Docstrings (Google Style)**
```python
def algorithm_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    One-line summary of what the function does.

    More detailed description if needed. Explain the algorithm,
    its use cases, and any important considerations.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When this error occurs
        TypeError: When this error occurs

    Time Complexity: O(n log n)
    Space Complexity: O(n)

    Example:
        >>> result = algorithm_name(value1, value2)
        >>> print(result)
        expected_output
    """
```

**File Structure for Algorithms**
```
algorithms/category/algorithm_name.py
├── Module docstring with overview
├── Import statements
├── Main algorithm implementation
├── Helper functions (prefixed with _)
├── Common utilities (if applicable)
└── if __name__ == "__main__": demo
```

**Test Structure**
```python
"""
Unit tests for algorithm_name.

Test coverage:
- Basic functionality
- Edge cases
- Error handling
- Performance characteristics
"""

import pytest
from algorithms.category.algorithm_name import function


class TestAlgorithmBasic:
    """Test basic functionality."""

    def test_simple_case(self) -> None:
        """Test with simple input."""
        assert function(input) == expected


class TestAlgorithmEdgeCases:
    """Test edge cases."""

    def test_empty_input(self) -> None:
        """Test with empty input."""
        assert function([]) == []


class TestAlgorithmErrors:
    """Test error handling."""

    def test_invalid_input(self) -> None:
        """Test error on invalid input."""
        with pytest.raises(ValueError, match="error message"):
            function(invalid_input)
```

#### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

**Format**: `<type>(<scope>): <description>`

**Types**:
- `feat:` New feature or algorithm
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `perf:` Performance improvement
- `refactor:` Code refactoring
- `style:` Formatting, white-space
- `chore:` Maintenance tasks

**Examples**:
```bash
git commit -m "feat(graphs): add Bellman-Ford algorithm"
git commit -m "fix(dijkstra): handle disconnected nodes correctly"
git commit -m "docs: update README with new algorithms"
git commit -m "test(astar): add edge case tests for invalid heuristics"
git commit -m "perf(dijkstra): optimize priority queue operations"
```

### Pull Request Process

1. **Update documentation** if needed (README, docstrings)
2. **Ensure all tests pass** and coverage is maintained
3. **Update the README** if you added new algorithms
4. **Describe your changes** clearly in the PR description:
   - What does this PR do?
   - Why is this change needed?
   - How has it been tested?
   - Are there any breaking changes?

5. **Link related issues**: Use keywords like "Fixes #123" or "Closes #456"

6. **Be responsive** to code review feedback

#### PR Review Criteria

Your PR will be reviewed for:
- ✅ Correctness of implementation
- ✅ Test coverage (>80%)
- ✅ Code style and formatting
- ✅ Documentation quality
- ✅ Performance considerations
- ✅ No unnecessary dependencies

### Adding a New Algorithm

**Complete Checklist**:

- [ ] Implementation file in `algorithms/<category>/algorithm_name.py`
- [ ] Comprehensive docstring with complexity analysis
- [ ] Type hints for all functions
- [ ] Test file in `tests/<category>/test_algorithm_name.py`
- [ ] Tests for basic cases, edge cases, and errors
- [ ] 80%+ test coverage
- [ ] Example usage in `if __name__ == "__main__"` block
- [ ] README update with algorithm in the table
- [ ] Benchmark script (optional but encouraged)
- [ ] Visualization (optional but encouraged)

**Example Implementation Template**:

```python
"""
Algorithm Name.

Brief description of what the algorithm does and when to use it.

Time Complexity: O(...)
Space Complexity: O(...)

Example:
    >>> # Usage example
"""

from typing import Any

def algorithm_name(input: Any) -> Any:
    """
    One-line description.

    Detailed explanation of the algorithm, its approach,
    and any important considerations.

    Args:
        input: Description

    Returns:
        Description

    Raises:
        ValueError: When...

    Time Complexity: O(...)
    Space Complexity: O(...)
    """
    # Implementation
    pass


if __name__ == "__main__":
    # Demonstration
    print("Algorithm Demo:")
    result = algorithm_name(example_input)
    print(f"Result: {result}")
```

## Questions?

If you have questions about contributing:
- Open an issue with the `question` label
- Review existing issues and PRs for examples
- Check the README and code documentation

## Recognition

Contributors will be acknowledged in:
- Git commit history
- GitHub contributors page
- Project README (for significant contributions)

Thank you for contributing! 🎉
