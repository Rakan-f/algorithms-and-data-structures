"""Setup configuration for algorithms-and-data-structures package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="algorithms-and-data-structures",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-quality implementations of classic CS algorithms and data structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/algorithms-and-data-structures",
    packages=find_packages(exclude=["tests*", "benchmarks*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "ruff>=0.1.0",
        ],
    },
)
