from setuptools import setup, find_packages

setup(
    name="ai-code-detector",
    version="0.1.0",
    description="Probabilistic AI-generated code detection for GitHub repositories",
    author="AI Code Detector Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "GitPython>=3.1.40",
        "pygit2>=1.13.0",
        "tree-sitter>=0.20.4",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
        "jinja2>=3.1.2",
        "markdown>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "mlx": [
            "mlx>=0.0.10",
            "mlx-lm>=0.0.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-code-detector=cli:main",
        ],
    },
    python_requires=">=3.9",
)
