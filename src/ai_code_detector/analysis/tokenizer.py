"""Code tokenization and basic analysis."""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TokenStats:
    """Statistics about tokenized code."""
    total_tokens: int
    unique_tokens: int
    token_entropy: float
    avg_token_length: float
    identifier_tokens: List[str]
    comment_tokens: List[str]


class CodeTokenizer:
    """Tokenizes source code into meaningful units."""

    # Simple regex patterns for different languages
    COMMENT_PATTERNS = {
        "python": [r"#.*?$", r'""".*?"""', r"'''.*?'''"],
        "javascript": [r"//.*?$", r"/\*.*?\*/"],
        "typescript": [r"//.*?$", r"/\*.*?\*/"],
        "go": [r"//.*?$", r"/\*.*?\*/"],
        "rust": [r"//.*?$", r"/\*.*?\*/"],
        "c": [r"//.*?$", r"/\*.*?\*/"],
        "cpp": [r"//.*?$", r"/\*.*?\*/"],
        "java": [r"//.*?$", r"/\*.*?\*/"],
        "ruby": [r"#.*?$"],
    }

    # String patterns
    STRING_PATTERNS = [
        r'"(?:[^"\\]|\\.)*"',  # Double quoted
        r"'(?:[^'\\]|\\.)*'",  # Single quoted
        r'`(?:[^`\\]|\\.)*`',  # Backticks
    ]

    # Identifier pattern
    IDENTIFIER_PATTERN = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'

    def __init__(self):
        """Initialize tokenizer."""
        pass

    def tokenize(self, code: str, language: str) -> TokenStats:
        """Tokenize code and compute statistics.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            TokenStats with tokenization results
        """
        # Extract comments
        comments = self.extract_comments(code, language)

        # Extract strings
        strings = self.extract_strings(code)

        # Extract identifiers
        identifiers = self.extract_identifiers(code)

        # Remove comments and strings for token analysis
        code_clean = code
        for comment in comments:
            code_clean = code_clean.replace(comment, " ")
        for string in strings:
            code_clean = code_clean.replace(string, " ")

        # Simple whitespace tokenization
        tokens = code_clean.split()
        tokens = [t for t in tokens if t.strip()]

        # Calculate entropy
        entropy = self._calculate_entropy(tokens)

        return TokenStats(
            total_tokens=len(tokens),
            unique_tokens=len(set(tokens)),
            token_entropy=entropy,
            avg_token_length=sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
            identifier_tokens=identifiers,
            comment_tokens=comments,
        )

    def extract_comments(self, code: str, language: str) -> List[str]:
        """Extract all comments from code.

        Args:
            code: Source code
            language: Programming language

        Returns:
            List of comment strings
        """
        comments = []
        patterns = self.COMMENT_PATTERNS.get(language, [])

        for pattern in patterns:
            flags = re.MULTILINE | re.DOTALL
            matches = re.findall(pattern, code, flags)
            comments.extend(matches)

        return comments

    def extract_strings(self, code: str) -> List[str]:
        """Extract all string literals from code.

        Args:
            code: Source code

        Returns:
            List of string literals
        """
        strings = []
        for pattern in self.STRING_PATTERNS:
            matches = re.findall(pattern, code)
            strings.extend(matches)
        return strings

    def extract_identifiers(self, code: str) -> List[str]:
        """Extract all identifiers from code.

        Args:
            code: Source code

        Returns:
            List of identifier names
        """
        matches = re.findall(self.IDENTIFIER_PATTERN, code)
        # Filter out keywords (simple heuristic: single-letter or very short common keywords)
        keywords = {
            "if", "else", "for", "while", "return", "def", "class",
            "import", "from", "as", "in", "is", "and", "or", "not",
            "try", "except", "finally", "with", "var", "let", "const",
            "function", "async", "await", "new", "this", "super",
        }
        return [m for m in matches if m not in keywords]

    def _calculate_entropy(self, tokens: List[str]) -> float:
        """Calculate Shannon entropy of token distribution.

        Args:
            tokens: List of tokens

        Returns:
            Entropy value
        """
        if not tokens:
            return 0.0

        from collections import Counter
        import math

        counts = Counter(tokens)
        total = len(tokens)

        entropy = 0.0
        for count in counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return entropy

    def analyze_comments(self, comments: List[str]) -> Dict[str, float]:
        """Analyze comment characteristics.

        Args:
            comments: List of comment strings

        Returns:
            Dict of comment metrics
        """
        if not comments:
            return {
                "avg_length": 0.0,
                "total_comments": 0,
                "avg_word_count": 0.0,
            }

        # Clean comments (remove comment markers)
        cleaned = []
        for comment in comments:
            c = re.sub(r'^[#/\*]+\s*', '', comment)
            c = re.sub(r'\*+/$', '', c)
            c = re.sub(r'^"""|"""$', '', c)
            c = re.sub(r"^'''|'''$", '', c)
            cleaned.append(c.strip())

        word_counts = [len(c.split()) for c in cleaned if c]

        return {
            "avg_length": sum(len(c) for c in cleaned) / len(cleaned) if cleaned else 0,
            "total_comments": len(comments),
            "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
        }
