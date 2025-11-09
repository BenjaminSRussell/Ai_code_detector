"""Natural language explanation generator using Qwen (Phase 3).

Generates human-readable explanations for AI detection results.
"""

from typing import Dict, List, Optional
from pathlib import Path
import json

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


class ExplanationGenerator:
    """Base class for generating explanations."""

    def explain(
        self,
        code: str,
        ai_probability: float,
        features: Dict[str, float],
        top_n: int = 3,
    ) -> str:
        """Generate explanation for detection result.

        Args:
            code: Source code being analyzed
            ai_probability: AI detection probability
            features: Dict of feature name -> value
            top_n: Number of top features to explain

        Returns:
            Natural language explanation
        """
        raise NotImplementedError


class QwenExplainer(ExplanationGenerator):
    """Qwen-based explanation generator.

    Uses Qwen model via MLX to generate natural language explanations
    for why code was flagged as AI-generated.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """Initialize Qwen explainer.

        Args:
            model_path: Path to Qwen model (uses default if None)
            max_tokens: Maximum tokens in explanation
            temperature: Sampling temperature
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

        if HAS_MLX_LM and model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load Qwen model via MLX.

        Args:
            model_path: Path to model
        """
        print(f"Loading Qwen model from {model_path}...")
        try:
            self.model, self.tokenizer = load(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Falling back to template-based explanations")

    def explain(
        self,
        code: str,
        ai_probability: float,
        features: Dict[str, float],
        top_n: int = 3,
    ) -> str:
        """Generate explanation using Qwen.

        Args:
            code: Source code
            ai_probability: AI probability
            features: Feature dict
            top_n: Number of features to explain

        Returns:
            Natural language explanation
        """
        if self.model is None:
            # Fallback to template-based
            return self._template_explanation(code, ai_probability, features, top_n)

        # Build prompt for Qwen
        prompt = self._build_prompt(code, ai_probability, features, top_n)

        # Generate explanation
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temp=self.temperature,
            )
            return response.strip()
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return self._template_explanation(code, ai_probability, features, top_n)

    def _build_prompt(
        self,
        code: str,
        ai_probability: float,
        features: Dict[str, float],
        top_n: int,
    ) -> str:
        """Build prompt for Qwen.

        Args:
            code: Source code
            ai_probability: AI probability
            features: Feature dict
            top_n: Number of features

        Returns:
            Prompt string
        """
        # Get top features
        top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Format features
        feature_list = "\n".join([
            f"  - {name.replace('_', ' ').title()}: {value:.2f}"
            for name, value in top_features
        ])

        # Truncate code if too long
        if len(code) > 500:
            code_snippet = code[:500] + "\n... (truncated)"
        else:
            code_snippet = code

        prompt = f"""You are an AI code detection expert. Analyze the following code snippet and explain why it was flagged as potentially AI-generated.

Code snippet:
```
{code_snippet}
```

AI Detection Probability: {ai_probability*100:.1f}%

Top indicators:
{feature_list}

Provide a clear, concise explanation (2-3 sentences) for why this code appears to be AI-generated, focusing on the specific patterns detected. Be specific and reference actual code patterns.

Explanation:"""

        return prompt

    def _template_explanation(
        self,
        code: str,
        ai_probability: float,
        features: Dict[str, float],
        top_n: int,
    ) -> str:
        """Generate template-based explanation without LLM.

        Args:
            code: Source code
            ai_probability: AI probability
            features: Feature dict
            top_n: Number of features

        Returns:
            Template-based explanation
        """
        # Get top features
        top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:top_n]

        if not top_features:
            return f"This code has an AI probability of {ai_probability*100:.1f}% based on overall patterns."

        # Build explanation
        explanations = []

        for feature_name, value in top_features:
            explanation = self._explain_feature(feature_name, value, code)
            if explanation:
                explanations.append(explanation)

        if not explanations:
            return f"This code shows AI-like patterns with {ai_probability*100:.1f}% probability."

        # Combine explanations
        intro = f"This code has a {ai_probability*100:.1f}% AI probability. "

        if len(explanations) == 1:
            return intro + explanations[0]
        elif len(explanations) == 2:
            return intro + explanations[0] + " Additionally, " + explanations[1].lower()
        else:
            main = explanations[0]
            others = ", ".join(explanations[1:-1])
            last = explanations[-1]
            return intro + main + f" The code also shows {others.lower()}, and {last.lower()}"

    def _explain_feature(self, feature_name: str, value: float, code: str) -> Optional[str]:
        """Explain a specific feature.

        Args:
            feature_name: Name of feature
            value: Feature value
            code: Source code

        Returns:
            Explanation string or None
        """
        explanations = {
            'boilerplate_comments': (
                "It contains boilerplate documentation with generic phrases like "
                "'This function' and 'Args:', which are common in AI-generated code."
            ),
            'generic_naming': (
                f"It uses generic variable names like 'result', 'data', and 'temp' "
                f"({value*100:.0f}% of identifiers), which AI assistants frequently employ."
            ),
            'code_duplication': (
                "It shows high code duplication with similar patterns repeated across functions, "
                "a telltale sign of AI scaffolding."
            ),
            'over_explained_functions': (
                f"It has simple functions with overly verbose documentation "
                f"({value*100:.0f}% of functions), typical of AI over-explanation."
            ),
            'generic_exceptions': (
                "It uses generic exception handling (catch-all 'except Exception' blocks) "
                "instead of specific exception types, which AI tends to generate."
            ),
            'unused_functions': (
                f"It contains unused or dead code ({value*100:.0f}% of functions are never called), "
                "suggesting AI-generated scaffolding that wasn't fully integrated."
            ),
            'print_error_patterns': (
                "It uses print-based error handling (e.g., 'print(f\"An error occurred: {e}\")') "
                "rather than proper logging, a common AI pattern."
            ),
        }

        return explanations.get(feature_name)


class TemplateExplainer(ExplanationGenerator):
    """Simple template-based explainer without LLM dependency.

    Faster and more predictable than LLM-based explanation,
    but less flexible and contextual.
    """

    def __init__(self):
        """Initialize template explainer."""
        pass

    def explain(
        self,
        code: str,
        ai_probability: float,
        features: Dict[str, float],
        top_n: int = 3,
    ) -> str:
        """Generate template-based explanation.

        Args:
            code: Source code
            ai_probability: AI probability
            features: Feature dict
            top_n: Number of features

        Returns:
            Explanation string
        """
        qwen_explainer = QwenExplainer()  # Will use template fallback
        return qwen_explainer._template_explanation(code, ai_probability, features, top_n)


class BatchExplainer:
    """Batch explanation generator for multiple code snippets.

    Optimizes explanation generation for large codebases.
    """

    def __init__(self, explainer: ExplanationGenerator):
        """Initialize batch explainer.

        Args:
            explainer: Base explainer to use
        """
        self.explainer = explainer

    def explain_batch(
        self,
        codes: List[str],
        probabilities: List[float],
        features_list: List[Dict[str, float]],
        top_n: int = 3,
    ) -> List[str]:
        """Generate explanations for batch of code snippets.

        Args:
            codes: List of source code strings
            probabilities: List of AI probabilities
            features_list: List of feature dicts
            top_n: Number of features per explanation

        Returns:
            List of explanations
        """
        explanations = []

        for code, prob, features in zip(codes, probabilities, features_list):
            explanation = self.explainer.explain(code, prob, features, top_n)
            explanations.append(explanation)

        return explanations


def get_explainer(
    backend: str = "qwen",
    model_path: Optional[str] = None,
    **kwargs
) -> ExplanationGenerator:
    """Factory function to get explainer.

    Args:
        backend: Explainer backend ('qwen', 'template')
        model_path: Path to model (for qwen backend)
        **kwargs: Additional arguments

    Returns:
        ExplanationGenerator instance
    """
    if backend == "qwen":
        if not HAS_MLX_LM and model_path:
            print("Warning: MLX-LM not available, falling back to template explainer")
            return TemplateExplainer()
        return QwenExplainer(model_path=model_path, **kwargs)
    elif backend == "template":
        return TemplateExplainer()
    else:
        raise ValueError(f"Unknown backend: {backend}")
