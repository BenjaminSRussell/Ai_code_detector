"""AST-based code parsing for structural analysis."""

import ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FunctionInfo:
    """Information about a function or method."""
    name: str
    start_line: int
    end_line: int
    params: List[str]
    returns_type: Optional[str]
    docstring: Optional[str]
    decorators: List[str]
    is_async: bool
    cyclomatic_complexity: int
    code: str


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    start_line: int
    end_line: int
    methods: List[FunctionInfo]
    base_classes: List[str]
    docstring: Optional[str]
    decorators: List[str]


@dataclass
class FileAST:
    """AST analysis results for a file."""
    file_path: Path
    language: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[str]
    global_vars: List[str]


class PythonASTParser:
    """Parser for Python AST analysis."""

    def parse_file(self, file_path: Path, code: str) -> FileAST:
        """Parse Python file and extract structural information.

        Args:
            file_path: Path to file
            code: Source code content

        Returns:
            FileAST with parsed information
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            # Re-raise the exception to be caught by the caller
            raise e

        functions = []
        classes = []
        imports = []
        global_vars = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Only top-level functions (not methods inside classes)
                if self._is_top_level(node, tree):
                    func_info = self._parse_function(node, code)
                    functions.append(func_info)

            elif isinstance(node, ast.ClassDef):
                class_info = self._parse_class(node, code)
                classes.append(class_info)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(self._parse_import(node))

            elif isinstance(node, ast.Assign):
                if self._is_top_level(node, tree):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            global_vars.append(target.id)

        return FileAST(
            file_path=file_path,
            language="python",
            functions=functions,
            classes=classes,
            imports=imports,
            global_vars=global_vars,
        )

    def _is_top_level(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if node is at module level."""
        for parent_node in ast.walk(tree):
            if isinstance(parent_node, ast.ClassDef):
                for child in ast.walk(parent_node):
                    if child is node and child is not parent_node:
                        return False
        return True

    def _parse_function(self, node: ast.FunctionDef, code: str) -> FunctionInfo:
        """Parse function node into FunctionInfo."""
        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract parameters
        params = [arg.arg for arg in node.args.args]

        # Extract decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        # Get return type annotation
        returns_type = None
        if node.returns:
            returns_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

        # Calculate cyclomatic complexity
        complexity = self._calculate_complexity(node)

        # Extract code
        lines = code.split('\n')
        func_code = '\n'.join(lines[node.lineno - 1:node.end_lineno])

        return FunctionInfo(
            name=node.name,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            params=params,
            returns_type=returns_type,
            docstring=docstring,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            cyclomatic_complexity=complexity,
            code=func_code,
        )

    def _parse_class(self, node: ast.ClassDef, code: str) -> ClassInfo:
        """Parse class node into ClassInfo."""
        docstring = ast.get_docstring(node)

        # Extract base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif hasattr(ast, 'unparse'):
                base_classes.append(ast.unparse(base))

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._parse_function(item, code)
                methods.append(method_info)

        # Extract decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        return ClassInfo(
            name=node.name,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            methods=methods,
            base_classes=base_classes,
            docstring=docstring,
            decorators=decorators,
        )

    def _parse_import(self, node: ast.AST) -> List[str]:
        """Parse import statement."""
        imports = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports

    def _get_decorator_name(self, node: ast.AST) -> str:
        """Get decorator name as string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
        elif hasattr(ast, 'unparse'):
            return ast.unparse(node)
        return str(node)

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function.

        Simplified version: count decision points.
        """
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # Each 'and'/'or' adds complexity
                complexity += len(child.values) - 1

        return complexity


class ASTParserFactory:
    """Factory for creating language-specific AST parsers."""

    @staticmethod
    def get_parser(language: str):
        """Get parser for language.

        Args:
            language: Programming language name

        Returns:
            Parser instance or None if not supported
        """
        if language == "python":
            return PythonASTParser()
        else:
            # For Phase 1, only Python is supported
            # Phase 2+ can add tree-sitter based parsers for other languages
            return None
