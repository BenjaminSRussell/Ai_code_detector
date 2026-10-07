"""Tree-sitter backends for structural metrics (#7).

Optional: ``pip install 'ai-code-detector[treesitter]'``. When the grammars
are missing, ``available()`` is False and the factory falls back to the
builtin Python ``ast`` parser for Python and to the regex-only path for
everything else.

Each language produces the same ``FileAST`` that ``PythonASTParser`` does
(top-level functions, classes with methods, imported names), so
``StructuralAnalyzer`` uses one code path for every language. The parsers
also fill ``FunctionInfo.unreachable``, computed from the syntax tree
(a statement after ``return`` / ``throw`` / ``break`` / ``continue`` in
the same block), which the text heuristic can't do for brace languages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

from .ast_parser import ClassInfo, FileAST, FunctionInfo

try:  # pragma: no cover - import guard
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
except ImportError:  # pragma: no cover
    HAS_TREE_SITTER = False


def _load_language(lang: str, suffix: str = ""):
    if not HAS_TREE_SITTER:
        return None
    try:
        if lang == "python":
            import tree_sitter_python as m
            return Language(m.language())
        if lang == "javascript":
            import tree_sitter_javascript as m
            return Language(m.language())
        if lang == "typescript":
            import tree_sitter_typescript as m
            return Language(m.language_tsx() if suffix == ".tsx" else m.language_typescript())
        if lang == "go":
            import tree_sitter_go as m
            return Language(m.language())
        if lang == "rust":
            import tree_sitter_rust as m
            return Language(m.language())
    except ImportError:
        return None
    return None


SPECS: Dict[str, Dict[str, Set[str]]] = {
    "python": {
        "function": {"function_definition"},
        "class": {"class_definition"},
        "decision": {"if_statement", "elif_clause", "for_statement", "while_statement", "except_clause",
                     "boolean_operator"},
        "terminal": {"return_statement", "raise_statement", "break_statement", "continue_statement"},
        "block": {"block"},
    },
    "javascript": {
        "function": {"function_declaration", "generator_function_declaration", "method_definition"},
        "class": {"class_declaration", "class"},
        "decision": {"if_statement", "for_statement", "for_in_statement", "while_statement", "do_statement",
                     "catch_clause", "ternary_expression", "switch_case"},
        "terminal": {"return_statement", "throw_statement", "break_statement", "continue_statement"},
        "block": {"statement_block", "switch_case"},
    },
    "go": {
        "function": {"function_declaration", "method_declaration"},
        "class": set(),
        "decision": {"if_statement", "for_statement", "expression_case", "type_case", "communication_case"},
        "terminal": {"return_statement", "break_statement", "continue_statement", "goto_statement"},
        "block": {"block", "statement_list"},
    },
    "rust": {
        "function": {"function_item"},
        "class": {"impl_item", "trait_item"},
        "decision": {"if_expression", "for_expression", "while_expression", "loop_expression", "match_arm"},
        "terminal": {"return_expression", "break_expression", "continue_expression"},
        "block": {"block"},
    },
}
SPECS["typescript"] = {**SPECS["javascript"],
                       "class": SPECS["javascript"]["class"] | {"abstract_class_declaration"}}
BOOL_OPS = {"&&", "||", "??"}
_COMMENT = {"comment", "line_comment", "block_comment"}


def available(language: str = "typescript") -> bool:
    return _load_language(language, "") is not None


class TreeSitterParser:
    """Language-agnostic FileAST builder on top of a tree-sitter grammar."""

    def __init__(self, language: str):
        if language not in SPECS:
            raise ValueError(f"no tree-sitter spec for {language}")
        self.language = language
        self.spec = SPECS[language]
        self._parsers: Dict[str, object] = {}

    def _parser(self, suffix: str):
        key = suffix if self.language == "typescript" else ""
        if key not in self._parsers:
            lang = _load_language(self.language, suffix)
            if lang is None:
                raise RuntimeError(f"tree-sitter grammar for {self.language} is not installed")
            self._parsers[key] = Parser(lang)
        return self._parsers[key]

    # ---- public ---------------------------------------------------------

    def parse_file(self, file_path: Path, code: str) -> FileAST:
        src = code.encode("utf-8", "surrogatepass")
        tree = self._parser(Path(file_path).suffix).parse(src)
        self._src = src
        functions: List[FunctionInfo] = []
        classes: List[ClassInfo] = []
        self._collect(tree.root_node, functions, classes, in_scope=False)
        return FileAST(file_path=Path(file_path), language=self.language, functions=functions,
                       classes=classes, imports=self._imports(tree.root_node), global_vars=[])

    # ---- tree walking ---------------------------------------------------

    def _text(self, node) -> str:
        return self._src[node.start_byte:node.end_byte].decode("utf-8", "ignore") if node is not None else ""

    def _collect(self, node, functions, classes, in_scope: bool):
        for child in node.named_children:
            t = child.type
            if t in self.spec["class"]:
                classes.append(self._class(child))
            elif t in self.spec["function"] or self._is_named_arrow(child):
                if not in_scope:
                    functions.append(self._function(child))
                # nested functions are not top-level (matches PythonASTParser)
            else:
                self._collect(child, functions, classes, in_scope)

    def _is_named_arrow(self, node) -> bool:
        # const f = (x) => ...  /  const f = function () {...}
        if self.language not in ("javascript", "typescript") or node.type != "variable_declarator":
            return False
        value = node.child_by_field_name("value")
        return value is not None and value.type in ("arrow_function", "function_expression", "function")

    def _class(self, node) -> ClassInfo:
        name_node = node.child_by_field_name("name") or node.child_by_field_name("type")
        methods: List[FunctionInfo] = []

        def walk(n):
            for c in n.named_children:
                if c.type in self.spec["function"]:
                    methods.append(self._function(c))
                elif c.type not in self.spec["class"]:
                    walk(c)
        walk(node)
        return ClassInfo(name=self._text(name_node), start_line=node.start_point[0] + 1,
                         end_line=node.end_point[0] + 1, methods=methods, base_classes=[],
                         docstring=self._doc(node), decorators=[])

    def _function(self, node) -> FunctionInfo:
        fn = node
        if node.type == "variable_declarator":
            fn = node.child_by_field_name("value")
        name = self._text(node.child_by_field_name("name"))
        params_node = fn.child_by_field_name("parameters") or fn.child_by_field_name("parameter")
        params = [self._text(p) for p in (params_node.named_children if params_node is not None else [])
                  if p.type not in _COMMENT]
        ret = fn.child_by_field_name("return_type") or fn.child_by_field_name("result")
        body = fn.child_by_field_name("body")
        doc_anchor = node.parent if node.type == "variable_declarator" else node
        return FunctionInfo(
            name=name, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
            params=params, returns_type=self._text(ret).lstrip(":").strip() or None,
            docstring=self._doc(doc_anchor, body), decorators=[],
            is_async=self._text(fn).lstrip().startswith("async"),
            cyclomatic_complexity=self._complexity(fn), code=self._text(node),
            unreachable=self._unreachable(body) if body is not None else False,
        )

    def _complexity(self, fn) -> int:
        count = 1
        stack = list(fn.named_children)
        while stack:
            n = stack.pop()
            if n.type in self.spec["decision"]:
                count += 1
            elif n.type in ("binary_expression", "boolean_operator"):
                op = n.child_by_field_name("operator")
                if n.type == "boolean_operator" or (op is not None and self._text(op) in BOOL_OPS):
                    count += 1
            if n.type in self.spec["function"] and n is not fn:
                continue  # nested functions have their own complexity
            stack.extend(n.named_children)
        return count

    def _unreachable(self, body) -> bool:
        stack = [body]
        while stack:
            n = stack.pop()
            if n.type in self.spec["block"]:
                stmts = [c for c in n.named_children if c.type not in _COMMENT]
                for i, s in enumerate(stmts[:-1]):
                    if self._is_terminal(s) and n.type != "switch_case" or (
                            n.type == "switch_case" and self._is_terminal(s) and i > 0):
                        return True
            stack.extend(c for c in n.named_children if c.type not in self.spec["function"])
        return False

    def _is_terminal(self, stmt) -> bool:
        if stmt.type in self.spec["terminal"]:
            return True
        # rust: `return x;` is an expression_statement wrapping return_expression
        return stmt.type == "expression_statement" and bool(stmt.named_children) and \
            stmt.named_children[0].type in self.spec["terminal"]

    def _doc(self, node, body=None) -> Optional[str]:
        if self.language == "python" and body is not None and body.named_children:
            first = body.named_children[0]
            if first.type == "expression_statement" and first.named_children and \
                    first.named_children[0].type == "string":
                return self._text(first.named_children[0]).strip("\"' \n")
            return None
        # brace languages: contiguous comments directly above the declaration
        anchor = node
        if anchor.parent is not None and anchor.parent.type in ("export_statement", "decorated_definition"):
            anchor = anchor.parent
        lines = []
        prev = anchor.prev_sibling
        expected_row = anchor.start_point[0]
        while prev is not None and prev.type in _COMMENT and prev.end_point[0] >= expected_row - 1:
            lines.insert(0, self._text(prev).rstrip())
            expected_row = prev.start_point[0]
            prev = prev.prev_sibling
        if not lines:
            return None
        text = "\n".join(lines)
        if self.language in ("javascript", "typescript") and not text.lstrip().startswith("/**"):
            return None  # only JSDoc counts as documentation
        if self.language == "rust" and not text.lstrip().startswith("///"):
            return None
        return text

    def _imports(self, root) -> List[str]:
        names: List[str] = []
        stack = [root]
        while stack:
            n = stack.pop()
            t = n.type
            if self.language == "python" and t in ("import_statement", "import_from_statement"):
                for c in n.named_children:
                    if c.type == "aliased_import":
                        names.append(self._text(c.child_by_field_name("alias")))
                    elif c.type == "dotted_name" and c != n.child_by_field_name("module_name"):
                        names.append(self._text(c))
                continue
            if self.language in ("javascript", "typescript") and t == "import_clause":
                for c in n.named_children:
                    if c.type == "identifier":
                        names.append(self._text(c))
                    elif c.type == "namespace_import":
                        names += [self._text(x) for x in c.named_children if x.type == "identifier"]
                    elif c.type == "named_imports":
                        for spec in c.named_children:
                            if spec.type == "import_specifier":
                                alias = spec.child_by_field_name("alias") or spec.child_by_field_name("name")
                                names.append(self._text(alias))
                continue
            if self.language == "go" and t == "import_spec":
                alias = n.child_by_field_name("name")
                path = self._text(n.child_by_field_name("path")).strip('"`')
                names.append(self._text(alias) if alias is not None else path.rsplit("/", 1)[-1])
                continue
            if self.language == "rust" and t == "use_declaration":
                names += self._rust_use_names(n.child_by_field_name("argument"))
                continue
            stack.extend(reversed(n.named_children))
        return [x for x in names if x and x not in ("_", ".", "*", "self")]

    def _rust_use_names(self, n) -> List[str]:
        if n is None:
            return []
        if n.type == "use_as_clause":
            return [self._text(n.child_by_field_name("alias"))]
        if n.type == "scoped_identifier":
            return [self._text(n.child_by_field_name("name"))]
        if n.type == "identifier":
            return [self._text(n)]
        if n.type in ("use_list", "scoped_use_list"):
            out: List[str] = []
            lst = n.child_by_field_name("list") if n.type == "scoped_use_list" else n
            for c in (lst.named_children if lst is not None else []):
                out += self._rust_use_names(c)
            return out
        return []
