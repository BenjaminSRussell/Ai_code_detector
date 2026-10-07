"""Tree-sitter structural backends (#7).

Tests marked ``treesitter`` need the optional grammars
(``pip install ".[treesitter]"``) and are skipped, with that reason, when
the grammars are missing. The CI job ``test-treesitter`` installs them.
"""
from dataclasses import asdict
from pathlib import Path

import pytest

from ai_code_detector.analysis import tree_sitter_parser as tsp
from ai_code_detector.analysis.ast_parser import (ASTParserFactory, ClassInfo, FileAST, FunctionInfo,
                                                  PythonASTParser)
from ai_code_detector.analysis.metrics_structural import StructuralAnalyzer

FIX = Path(__file__).resolve().parent / "fixtures" / "tree_sitter"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def needs(lang):
    return pytest.mark.skipif(not tsp.available(lang),
                              reason=f"tree-sitter grammar for {lang} not installed (pip install '.[treesitter]')")


def parse(name, lang, backend="auto"):
    code = (FIX / name).read_text()
    return code, ASTParserFactory.get_parser(lang, backend).parse_file(FIX / name, code)


def by_name(file_ast):
    return {f.name: f for f in file_ast.functions + [m for c in file_ast.classes for m in c.methods]}


# ---- always run --------------------------------------------------------------

def test_backend_selection_and_fallbacks(monkeypatch):
    assert isinstance(ASTParserFactory.get_parser("python"), PythonASTParser)  # default: unchanged behavior
    assert ASTParserFactory.get_parser("typescript", "python") is None
    with pytest.raises(ValueError, match="ast_backend"):
        ASTParserFactory.get_parser("python", "clang")
    monkeypatch.setattr(tsp, "available", lambda lang="typescript": False)
    assert ASTParserFactory.get_parser("typescript", "auto") is None  # regex fallback
    assert ASTParserFactory.get_parser("go", "tree_sitter") is None
    assert isinstance(ASTParserFactory.get_parser("python", "tree_sitter"), PythonASTParser)
    assert ASTParserFactory.get_parser("java", "auto") is None


def test_detector_validates_backend(tmp_path):
    import yaml
    from ai_code_detector.detector_enhanced import EnhancedAICodeDetector
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "src/ai_code_detector/configs/default.yaml").read_text())
    assert cfg["analysis"]["ast_backend"] == "auto"
    cfg["analysis"]["ast_backend"] = "regex"
    bad = tmp_path / "c.yaml"
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="ast_backend"):
        EnhancedAICodeDetector(config_path=bad, use_ml=False, use_explanations=False)
    assert EnhancedAICodeDetector(use_ml=False, use_explanations=False).ast_backend == "auto"


def test_structural_metrics_do_not_mutate_the_ast():
    fn = lambda name: FunctionInfo(name=name, start_line=1, end_line=2, params=[], returns_type=None,
                                   docstring=None, decorators=[], is_async=False, cyclomatic_complexity=2,
                                   code=f"def {name}():\n    return 1\n")
    ast_ = FileAST(file_path=Path("x.py"), language="python", functions=[fn("a")],
                   classes=[ClassInfo(name="K", start_line=1, end_line=5, methods=[fn("m1"), fn("m2")],
                                      base_classes=[], docstring=None, decorators=[])],
                   imports=[], global_vars=[])
    StructuralAnalyzer().analyze_file("def a(): pass", "python", ast_)
    assert [f.name for f in ast_.functions] == ["a"]  # used to grow by 2 methods per metric


# ---- tree-sitter -------------------------------------------------------------

@pytest.mark.treesitter
@needs("typescript")
def test_typescript_fixture():
    code, a = parse("service.ts", "typescript")
    assert a.language == "typescript" and [c.name for c in a.classes] == ["UserCache"]
    assert a.imports == ["readFile", "path", "axios", "httpGet", "unusedThing"]  # local (aliased) names
    f = by_name(a)
    assert {k: v.cyclomatic_complexity for k, v in f.items()} == \
        {"loadUser": 2, "pick": 3, "classify": 4, "get": 1, "refresh": 4}
    assert f["pick"].unreachable is True and not any(v.unreachable for k, v in f.items() if k != "pick")
    assert f["loadUser"].docstring.startswith("/**") and f["get"].docstring and f["refresh"].docstring is None
    assert f["loadUser"].is_async and f["loadUser"].returns_type == "Promise<string>"
    assert f["pick"].params == ["xs: number[]"]


@pytest.mark.treesitter
@needs("typescript")
def test_typescript_metrics_differ_from_regex_baseline_as_documented():
    code, a = parse("service.ts", "typescript")
    sa = StructuralAnalyzer()
    regex = asdict(sa.analyze_file(code, "typescript", None))
    tree = asdict(sa.analyze_file(code, "typescript", a))
    # The regex path can only see text patterns: every AST-derived metric is 0.
    for k in ("avg_cyclomatic_complexity", "complexity_to_docstring_ratio", "try_except_ratio",
              "unused_import_ratio", "unreachable_code_score"):
        assert regex[k] == 0.0
    assert tree["avg_cyclomatic_complexity"] == pytest.approx(14 / 5)  # 2+3+4+1+4 over 5 functions
    assert tree["try_except_ratio"] == pytest.approx(1 / 5)            # loadUser
    assert tree["unused_import_ratio"] == pytest.approx(1 / 5)         # unusedThing
    assert tree["unreachable_code_score"] == pytest.approx(1 / 5)      # pick: statement after return
    assert tree["complexity_to_docstring_ratio"] > 0
    # text-only metrics are identical on both paths
    for k in ("generic_exception_ratio", "print_error_pattern_score", "missing_cleanup_score"):
        assert tree[k] == regex[k]


@pytest.mark.treesitter
@needs("go")
def test_go_fixture():
    code, a = parse("server.go", "go")
    assert a.imports == ["errors", "fmt", "str", "os"]
    f = by_name(a)
    assert {k: v.cyclomatic_complexity for k, v in f.items()} == {"Greet": 4, "Stop": 1, "route": 3}
    assert f["Stop"].unreachable and not f["Greet"].unreachable and not f["route"].unreachable
    assert f["Greet"].docstring.splitlines() == ["// Greet returns a greeting.", "// It upper-cases the name."]
    assert f["Stop"].docstring is None and f["Greet"].returns_type == "(string, error)"
    m = StructuralAnalyzer().analyze_file(code, "go", a)
    assert m.unused_import_ratio == pytest.approx(1 / 4)  # os
    assert m.unreachable_code_score == pytest.approx(1 / 3)


@pytest.mark.treesitter
@needs("rust")
def test_rust_fixture():
    code, a = parse("lib.rs", "rust")
    assert a.imports == ["HashMap", "Set", "io"] and [c.name for c in a.classes] == ["Store"]
    f = by_name(a)
    assert f["classify"].cyclomatic_complexity == 5 and f["classify"].docstring == "/// Classify a number."
    assert f["get"].unreachable is True and f["classify"].unreachable is False
    m = StructuralAnalyzer().analyze_file(code, "rust", a)
    assert m.unused_import_ratio == pytest.approx(2 / 3)  # Set, io (word match: `io` is not in `Option`)


@pytest.mark.treesitter
@needs("python")
def test_python_tree_sitter_matches_builtin_ast_on_examples():
    for path in EXAMPLES.glob("*.py"):
        code = path.read_text()
        builtin = PythonASTParser().parse_file(path, code)
        tree = ASTParserFactory.get_parser("python", "tree_sitter").parse_file(path, code)
        sig = lambda a: ([(f.name, f.cyclomatic_complexity, bool(f.docstring)) for f in a.functions],
                         [(c.name, [(m.name, m.cyclomatic_complexity) for m in c.methods]) for c in a.classes])
        assert sig(tree) == sig(builtin), path.name


@pytest.mark.treesitter
@needs("typescript")
def test_enhanced_detector_scores_typescript_structurally(tmp_path):
    import yaml
    from ai_code_detector.aicd import _file_info
    from ai_code_detector.detector_enhanced import EnhancedAICodeDetector
    src = tmp_path / "service.ts"
    src.write_text((FIX / "service.ts").read_text())
    cfg_path = Path(__file__).resolve().parents[1] / "src/ai_code_detector/configs/default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    scores = {}
    for backend in ("python", "auto"):
        cfg["analysis"]["ast_backend"] = backend
        p = tmp_path / f"{backend}.yaml"
        p.write_text(yaml.safe_dump(cfg))
        det = EnhancedAICodeDetector(config_path=p, use_ml=False, use_explanations=False)
        scores[backend] = det._analyze_file_enhanced(_file_info(src), tmp_path)
    assert scores["python"].structural_score < scores["auto"].structural_score
    assert scores["auto"].features["structural"]["avg_cyclomatic_complexity"] > 0
