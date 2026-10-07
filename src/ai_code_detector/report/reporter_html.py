"""Self-contained HTML findings viewer (#10).

``aicd scan PATH --html out/report.html`` writes one file with no network
dependencies: inline CSS/JS and the scan data embedded as JSON. It has
three screens:

* repo summary: probability, confidence, component scores, languages,
  suppressed count
* file table: sortable, text filter, threshold slider (``?threshold=0.6``
  or ``#threshold=0.6`` in the URL presets it)
* file detail (click a row, or ``#file=path``): the top 5 features by
  their share of the file's AI probability (from
  ``FileScore.feature_contributions``), component scores, the flagged
  indicators and the natural-language explanation

Every number shown comes from the RepoScore. Nothing is computed or
invented in the browser, and an empty scan renders an empty state.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..model.aggregator import RepoScore

FEATURE_LABELS = {
    "high_comment_ratio": "High comment-to-code ratio",
    "boilerplate_comments": "Boilerplate comments",
    "tutorial_comments": "Tutorial-style comments",
    "generic_naming": "Generic identifier names",
    "low_identifier_entropy": "Low identifier entropy",
    "uniform_indentation": "Perfectly uniform indentation",
    "no_trailing_whitespace": "No trailing whitespace",
    "code_duplication": "Code duplication",
    "intra_file_similarity": "Intra-file similarity",
    "over_explained_functions": "Over-explained simple functions",
    "docstring_heavy_simple_code": "Docstring-heavy simple code",
    "generic_exceptions": "Generic except Exception",
    "print_on_error": "print() on error",
    "unused_functions": "Unused functions",
    "unused_imports": "Unused imports",
    "unreachable_code": "Unreachable code",
    "missing_cleanup": "Missing resource cleanup",
    "ml_classifier": "ML classifier (0.6 × p)",
}


def _verdict(p: float) -> str:
    if p >= 0.8:
        return "Very likely AI-generated"
    if p >= 0.6:
        return "Likely AI-generated"
    if p >= 0.4:
        return "Possibly AI-assisted"
    return "Likely human-written"


def build_data(repo_score: RepoScore, mode: str = "enhanced", threshold: float = 0.0) -> Dict[str, Any]:
    files = []
    for fs in sorted(repo_score.file_scores, key=lambda f: -float(f.ai_probability)):
        contrib = sorted(((k, float(v)) for k, v in (getattr(fs, "feature_contributions", None) or {}).items()
                          if v > 0), key=lambda kv: -kv[1])
        indicators = fs.feature_explanations if isinstance(fs.feature_explanations, dict) else {}
        files.append({
            "path": str(fs.file_path),
            "ai_probability": round(float(fs.ai_probability), 4),
            "stylometry": round(float(fs.stylometry_score), 4),
            "structural": round(float(fs.structural_score), 4),
            "parse_failed": bool(getattr(fs, "parse_failed", False)),
            "contributions": [{"feature": k, "label": FEATURE_LABELS.get(k, k), "share": round(v, 4)}
                              for k, v in contrib],
            "indicators": {k: (round(v, 3) if isinstance(v, (int, float)) else str(v))
                           for k, v in indicators.items()},
            "explanation": getattr(fs, "explanation", None) or "",
        })
    sup = getattr(repo_score, "suppressed", None) or {}
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "mode": mode,
        "default_threshold": threshold,
        "summary": {
            "repo_path": str(repo_score.repo_path),
            "ai_probability": round(float(repo_score.ai_probability), 4),
            "confidence": round(float(repo_score.confidence), 4),
            "verdict": _verdict(float(repo_score.ai_probability)) if files else "No files scanned",
            "stylometry": round(float(repo_score.stylometry_score), 4),
            "structural": round(float(repo_score.structural_score), 4),
            "history": round(float(repo_score.history_score), 4),
            "total_files": int(repo_score.total_files_analyzed),
            "total_lines": int(repo_score.total_lines_analyzed),
            "languages": dict(repo_score.language_distribution or {}),
            "suppressed": int(sup.get("count", 0) or 0),
        },
        "files": files,
    }


def _embed_json(data: Dict[str, Any]) -> str:
    # Safe inside <script type="application/json">: no `</script>`, no HTML parsing surprises.
    return (json.dumps(data, ensure_ascii=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


class HTMLReporter:
    def __init__(self, mode: str = "enhanced", threshold: float = 0.0):
        self.mode = mode
        self.threshold = threshold

    def render(self, repo_score: RepoScore) -> str:
        data = build_data(repo_score, self.mode, self.threshold)
        title = f"aicd report: {data['summary']['repo_path'] or 'scan'}"
        return (_TEMPLATE.replace("__TITLE__", html.escape(title))
                .replace("__DATA__", _embed_json(data)))

    def generate(self, repo_score: RepoScore, output_path: Optional[Path]) -> str:
        page = self.render(repo_score)
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(page, encoding="utf-8")
        return page


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f7f8;--card:#fff;--ink:#1d1d1f;--mute:#6b6b76;--line:#e3e3e8;--hi:#d93025;--mid:#e8a100;--lo:#188038;--bar:#4c6ef5}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:18px 24px;border-bottom:1px solid var(--line);background:var(--card)}h1{font-size:18px;margin:0}
.sub{color:var(--mute);font-size:12px}main{padding:18px 24px;display:grid;gap:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.card .k{color:var(--mute);font-size:12px}.card .v{font-size:20px;font-weight:600}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin-bottom:10px}
.controls label{display:flex;gap:8px;align-items:center}input[type=search]{padding:5px 8px;border:1px solid var(--line);border-radius:6px;min-width:220px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{cursor:pointer;user-select:none;font-weight:600;white-space:nowrap}th[aria-sort=ascending]::after{content:" ▲"}th[aria-sort=descending]::after{content:" ▼"}
tbody tr{cursor:pointer}tbody tr:hover{background:#f0f2ff}tbody tr.sel{background:#e5e9ff}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
.pill{display:inline-block;min-width:54px;text-align:right;font-weight:600}.hi{color:var(--hi)}.mid{color:var(--mid)}.lo{color:var(--lo)}
.bars{display:grid;gap:6px}.bar{display:grid;grid-template-columns:240px 1fr 64px;gap:8px;align-items:center}
.track{background:#eef0f6;border-radius:4px;height:12px;overflow:hidden}.fill{background:var(--bar);height:100%}
.empty{color:var(--mute);padding:24px;text-align:center}.note{color:var(--mute);font-size:12px}
pre.expl{white-space:pre-wrap;background:#fafafa;border:1px solid var(--line);border-radius:6px;padding:10px;margin:0}
.layout{display:grid;grid-template-columns:minmax(0,3fr) minmax(320px,2fr);gap:16px}@media(max-width:1000px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><h1 id="title">aicd report</h1><div class="sub" id="meta"></div></header>
<main>
  <section class="cards" id="summary" aria-label="Repository summary"></section>
  <div class="layout">
    <section class="panel" aria-label="Files">
      <div class="controls">
        <label>Threshold <input type="range" id="thr" min="0" max="100" step="1"> <span id="thrv" class="pill"></span></label>
        <label><input type="search" id="q" placeholder="Filter paths…"></label>
        <span class="note" id="count"></span>
      </div>
      <div id="tablewrap"></div>
    </section>
    <section class="panel" id="detail" aria-label="File detail"><div class="empty">Select a file to see its contributing features.</div></section>
  </div>
  <p class="note">Probabilistic signals, not proof of authorship. Feature shares come from the heuristic aggregator; in enhanced mode the heuristics are weighted 0.4 and the ML classifier 0.6.</p>
</main>
<script type="application/json" id="aicd-data">__DATA__</script>
<script>
(function(){
"use strict";
var D = JSON.parse(document.getElementById("aicd-data").textContent);
var S = D.summary, files = D.files;
function el(tag, attrs, kids){var e=document.createElement(tag);for(var k in (attrs||{})){if(k==="text")e.textContent=attrs[k];else if(k==="cls")e.className=attrs[k];else e.setAttribute(k,attrs[k]);}(kids||[]).forEach(function(c){e.appendChild(c)});return e;}
function pct(x){return (x*100).toFixed(1)+"%";}
function tone(x){return x>=0.6?"hi":x>=0.4?"mid":"lo";}
function params(){var out={};[location.search.slice(1),location.hash.slice(1)].forEach(function(s){s.split("&").forEach(function(kv){if(!kv)return;var i=kv.indexOf("=");var k=decodeURIComponent(i<0?kv:kv.slice(0,i));out[k]=i<0?"":decodeURIComponent(kv.slice(i+1));});});return out;}
var P = params();
var thr = isFinite(parseFloat(P.threshold)) ? Math.min(1,Math.max(0,parseFloat(P.threshold))) : D.default_threshold;
var sortKey = "ai_probability", sortDir = -1, selected = P.file || null;

document.getElementById("title").textContent = "aicd report: " + (S.repo_path || "scan");
document.getElementById("meta").textContent = "mode: " + D.mode + " · generated " + D.generated_at;

function card(k, v, cls){return el("div",{cls:"card"},[el("div",{cls:"k",text:k}),el("div",{cls:"v "+(cls||""),text:v})]);}
var sum = document.getElementById("summary");
if (!files.length) {
  sum.appendChild(card("Files scanned","0"));
  if (S.suppressed) sum.appendChild(card("Suppressed (.aicdignore)", String(S.suppressed)));
} else {
  sum.appendChild(card("Repository AI probability", pct(S.ai_probability), tone(S.ai_probability)));
  sum.appendChild(card("Verdict", S.verdict));
  sum.appendChild(card("Confidence", pct(S.confidence)));
  sum.appendChild(card("Stylometry / structural", pct(S.stylometry)+" / "+pct(S.structural)));
  sum.appendChild(card("History", pct(S.history)));
  sum.appendChild(card("Files / lines", S.total_files+" / "+S.total_lines));
  var langs = Object.keys(S.languages).map(function(k){return k+" "+S.languages[k];}).join(", ");
  sum.appendChild(card("Languages", langs || "n/a"));
  sum.appendChild(card("Suppressed (.aicdignore)", String(S.suppressed)));
}

var slider = document.getElementById("thr"), thrv = document.getElementById("thrv"), q = document.getElementById("q");
slider.value = Math.round(thr*100);
function syncHash(){var h=["threshold="+thr.toFixed(2)];if(selected)h.push("file="+encodeURIComponent(selected));history.replaceState(null,"","#"+h.join("&"));}
slider.addEventListener("input", function(){thr = slider.value/100; renderTable(); syncHash();});
q.addEventListener("input", renderTable);

var COLS = [["path","File"],["ai_probability","AI probability"],["stylometry","Stylometry"],["structural","Structural"]];
function renderTable(){
  thrv.textContent = pct(thr);
  var wrap = document.getElementById("tablewrap"); wrap.textContent = "";
  if (!files.length) {
    wrap.appendChild(el("div",{cls:"empty",text:"No files were scanned" + (S.suppressed ? " (" + S.suppressed + " suppressed by .aicdignore)." : ".")}));
    document.getElementById("count").textContent = "";
    return;
  }
  var needle = q.value.trim().toLowerCase();
  var rows = files.filter(function(f){return f.ai_probability >= thr && (!needle || f.path.toLowerCase().indexOf(needle) >= 0);});
  rows.sort(function(a,b){var x=a[sortKey], y=b[sortKey]; return (x<y?-1:x>y?1:0)*sortDir;});
  document.getElementById("count").textContent = rows.length + " of " + files.length + " files at or above " + pct(thr);
  var thead = el("thead",{},[el("tr",{},COLS.map(function(c){var th=el("th",{text:c[1],scope:"col"}); if(c[0]===sortKey) th.setAttribute("aria-sort", sortDir>0?"ascending":"descending");
    th.addEventListener("click",function(){ if(sortKey===c[0]) sortDir=-sortDir; else {sortKey=c[0]; sortDir=c[0]==="path"?1:-1;} renderTable();}); return th;}))]);
  var tbody = el("tbody");
  rows.forEach(function(f){
    var tr = el("tr",{"data-path":f.path},[el("td",{},[el("code",{text:f.path})]),
      el("td",{cls:"num"},[el("span",{cls:"pill "+tone(f.ai_probability),text:pct(f.ai_probability)})]),
      el("td",{cls:"num",text:pct(f.stylometry)}), el("td",{cls:"num",text:pct(f.structural)})]);
    if (f.path === selected) tr.className = "sel";
    tr.addEventListener("click", function(){ selected = f.path; renderTable(); renderDetail(); syncHash(); });
    tbody.appendChild(tr);
  });
  wrap.appendChild(rows.length ? el("table",{},[thead,tbody]) : el("div",{cls:"empty",text:"No files at or above " + pct(thr) + (needle ? " matching \u201c" + q.value + "\u201d" : "") + "."}));
}

function renderDetail(){
  var box = document.getElementById("detail"); box.textContent = "";
  var f = files.filter(function(x){return x.path === selected;})[0];
  if (!f) { box.appendChild(el("div",{cls:"empty",text: files.length ? "Select a file to see its contributing features." : "Nothing to show."})); return; }
  box.appendChild(el("h2",{style:"font-size:15px;margin:0 0 4px"},[el("code",{text:f.path})]));
  box.appendChild(el("div",{},[el("span",{text:"AI probability "}), el("span",{cls:"pill "+tone(f.ai_probability),text:pct(f.ai_probability)}),
    el("span",{cls:"note",text:"  stylometry "+pct(f.stylometry)+" · structural "+pct(f.structural)+(f.parse_failed?" · parse failed":"")})]));
  box.appendChild(el("h3",{style:"font-size:13px;margin:14px 0 6px",text:"Top contributing features"}));
  var top = f.contributions.slice(0,5);
  if (!top.length) box.appendChild(el("div",{cls:"note",text:"No feature contributed to this file's score."}));
  else {
    var max = top[0].share, bars = el("div",{cls:"bars"});
    top.forEach(function(c){ bars.appendChild(el("div",{cls:"bar",title:c.feature},[el("span",{text:c.label}),
      el("div",{cls:"track"},[el("div",{cls:"fill",style:"width:"+(max>0?(100*c.share/max).toFixed(1):0)+"%"})]),
      el("span",{cls:"num",text:"+"+pct(c.share)})])); });
    box.appendChild(bars);
    box.appendChild(el("div",{cls:"note",style:"margin-top:4px",text:"Each bar is the feature's share of the file's AI probability (shares sum to the probability)."}));
  }
  var ind = Object.keys(f.indicators);
  if (ind.length) {
    box.appendChild(el("h3",{style:"font-size:13px;margin:14px 0 6px",text:"Indicators over threshold"}));
    box.appendChild(el("ul",{},ind.map(function(k){return el("li",{},[el("code",{text:k}), el("span",{text:": "+f.indicators[k]})]);})));
  }
  if (f.explanation) {
    box.appendChild(el("h3",{style:"font-size:13px;margin:14px 0 6px",text:"Explanation"}));
    box.appendChild(el("pre",{cls:"expl",text:f.explanation}));
  }
}
renderTable(); renderDetail();
})();
</script>
</body>
</html>
"""
