#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "Runs"
DEFAULT_OUTPUT = ROOT / "Report" / "semantic_repair_queries.html"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except Exception:
        return default
    return text if text else default


def _safe_rel(path: Any) -> str:
    if not path:
        return ""
    p = Path(str(path))
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def _safe_uri(path: Any) -> str:
    if not path:
        return ""
    try:
        return Path(str(path)).expanduser().resolve().as_uri()
    except Exception:
        return ""


def _path_for_scanned_root(path: Any, runs_dir: Path) -> Any:
    if not path:
        return path
    p = Path(str(path)).expanduser()
    if p.exists():
        return p
    try:
        rel = p.resolve(strict=False).relative_to((ROOT / "Runs").resolve(strict=False))
    except Exception:
        return path
    candidate = runs_dir / rel
    return candidate if candidate.exists() else path


def _is_top_level_run_metadata(path: Path) -> bool:
    return not any(part.startswith("ciclo") for part in path.parts)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _semantic_failed_queries_from_cycle(cycle: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return (stage, query) pairs for semantic failures, avoiding cycle/stage duplicates."""
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    stages = cycle.get("stages") if isinstance(cycle.get("stages"), list) else []
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("failure_type") != "semantic":
            continue
        details = stage.get("failure_details") if isinstance(stage.get("failure_details"), dict) else {}
        failed_queries = details.get("failed_queries") if isinstance(details.get("failed_queries"), list) else []
        for query in failed_queries:
            if isinstance(query, dict):
                out.append((stage, query))

    if out:
        return out

    if cycle.get("failure_type") != "semantic":
        return []
    details = cycle.get("failure_details") if isinstance(cycle.get("failure_details"), dict) else {}
    failed_queries = details.get("failed_queries") if isinstance(details.get("failed_queries"), list) else []
    fallback_stage = {
        "stage": cycle.get("failed_stage") or "unknown",
        "failure_type": "semantic",
        "failure_reason": cycle.get("failure_reason") or "",
        "details": {},
        "artifacts": {},
    }
    for query in failed_queries:
        if isinstance(query, dict):
            out.append((fallback_stage, query))
    return out


def _collect_rows_from_dir(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(runs_dir.glob("**/run_metadata.json")):
        if not _is_top_level_run_metadata(meta_path):
            continue
        meta = _read_json(meta_path)
        if not isinstance(meta, dict):
            continue

        cycles = meta.get("cycles")
        if not isinstance(cycles, list):
            continue

        for cycle in cycles:
            if not isinstance(cycle, dict):
                continue
            cycle_no = cycle.get("cycle")
            for stage, query in _semantic_failed_queries_from_cycle(cycle):
                details = stage.get("details") if isinstance(stage.get("details"), dict) else {}
                artifacts = stage.get("artifacts") if isinstance(stage.get("artifacts"), dict) else {}
                raw_run_dir = _path_for_scanned_root(meta.get("run_dir") or meta_path.parent, runs_dir)
                raw_cycle_metadata = _path_for_scanned_root(cycle.get("metadata_path"), runs_dir)
                raw_query_path = _path_for_scanned_root(artifacts.get("query_path") or cycle.get("adapted_query_path"), runs_dir)
                raw_xml_path = _path_for_scanned_root(artifacts.get("xml_path") or cycle.get("compiled_xml_path"), runs_dir)
                rows.append(
                    {
                        "specification": _safe_str(meta.get("scenario"), "unknown"),
                        "source_root": _safe_rel(runs_dir),
                        "run_id": _safe_str(meta.get("run_id") or meta_path.parent.name),
                        "run_started_at": meta.get("run_started_at"),
                        "model": _safe_str(meta.get("generation_model") or meta.get("repair_model"), "unknown"),
                        "system_prompt": _safe_str(meta.get("system_prompt"), "unknown"),
                        "repair_prompt": _safe_str(meta.get("repair_prompt"), "unknown"),
                        "cycle": cycle_no,
                        "stage": _safe_str(stage.get("stage"), _safe_str(cycle.get("failed_stage"), "unknown")),
                        "failure_kind": _safe_str(query.get("failure_kind"), "unknown"),
                        "failure_reason": _safe_str(query.get("failure_reason") or stage.get("failure_reason")),
                        "query_index": query.get("index"),
                        "query_line": query.get("query_line"),
                        "description": _safe_str(query.get("description")),
                        "source_formula": _safe_str(query.get("source_formula")),
                        "adapted_formula": _safe_str(query.get("adapted_formula")),
                        "expected_probability": _to_float(query.get("expected_probability")),
                        "obtained_probability": _to_float(query.get("obtained_probability")),
                        "probability_interval_lower": _to_float(query.get("probability_interval_lower")),
                        "probability_interval_upper": _to_float(query.get("probability_interval_upper")),
                        "probability_delta": _to_float(query.get("probability_delta")),
                        "probability_threshold": _to_float(query.get("probability_threshold") or details.get("probability_threshold")),
                        "verifyta_output": _safe_str(query.get("verifyta_output")),
                        "run_dir": _safe_rel(raw_run_dir),
                        "metadata_path": _safe_rel(meta_path),
                        "metadata_uri": _safe_uri(meta_path),
                        "cycle_metadata_path": _safe_rel(raw_cycle_metadata),
                        "cycle_metadata_uri": _safe_uri(raw_cycle_metadata),
                        "query_path": _safe_rel(raw_query_path),
                        "query_uri": _safe_uri(raw_query_path),
                        "xml_path": _safe_rel(raw_xml_path),
                        "xml_uri": _safe_uri(raw_xml_path),
                    }
                )

    rows.sort(
        key=lambda row: (
            row["specification"],
            str(row.get("run_started_at") or ""),
            int(row.get("cycle") or 0),
            int(row.get("query_index") or 0),
        )
    )
    return rows


def _collect_rows(runs_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for runs_dir in runs_dirs:
        rows.extend(_collect_rows_from_dir(runs_dir))
    rows.sort(
        key=lambda row: (
            row["specification"],
            row["source_root"],
            str(row.get("run_started_at") or ""),
            int(row.get("cycle") or 0),
            int(row.get("query_index") or 0),
        )
    )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    specs = Counter(row["specification"] for row in rows)
    source_roots = Counter(row["source_root"] for row in rows)
    models = Counter(row["model"] for row in rows)
    runs = {row["run_id"] for row in rows}
    deltas = [row["probability_delta"] for row in rows if row.get("probability_delta") is not None]
    by_spec: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["specification"]].append(row)
    for spec, spec_rows in sorted(grouped.items()):
        spec_deltas = [row["probability_delta"] for row in spec_rows if row.get("probability_delta") is not None]
        by_spec[spec] = {
            "queries": len(spec_rows),
            "runs": len({row["run_id"] for row in spec_rows}),
            "max_delta": max(spec_deltas) if spec_deltas else None,
            "avg_delta": (sum(spec_deltas) / len(spec_deltas)) if spec_deltas else None,
        }
    return {
        "total_queries": len(rows),
        "total_runs": len(runs),
        "total_specifications": len(specs),
        "specification_counts": dict(specs),
        "source_root_counts": dict(source_roots),
        "model_counts": dict(models),
        "max_delta": max(deltas) if deltas else None,
        "avg_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "by_specification": by_spec,
    }


def _build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False).replace("</script", "<\\/script")
    html = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Semantic Repair Queries</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #5d6b82;
      --line: #d9e1ee;
      --accent: #0c65d8;
      --accent-soft: #e7f0ff;
      --bad: #b42318;
      --warn: #a15c07;
      --good: #16794c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", Arial, sans-serif;
    }
    .wrap {
      max-width: 1500px;
      margin: 0 auto;
      padding: 22px;
      display: grid;
      gap: 14px;
    }
    .hero {
      border-bottom: 1px solid var(--line);
      padding: 4px 0 14px;
    }
    h1 { margin: 0 0 6px; font-size: 28px; }
    .hero p { margin: 0; color: var(--muted); }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }
    .card,
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .card { padding: 12px 14px; }
    .card .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }
    .card .value {
      margin-top: 3px;
      font-size: 24px;
      font-weight: 750;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(220px, 1.4fr) repeat(4, minmax(150px, 1fr));
      gap: 10px;
    }
    @media (max-width: 980px) {
      .controls { grid-template-columns: 1fr; }
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    input,
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      color: var(--ink);
      background: #fff;
      font-size: 14px;
    }
    .spec {
      overflow: hidden;
    }
    .spec-head {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 12px 14px;
      background: #f9fbfe;
      border-bottom: 1px solid var(--line);
    }
    .spec h2 { margin: 0; font-size: 17px; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      background: #fff;
      color: #344259;
      font-size: 12px;
      white-space: nowrap;
    }
    .table-wrap { overflow: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      min-width: 1180px;
    }
    th {
      text-align: left;
      position: sticky;
      top: 0;
      background: #f9fbfe;
      border-bottom: 1px solid var(--line);
      color: #344259;
      padding: 8px;
      white-space: nowrap;
    }
    td {
      vertical-align: top;
      border-bottom: 1px solid #edf1f7;
      padding: 8px;
    }
    .mono {
      font-family: "Cascadia Mono", "Consolas", "Courier New", monospace;
      font-size: 12px;
    }
    .formula {
      max-width: 520px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .desc { max-width: 360px; }
    .delta {
      font-weight: 750;
      color: var(--bad);
    }
    .muted { color: var(--muted); }
    details {
      max-width: 680px;
    }
    summary {
      cursor: pointer;
      color: var(--accent);
      font-weight: 650;
    }
    pre {
      margin: 8px 0 0;
      background: #101828;
      color: #e6edf7;
      border-radius: 8px;
      padding: 10px;
      white-space: pre-wrap;
      max-height: 240px;
      overflow: auto;
    }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .empty {
      padding: 20px;
      color: var(--muted);
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Query che hanno causato repair semantic</h1>
      <p>Raggruppate per specifica, con delta di probabilita', valori attesi/ottenuti e formula sorgente/adattata.</p>
    </section>
    <section class="cards" id="cards"></section>
    <section class="controls panel" style="padding: 12px;">
      <div><label>Cerca</label><input id="search" placeholder="descrizione, formula, run, modello" /></div>
      <div><label>Cartella</label><select id="sourceFilter"></select></div>
      <div><label>Specifica</label><select id="specFilter"></select></div>
      <div><label>Modello</label><select id="modelFilter"></select></div>
      <div><label>Tipo fallimento</label><select id="kindFilter"></select></div>
    </section>
    <main id="content"></main>
  </div>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent || '{}');
    const records = Array.isArray(payload.records) ? payload.records : [];
    const summary = payload.summary || {};
    const el = {
      cards: document.getElementById('cards'),
      content: document.getElementById('content'),
      search: document.getElementById('search'),
      source: document.getElementById('sourceFilter'),
      spec: document.getElementById('specFilter'),
      model: document.getElementById('modelFilter'),
      kind: document.getElementById('kindFilter'),
    };

    function esc(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }
    function fmtNum(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: 6 });
    }
    function fmtDelta(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
      return Number(value).toFixed(6);
    }
    function uniq(values) {
      return [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b)));
    }
    function fillSelect(select, values, label) {
      select.innerHTML = '<option value="">' + esc(label) + '</option>' +
        values.map(v => '<option value="' + esc(v) + '">' + esc(v) + '</option>').join('');
    }
    function card(label, value) {
      return '<article class="card"><div class="label">' + esc(label) + '</div><div class="value">' + esc(value) + '</div></article>';
    }
    function renderCards(rows) {
      const specs = new Set(rows.map(r => r.specification));
      const sources = new Set(rows.map(r => r.source_root));
      const runs = new Set(rows.map(r => r.run_id));
      const deltas = rows.map(r => Number(r.probability_delta)).filter(Number.isFinite);
      const avg = deltas.length ? deltas.reduce((a, b) => a + b, 0) / deltas.length : null;
      const max = deltas.length ? Math.max(...deltas) : null;
      el.cards.innerHTML = [
        card('Query semantic', rows.length),
        card('Specifiche', specs.size),
        card('Cartelle', sources.size),
        card('Run coinvolte', runs.size),
        card('Delta medio', fmtDelta(avg)),
        card('Delta massimo', fmtDelta(max)),
      ].join('');
    }
    function filteredRows() {
      const q = el.search.value.trim().toLowerCase();
      return records.filter(r => {
        if (el.source.value && r.source_root !== el.source.value) return false;
        if (el.spec.value && r.specification !== el.spec.value) return false;
        if (el.model.value && r.model !== el.model.value) return false;
        if (el.kind.value && r.failure_kind !== el.kind.value) return false;
        if (!q) return true;
        return [
          r.source_root,
          r.specification,
          r.run_id,
          r.model,
          r.system_prompt,
          r.description,
          r.source_formula,
          r.adapted_formula,
          r.failure_reason,
        ].join(' ').toLowerCase().includes(q);
      });
    }
    function rowHtml(r) {
      const metaLink = r.metadata_uri ? '<a href="' + esc(r.metadata_uri) + '">' + esc(r.run_id) + '</a>' : esc(r.run_id);
      return '<tr>' +
        '<td class="mono">' + metaLink + '<br><span class="muted">' + esc(r.source_root || '-') + ', ciclo ' + esc(r.cycle ?? '-') + '</span></td>' +
        '<td>' + esc(r.model) + '<br><span class="muted">' + esc(r.system_prompt) + '</span></td>' +
        '<td class="mono">' + esc(r.query_index ?? '-') + '<br><span class="muted">linea ' + esc(r.query_line ?? '-') + '</span></td>' +
        '<td class="desc">' + esc(r.description || '-') + '</td>' +
        '<td class="mono">' + esc(r.failure_kind || '-') + '</td>' +
        '<td>' + fmtNum(r.expected_probability) + '</td>' +
        '<td>' + fmtNum(r.obtained_probability) + '</td>' +
        '<td>' + fmtNum(r.probability_interval_lower) + ' - ' + fmtNum(r.probability_interval_upper) + '</td>' +
        '<td class="delta">' + fmtDelta(r.probability_delta) + '</td>' +
        '<td>' + fmtNum(r.probability_threshold) + '</td>' +
        '<td class="formula mono">' + esc(r.adapted_formula || '-') +
          (r.source_formula && r.source_formula !== r.adapted_formula
            ? '<details><summary>Formula sorgente</summary><pre>' + esc(r.source_formula) + '</pre></details>'
            : '') +
          (r.verifyta_output ? '<details><summary>Verifyta</summary><pre>' + esc(r.verifyta_output) + '</pre></details>' : '') +
        '</td>' +
      '</tr>';
    }
    function render() {
      const rows = filteredRows();
      renderCards(rows);
      if (!rows.length) {
        el.content.innerHTML = '<section class="panel empty">Nessuna query semantic trovata con i filtri correnti.</section>';
        return;
      }
      const bySpec = new Map();
      for (const row of rows) {
        if (!bySpec.has(row.specification)) bySpec.set(row.specification, []);
        bySpec.get(row.specification).push(row);
      }
      const sections = [...bySpec.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([spec, specRows]) => {
        const deltas = specRows.map(r => Number(r.probability_delta)).filter(Number.isFinite);
        const avg = deltas.length ? deltas.reduce((a, b) => a + b, 0) / deltas.length : null;
        const max = deltas.length ? Math.max(...deltas) : null;
        return '<section class="panel spec">' +
          '<div class="spec-head">' +
            '<h2>' + esc(spec) + '</h2>' +
            '<div class="chips">' +
              '<span class="chip">' + specRows.length + ' query</span>' +
              '<span class="chip">' + new Set(specRows.map(r => r.run_id)).size + ' run</span>' +
              '<span class="chip">' + [...new Set(specRows.map(r => r.source_root))].join(', ') + '</span>' +
              '<span class="chip">delta medio ' + fmtDelta(avg) + '</span>' +
              '<span class="chip">delta max ' + fmtDelta(max) + '</span>' +
            '</div>' +
          '</div>' +
          '<div class="table-wrap"><table><thead><tr>' +
            '<th>Run</th><th>Modello</th><th>Query</th><th>Descrizione</th><th>Tipo</th>' +
            '<th>Expected</th><th>Obtained</th><th>Intervallo</th><th>Delta</th><th>Soglia</th><th>Formula adattata</th>' +
          '</tr></thead><tbody>' +
            specRows.map(rowHtml).join('') +
          '</tbody></table></div>' +
        '</section>';
      });
      el.content.innerHTML = sections.join('');
    }
    fillSelect(el.source, uniq(records.map(r => r.source_root)), 'Tutte');
    fillSelect(el.spec, uniq(records.map(r => r.specification)), 'Tutte');
    fillSelect(el.model, uniq(records.map(r => r.model)), 'Tutti');
    fillSelect(el.kind, uniq(records.map(r => r.failure_kind)), 'Tutti');
    [el.search, el.source, el.spec, el.model, el.kind].forEach(node => {
      node.addEventListener('input', render);
      node.addEventListener('change', render);
    });
    renderCards(records);
    render();
  </script>
</body>
</html>
"""
    return html.replace("__PAYLOAD__", data_json)


def build_page(runs_dirs: list[Path], output_html: Path, print_summary: bool = False) -> None:
    records = _collect_rows(runs_dirs)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs_dirs": [_safe_rel(runs_dir) for runs_dir in runs_dirs],
        "summary": _summary(records),
        "records": records,
    }
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_build_html(payload), encoding="utf-8")
    if print_summary:
        print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"[OK] Semantic repair query page written: {output_html}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML page listing queries that triggered semantic repair, grouped by specification."
    )
    parser.add_argument(
        "--runs-dir",
        nargs="+",
        default=[str(DEFAULT_RUNS_DIR)],
        help="One or more run directories to scan",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output HTML path")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary to stdout")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs_dirs = []
    for raw_runs_dir in args.runs_dir:
        runs_dir = Path(raw_runs_dir).expanduser()
        if not runs_dir.is_absolute():
            runs_dir = ROOT / runs_dir
        runs_dirs.append(runs_dir)
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    for runs_dir in runs_dirs:
        if not runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    build_page(runs_dirs, output, print_summary=args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
