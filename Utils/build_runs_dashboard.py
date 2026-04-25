#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "Runs"
DEFAULT_OUTPUT = ROOT / "Report" / "runs_dashboard.html"


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    raw = ts.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _duration_seconds(start_ts: Any, end_ts: Any) -> float | None:
    start = _parse_iso(start_ts)
    end = _parse_iso(end_ts)
    if not start or not end:
        return None
    return max((end - start).total_seconds(), 0.0)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_rel(path: Any) -> str:
    if not isinstance(path, str) or not path:
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def _safe_uri(path: Any) -> str:
    if not isinstance(path, str) or not path:
        return ""
    try:
        return Path(path).expanduser().resolve().as_uri()
    except Exception:
        return ""


def _safe_read_text(path: Any, max_chars: int | None = 120000) -> str:
    if not isinstance(path, str) or not path:
        return ""
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars] + "\n\n[truncated in dashboard view]"
        return text
    except Exception:
        return ""


def _pick_artifacts(meta: dict[str, Any]) -> dict[str, Any]:
    iterations = meta.get("iterations")
    if not isinstance(iterations, list):
        iterations = []

    chosen: dict[str, Any] | None = None
    for it in reversed(iterations):
        if not isinstance(it, dict):
            continue
        has_compiler = isinstance(it.get("compiler_output_path"), str) and bool(it.get("compiler_output_path"))
        is_valid = it.get("is_valid")
        if has_compiler and is_valid is False:
            chosen = it
            break

    if chosen is None:
        for it in reversed(iterations):
            if not isinstance(it, dict):
                continue
            if isinstance(it.get("compiler_output_path"), str) and it.get("compiler_output_path"):
                chosen = it
                break

    if chosen is None and iterations:
        for it in reversed(iterations):
            if isinstance(it, dict):
                chosen = it
                break

    if not isinstance(chosen, dict):
        return {
            "selected_iteration": None,
            "selected_is_valid": None,
            "selected_dsl_path": "",
            "selected_compiler_output_path": "",
        }

    dsl_path = ""
    if isinstance(chosen.get("dsl_path"), str) and chosen.get("dsl_path"):
        dsl_path = chosen["dsl_path"]
    elif isinstance(chosen.get("success_artifact_path"), str) and chosen.get("success_artifact_path"):
        dsl_path = chosen["success_artifact_path"]

    compiler_output = chosen.get("compiler_output_path")
    if not isinstance(compiler_output, str):
        compiler_output = ""

    return {
        "selected_iteration": chosen.get("iteration"),
        "selected_is_valid": chosen.get("is_valid"),
        "selected_dsl_path": dsl_path,
        "selected_compiler_output_path": compiler_output,
    }


def _iteration_artifacts(
  meta: dict[str, Any],
  embed_all_data: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    iterations = meta.get("iterations")
    if not isinstance(iterations, list):
        return [], None

    options: list[dict[str, Any]] = []
    max_chars = None if embed_all_data else 120000
    for it in iterations:
        if not isinstance(it, dict):
            continue

        dsl_path = ""
        if isinstance(it.get("dsl_path"), str) and it.get("dsl_path"):
            dsl_path = it["dsl_path"]
        elif isinstance(it.get("success_artifact_path"), str) and it.get("success_artifact_path"):
            dsl_path = it["success_artifact_path"]

        compiler_output = it.get("compiler_output_path")
        if not isinstance(compiler_output, str):
            compiler_output = ""

        options.append(
            {
                "iteration": it.get("iteration"),
                "is_valid": it.get("is_valid"),
                "dsl_path": _safe_rel(dsl_path),
                "dsl_uri": _safe_uri(dsl_path),
                "dsl_text": _safe_read_text(dsl_path, max_chars=max_chars),
                "compiler_output_path": _safe_rel(compiler_output),
                "compiler_output_uri": _safe_uri(compiler_output),
                "compiler_output_text": _safe_read_text(compiler_output, max_chars=max_chars),
            }
        )

    def _iter_sort_key(v: dict[str, Any]) -> tuple[int, int]:
        raw = v.get("iteration")
        if isinstance(raw, int):
            return (0, raw)
        try:
            return (0, int(raw))
        except Exception:
            return (1, 0)

    options.sort(key=_iter_sort_key)

    working: dict[str, Any] | None = None
    for opt in options:
        if opt.get("is_valid") is True:
            working = opt
            break

    return options, working


def _build_record(meta: dict[str, Any], meta_path: Path, embed_all_data: bool = False) -> dict[str, Any]:
    summary = meta.get("summary") if isinstance(meta.get("summary"), dict) else {}
    telemetry = meta.get("telemetry") if isinstance(meta.get("telemetry"), dict) else {}
    breaking_error = meta.get("breaking_error") if isinstance(meta.get("breaking_error"), dict) else {}

    run_dir = meta.get("run_dir")
    if not run_dir:
        run_dir = str(meta_path.parent)

    error_message = ""
    if isinstance(breaking_error.get("message"), str):
        error_message = breaking_error["message"]

    artifacts = _pick_artifacts(meta)
    iteration_options, working = _iteration_artifacts(meta, embed_all_data=embed_all_data)
    max_chars = None if embed_all_data else 120000
    selected_dsl_path = artifacts.get("selected_dsl_path") if isinstance(artifacts.get("selected_dsl_path"), str) else ""
    selected_compiler_output_path = (
      artifacts.get("selected_compiler_output_path")
      if isinstance(artifacts.get("selected_compiler_output_path"), str)
      else ""
    )

    record = {
        "run_id": str(meta.get("run_id") or meta_path.parent.name),
        "provider": str(meta.get("provider") or "unknown"),
        "generation_model": str(meta.get("generation_model") or "unknown"),
        "repair_model": str(meta.get("repair_model") or "unknown"),
        "scenario": str(meta.get("scenario") or "unknown"),
        "system_prompt": str(meta.get("system_prompt") or "unknown"),
        "shots": meta.get("shots"),
        "status": str(meta.get("status") or "unknown"),
        "run_started_at": meta.get("run_started_at"),
        "run_finished_at": meta.get("run_finished_at"),
        "duration_seconds": _duration_seconds(meta.get("run_started_at"), meta.get("run_finished_at")),
        "iterations_recorded": summary.get("iterations_recorded"),
        "first_success_iteration": summary.get("first_success_iteration"),
        "compiler_failures": summary.get("compiler_failures"),
        "compiler_successes": summary.get("compiler_successes"),
        "llm_calls": summary.get("llm_calls", telemetry.get("llm_calls")),
        "prompt_tokens_est_total": summary.get("prompt_tokens_est_total", telemetry.get("prompt_tokens_est_total")),
        "response_tokens_est_total": summary.get("response_tokens_est_total", telemetry.get("response_tokens_est_total")),
        "error_type": str(breaking_error.get("type") or ""),
        "error_message": error_message,
        "run_dir": _safe_rel(run_dir),
        "metadata_path": _safe_rel(str(meta_path)),
        "metadata_uri": _safe_uri(str(meta_path)),
        "metadata_text": _safe_read_text(str(meta_path), max_chars=max_chars),
        "selected_iteration": artifacts.get("selected_iteration"),
        "selected_is_valid": artifacts.get("selected_is_valid"),
        "selected_dsl_path": _safe_rel(selected_dsl_path),
        "selected_compiler_output_path": _safe_rel(selected_compiler_output_path),
        "selected_dsl_uri": _safe_uri(selected_dsl_path),
        "selected_compiler_output_uri": _safe_uri(selected_compiler_output_path),
        "selected_dsl_text": _safe_read_text(selected_dsl_path, max_chars=max_chars),
        "selected_compiler_output_text": _safe_read_text(selected_compiler_output_path, max_chars=max_chars),
        "iteration_options": iteration_options,
        "working_iteration": working.get("iteration") if isinstance(working, dict) else None,
        "working_dsl_path": working.get("dsl_path") if isinstance(working, dict) else "",
        "working_dsl_uri": working.get("dsl_uri") if isinstance(working, dict) else "",
        "working_compiler_output_path": working.get("compiler_output_path") if isinstance(working, dict) else "",
        "working_compiler_output_uri": working.get("compiler_output_uri") if isinstance(working, dict) else "",
    }
    return record


def _collect_records(runs_dir: Path) -> list[dict[str, Any]]:
    return _collect_records_with_options(runs_dir, embed_all_data=False)


def _collect_records_with_options(runs_dir: Path, embed_all_data: bool = False) -> list[dict[str, Any]]:
  records: list[dict[str, Any]] = []
  for meta_path in sorted(runs_dir.glob("**/run_metadata.json")):
    meta = _read_json(meta_path)
    if not isinstance(meta, dict):
      continue
    records.append(_build_record(meta, meta_path, embed_all_data=embed_all_data))
  return records


def _to_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(r.get("status", "unknown") for r in records)
    providers = Counter(r.get("provider", "unknown") for r in records)
    models = Counter(r.get("generation_model", "unknown") for r in records)
    scenarios = Counter(r.get("scenario", "unknown") for r in records)

    total_duration = sum(_to_number(r.get("duration_seconds"), 0.0) for r in records)
    total_calls = sum(_to_number(r.get("llm_calls"), 0.0) for r in records)
    total_prompt_tokens = sum(_to_number(r.get("prompt_tokens_est_total"), 0.0) for r in records)
    total_response_tokens = sum(_to_number(r.get("response_tokens_est_total"), 0.0) for r in records)

    latest_ts: datetime | None = None
    for r in records:
        ts = _parse_iso(r.get("run_started_at"))
        if ts and (latest_ts is None or ts > latest_ts):
            latest_ts = ts

    return {
        "total_runs": len(records),
        "status_counts": dict(statuses),
        "provider_counts": dict(providers),
        "model_counts": dict(models),
        "scenario_counts": dict(scenarios),
        "total_duration_seconds": round(total_duration, 2),
        "total_llm_calls": int(total_calls),
        "total_prompt_tokens_est": int(total_prompt_tokens),
        "total_response_tokens_est": int(total_response_tokens),
        "latest_run_started_at": latest_ts.isoformat() if latest_ts else None,
    }


def _build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    safe_data_json = data_json.replace("</script", "<\\/script")
    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>LIRAS Runs Dashboard</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #19202b;
      --muted: #5a667a;
      --line: #d8e0ea;
      --accent: #0b6cff;
      --accent-soft: #dce9ff;
      --good: #1d8348;
      --warn: #b9770e;
      --bad: #b03a2e;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 80% -20%, #dbeafe 0, #dbeafe 22%, transparent 45%),
        radial-gradient(circle at -10% 130%, #fde68a 0, #fde68a 25%, transparent 50%),
        var(--bg);
      min-height: 100vh;
    }

    .wrap {
      max-width: 1600px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      gap: 16px;
    }

    .hero {
      background: linear-gradient(100deg, #0b6cff 0%, #1348b8 60%, #1c2f79 100%);
      color: #fff;
      padding: 20px;
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(16, 36, 94, 0.25);
      animation: fadeUp 350ms ease-out;
    }

    .hero h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0.3px; }
    .hero p { margin: 0; opacity: 0.94; }

    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      animation: fadeUp 450ms ease-out;
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
    }

    .card .k { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
    .card .v { font-size: 24px; font-weight: 700; margin-top: 4px; }

    .controls {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      animation: fadeUp 550ms ease-out;
    }

    .control label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; }
    .control input,
    .control select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
      color: var(--ink);
      background: #fff;
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }

    @media (max-width: 1150px) {
      .grid { grid-template-columns: 1fr; }
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      animation: fadeUp 650ms ease-out;
    }

    .panel h2 {
      margin: 0;
      font-size: 15px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #f8fafc;
    }

    .table-wrap { max-height: 68vh; overflow: auto; }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    thead th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafc;
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 8px;
      white-space: nowrap;
    }

    tbody td {
      border-bottom: 1px solid #edf2f7;
      padding: 7px 8px;
      vertical-align: top;
    }

    tbody tr { cursor: pointer; transition: background 0.15s ease; }
    tbody tr:hover { background: #f3f8ff; }
    tbody tr.active { background: var(--accent-soft); }

    .status {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 12px;
      border: 1px solid transparent;
    }

    .status.success { color: var(--good); background: #eafaf1; border-color: #c7eed8; }
    .status.max_iterations_reached,
    .status.success_no_output,
    .status.generation_only { color: var(--warn); background: #fef5e7; border-color: #f8ddb0; }
    .status.crashed,
    .status.failed,
    .status.unknown { color: var(--bad); background: #fdecea; border-color: #f7c7c3; }

    .mono { font-family: "Cascadia Mono", "Consolas", "Courier New", monospace; font-size: 12px; }
    .muted { color: var(--muted); }
    .clamp { max-width: 520px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    .detail { padding: 12px; }
    .detail-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 10px;
    }
    .detail-actions a {
      display: inline-block;
      text-decoration: none;
      color: #0b3c97;
      background: #e9f0ff;
      border: 1px solid #c9dafc;
      border-radius: 8px;
      padding: 6px 9px;
      font-size: 12px;
      font-weight: 600;
    }
    .detail-actions select {
      border: 1px solid #c9dafc;
      border-radius: 8px;
      padding: 6px 8px;
      font-size: 12px;
      color: #173b7a;
      background: #f7faff;
    }
    .detail-actions a:hover { background: #dce8ff; }
    .detail pre {
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 8px;
      padding: 10px;
      font-size: 12px;
      line-height: 1.35;
      overflow: auto;
      max-height: 58vh;
    }
    .detail pre .section-title {
      color: #93c5fd;
      font-weight: 700;
    }
    .detail pre .diff-line {
      display: block;
      white-space: pre;
    }
    .detail pre .diff-line.ctx {
      color: #cbd5e1;
    }
    .detail pre .diff-line.add {
      background: rgba(34, 197, 94, 0.18);
      color: #dcfce7;
    }
    .detail pre .diff-line.del {
      background: rgba(239, 68, 68, 0.2);
      color: #fee2e2;
    }
    .detail pre .diff-gutter {
      display: inline-block;
      width: 5ch;
      color: #94a3b8;
      text-align: right;
      margin-right: 1ch;
    }
    .detail pre .diff-sign {
      display: inline-block;
      width: 2ch;
      color: #94a3b8;
    }

    .chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 12px; border-bottom: 1px solid var(--line); }
    .chip {
      background: #f4f7fb;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: #304057;
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <h1>LIRAS Runs Dashboard</h1>
      <p>Filter by provider, model, scenario, and status. Click a row to inspect complete run details.</p>
    </section>

    <section class=\"cards\" id=\"kpiCards\"></section>

    <section class=\"controls\">
      <div class=\"control\"><label>Search</label><input id=\"search\" placeholder=\"run id, error message, model, scenario\" /></div>
      <div class=\"control\"><label>Provider</label><select id=\"providerFilter\"></select></div>
      <div class=\"control\"><label>Model</label><select id=\"modelFilter\"></select></div>
      <div class=\"control\"><label>Scenario</label><select id=\"scenarioFilter\"></select></div>
      <div class=\"control\"><label>Status</label><select id=\"statusFilter\"></select></div>
      <div class=\"control\"><label>Sort</label>
        <select id=\"sortBy\">
          <option value=\"started_desc\">Newest first</option>
          <option value=\"started_asc\">Oldest first</option>
          <option value=\"duration_desc\">Duration desc</option>
          <option value=\"duration_asc\">Duration asc</option>
          <option value=\"tokens_desc\">Prompt tokens desc</option>
        </select>
      </div>
    </section>

    <section class=\"grid\">
      <div class=\"panel\">
        <h2>Runs <span id=\"rowCount\" class=\"muted\"></span></h2>
        <div class=\"table-wrap\">
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Started</th>
                <th>Provider</th>
                <th>Model</th>
                <th>Scenario</th>
                <th>Status</th>
                <th>Iter</th>
                <th>LLM</th>
                <th>PromptTok</th>
                <th>Error</th>
              </tr>
            </thead>
            <tbody id=\"rows\"></tbody>
          </table>
        </div>
      </div>

      <div class=\"panel\">
        <h2>Selected Run</h2>
        <div class=\"chips\" id=\"detailChips\"></div>
        <div class=\"detail\">
          <div id=\"detailActions\" class=\"detail-actions\"></div>
          <pre id=\"detail\">Select one row.</pre>
        </div>
      </div>
    </section>
  </div>

  <script id=\"dashboard-data\" type=\"application/json\">__DASHBOARD_DATA__</script>
  <script>
    let payload = { records: [], summary: {}, vscode_open_mode: false };
    let bootstrapError = '';

    try {
      const dataNode = document.getElementById('dashboard-data');
      if (!dataNode) {
        throw new Error('Missing #dashboard-data node');
      }
      payload = JSON.parse(dataNode.textContent || '{}');
    } catch (err) {
      bootstrapError = String(err && err.message ? err.message : err);
      console.error('Dashboard bootstrap error:', err);
    }

    const records = Array.isArray(payload.records) ? payload.records : [];
    const summary = payload.summary && typeof payload.summary === 'object' ? payload.summary : {};
    const vscodeMode = Boolean(payload.vscode_open_mode);
    const artifactsDetailMode = Boolean(payload.artifacts_detail_mode);
    const lirasDiffMode = Boolean(payload.liras_diff_mode);

    const el = {
      cards: document.getElementById('kpiCards'),
      search: document.getElementById('search'),
      provider: document.getElementById('providerFilter'),
      model: document.getElementById('modelFilter'),
      scenario: document.getElementById('scenarioFilter'),
      status: document.getElementById('statusFilter'),
      sortBy: document.getElementById('sortBy'),
      rows: document.getElementById('rows'),
      detail: document.getElementById('detail'),
      detailChips: document.getElementById('detailChips'),
      detailActions: document.getElementById('detailActions'),
      rowCount: document.getElementById('rowCount'),
    };

    let filtered = [...records];
    let selectedRunId = null;

    function uniqSorted(list) {
      return [...new Set(list.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b)));
    }

    function fmtNum(v) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return '-';
      return Number(v).toLocaleString();
    }

    function fmtDate(v) {
      if (!v) return '-';
      try { return new Date(v).toLocaleString(); } catch { return String(v); }
    }

    function fmtDuration(v) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return '-';
      const sec = Number(v);
      if (sec < 60) return sec.toFixed(1) + 's';
      const m = Math.floor(sec / 60);
      const s = Math.round(sec % 60);
      return m + 'm ' + s + 's';
    }

    function withLineNumbers(text) {
      if (!text) return '';
      const lines = String(text).split('\\n');
      const pad = String(lines.length).length;
      return lines
        .map((line, idx) => String(idx + 1).padStart(pad, ' ') + ' | ' + line)
        .join('\\n');
    }

    function escapeHtml(text) {
      return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function buildLirasDiffHtml(previousText, currentText) {
      const prev = String(previousText || '').split('\\n');
      const curr = String(currentText || '').split('\\n');
      const n = prev.length;
      const m = curr.length;
      const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));

      for (let i = n - 1; i >= 0; i -= 1) {
        for (let j = m - 1; j >= 0; j -= 1) {
          if (prev[i] === curr[j]) {
            dp[i][j] = dp[i + 1][j + 1] + 1;
          } else {
            dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
          }
        }
      }

      const rows = [];
      let i = 0;
      let j = 0;
      let hasChanges = false;

      function lineHtml(kind, oldLn, newLn, sign, text) {
        const oldLabel = oldLn ? String(oldLn) : '';
        const newLabel = newLn ? String(newLn) : '';
        return '<span class="diff-line ' + kind + '">' +
          '<span class="diff-gutter">' + oldLabel + '</span>' +
          '<span class="diff-gutter">' + newLabel + '</span>' +
          '<span class="diff-sign">' + sign + '</span>' +
          escapeHtml(text) +
        '</span>';
      }

      while (i < n && j < m) {
        if (prev[i] === curr[j]) {
          rows.push(lineHtml('ctx', i + 1, j + 1, ' ', prev[i]));
          i += 1;
          j += 1;
        } else if (dp[i + 1][j] >= dp[i][j + 1]) {
          hasChanges = true;
          rows.push(lineHtml('del', i + 1, '', '-', prev[i]));
          i += 1;
        } else {
          hasChanges = true;
          rows.push(lineHtml('add', '', j + 1, '+', curr[j]));
          j += 1;
        }
      }

      while (i < n) {
        hasChanges = true;
        rows.push(lineHtml('del', i + 1, '', '-', prev[i]));
        i += 1;
      }

      while (j < m) {
        hasChanges = true;
        rows.push(lineHtml('add', '', j + 1, '+', curr[j]));
        j += 1;
      }

      if (!hasChanges) {
        return '<span class="diff-line ctx">[no differences]</span>';
      }

      return rows.join('\\n');
    }

    function toVsCodeUri(uri) {
      if (!uri || typeof uri !== 'string') return '';
      if (uri.startsWith('file://')) {
        return 'vscode://file' + uri.slice('file://'.length);
      }
      return uri;
    }

    function adaptOpenUri(uri) {
      if (!vscodeMode) return uri;
      return toVsCodeUri(uri);
    }

    function statusClass(status) {
      return 'status ' + String(status || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
    }

    function card(label, value) {
      const d = document.createElement('div');
      d.className = 'card';
      d.innerHTML = '<div class="k">' + label + '</div><div class="v">' + value + '</div>';
      return d;
    }

    function renderCards() {
      const status = summary.status_counts || {};
      el.cards.innerHTML = '';
      el.cards.appendChild(card('Total Runs', fmtNum(summary.total_runs || records.length)));
      el.cards.appendChild(card('Success', fmtNum(status.success || 0)));
      el.cards.appendChild(card('Crashed', fmtNum(status.crashed || 0)));
      el.cards.appendChild(card('LLM Calls', fmtNum(summary.total_llm_calls || 0)));
      el.cards.appendChild(card('Prompt Tokens (est)', fmtNum(summary.total_prompt_tokens_est || 0)));
      el.cards.appendChild(card('Latest Start', summary.latest_run_started_at ? fmtDate(summary.latest_run_started_at) : '-'));
    }

    function fillSelect(selectEl, values, defaultLabel) {
      selectEl.innerHTML = '';
      const all = document.createElement('option');
      all.value = '';
      all.textContent = defaultLabel;
      selectEl.appendChild(all);
      for (const v of values) {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        selectEl.appendChild(opt);
      }
    }

    function setupFilters() {
      fillSelect(el.provider, uniqSorted(records.map(r => r.provider)), 'All providers');
      fillSelect(el.model, uniqSorted(records.map(r => r.generation_model)), 'All models');
      fillSelect(el.scenario, uniqSorted(records.map(r => r.scenario)), 'All scenarios');
      fillSelect(el.status, uniqSorted(records.map(r => r.status)), 'All statuses');
    }

    function sortRows(items) {
      const mode = el.sortBy.value;
      const arr = [...items];
      arr.sort((a, b) => {
        if (mode === 'started_asc') return String(a.run_started_at || '').localeCompare(String(b.run_started_at || ''));
        if (mode === 'started_desc') return String(b.run_started_at || '').localeCompare(String(a.run_started_at || ''));
        if (mode === 'duration_asc') return Number(a.duration_seconds || 0) - Number(b.duration_seconds || 0);
        if (mode === 'duration_desc') return Number(b.duration_seconds || 0) - Number(a.duration_seconds || 0);
        if (mode === 'tokens_desc') return Number(b.prompt_tokens_est_total || 0) - Number(a.prompt_tokens_est_total || 0);
        return 0;
      });
      return arr;
    }

    function applyFilters() {
      try {
        const q = (el.search.value || '').trim().toLowerCase();
        filtered = records.filter(r => {
          if (!r || typeof r !== 'object') return false;
          if (el.provider.value && r.provider !== el.provider.value) return false;
          if (el.model.value && r.generation_model !== el.model.value) return false;
          if (el.scenario.value && r.scenario !== el.scenario.value) return false;
          if (el.status.value && r.status !== el.status.value) return false;

          if (!q) return true;
          const hay = [
            r.run_id,
            r.provider,
            r.generation_model,
            r.scenario,
            r.status,
            r.error_message,
            r.error_type,
            r.run_dir,
          ].join(' ').toLowerCase();
          return hay.includes(q);
        });

        filtered = sortRows(filtered);
        renderRows();
      } catch (err) {
        console.error('applyFilters failed:', err);
        el.rows.innerHTML = '';
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 10;
        td.className = 'muted';
        td.textContent = 'Render error: ' + String(err && err.message ? err.message : err);
        tr.appendChild(td);
        el.rows.appendChild(tr);
      }
    }

    function renderRows() {
      el.rows.innerHTML = '';
      el.rowCount.textContent = '(' + filtered.length + ' visible)';

      for (const r of filtered) {
        const tr = document.createElement('tr');
        if (selectedRunId && selectedRunId === r.run_id) tr.classList.add('active');

        const errShort = (r.error_message || '').slice(0, 110);

        function appendCell(text, className) {
          const td = document.createElement('td');
          if (className) td.className = className;
          td.textContent = text;
          tr.appendChild(td);
          return td;
        }

        appendCell(String(r.run_id || '-'), 'mono');
        appendCell(fmtDate(r.run_started_at));
        appendCell(String(r.provider || '-'));
        appendCell(String(r.generation_model || '-'), 'clamp');
        appendCell(String(r.scenario || '-'));

        const statusTd = document.createElement('td');
        const badge = document.createElement('span');
        badge.className = statusClass(r.status);
        badge.textContent = String(r.status || 'unknown');
        statusTd.appendChild(badge);
        tr.appendChild(statusTd);

        appendCell(fmtNum(r.iterations_recorded));
        appendCell(fmtNum(r.llm_calls));
        appendCell(fmtNum(r.prompt_tokens_est_total));
        appendCell(String(errShort || '-'), 'clamp muted');

        tr.addEventListener('click', () => {
          selectedRunId = r.run_id;
          renderRows();
          renderDetail(r);
        });

        el.rows.appendChild(tr);
      }

      if (!filtered.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="10" class="muted">No runs match current filters.</td>';
        el.rows.appendChild(tr);
        renderDetail(null);
      }
    }

    function renderDetail(r) {
      el.detailChips.innerHTML = '';
      el.detailActions.innerHTML = '';
      if (!r) {
        el.detail.textContent = 'No run selected.';
        return;
      }

      const chips = [
        'Status: ' + (r.status || '-'),
        'Duration: ' + fmtDuration(r.duration_seconds),
        'LLM calls: ' + fmtNum(r.llm_calls),
        'Prompt tokens: ' + fmtNum(r.prompt_tokens_est_total),
        'Response tokens: ' + fmtNum(r.response_tokens_est_total),
      ];
      for (const c of chips) {
        const d = document.createElement('div');
        d.className = 'chip';
        d.textContent = c;
        el.detailChips.appendChild(d);
      }

      function addLink(label, href) {
        if (!href) return false;
        const a = document.createElement('a');
        a.href = adaptOpenUri(href);
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.textContent = label;
        el.detailActions.appendChild(a);
        return true;
      }

      const versions = Array.isArray(r.iteration_options) ? r.iteration_options : [];
      let selectedIndex = -1;
      let picked = {
        iteration: r.selected_iteration,
        is_valid: r.selected_is_valid,
        dsl_uri: r.selected_dsl_uri,
        compiler_output_uri: r.selected_compiler_output_uri,
        dsl_text: r.selected_dsl_text,
        compiler_output_text: r.selected_compiler_output_text,
      };

      if (versions.length) {
        const hasUiSelection = Object.prototype.hasOwnProperty.call(r, 'ui_selected_iteration');
        selectedIndex = versions.length - 1;
        if (hasUiSelection) {
          const idx = versions.findIndex(v => String(v.iteration) === String(r.ui_selected_iteration));
          if (idx >= 0) selectedIndex = idx;
        }

        const select = document.createElement('select');
        for (let i = 0; i < versions.length; i += 1) {
          const v = versions[i];
          const opt = document.createElement('option');
          const status = v.is_valid === true ? 'valid' : (v.is_valid === false ? 'invalid' : 'unknown');
          opt.value = String(i);
          opt.textContent = 'Iteration ' + String(v.iteration ?? '-') + ' (' + status + ')';
          if (i === selectedIndex) opt.selected = true;
          select.appendChild(opt);
        }
        picked = versions[selectedIndex] || picked;
        select.addEventListener('change', (ev) => {
          const idx = Number(ev.target.value);
          const next = versions[idx];
          if (!next) return;
          renderDetail({
            ...r,
            ui_selected_iteration: next.iteration,
            selected_iteration: next.iteration,
            selected_is_valid: next.is_valid,
            selected_dsl_uri: next.dsl_uri,
            selected_compiler_output_uri: next.compiler_output_uri,
            selected_dsl_text: next.dsl_text,
            selected_compiler_output_text: next.compiler_output_text,
            iteration_options: versions,
          });
        });
        el.detailActions.appendChild(select);
      }

      const dslLabel = picked.is_valid === true ? 'Open Selected Valid DSL (.LIRAs)' : 'Open Selected DSL (.LIRAs)';
      const compilerLabel = picked.is_valid === true ? 'Open Selected Valid Compiler Output (.txt)' : 'Open Selected Compiler Output (.txt)';

      const hasVsCodeDsl = vscodeMode
        ? addLink('Open Selected DSL in VS Code', picked.dsl_uri)
        : false;
      const hasVsCodeCompiler = vscodeMode
        ? addLink('Open Selected Compiler Output in VS Code', picked.compiler_output_uri)
        : false;

      // In VS Code mode, keep the action bar focused only on VS Code buttons.
      const hasSelectedDsl = vscodeMode ? false : addLink(dslLabel, picked.dsl_uri);
      const hasSelectedCompiler = vscodeMode ? false : addLink(compilerLabel, picked.compiler_output_uri);
      const hasMetadata = vscodeMode ? false : addLink('Open Run Metadata (JSON)', r.metadata_uri);

      if (!hasVsCodeDsl && !hasVsCodeCompiler && !hasSelectedDsl && !hasSelectedCompiler && !hasMetadata) {
        const noArtifacts = document.createElement('span');
        noArtifacts.className = 'muted';
        noArtifacts.textContent = 'No artifacts available for this run yet.';
        el.detailActions.appendChild(noArtifacts);
      }

      if (artifactsDetailMode) {
        const compilerTitle = picked.is_valid === true
          ? 'Compiler output (selected valid iteration)'
          : 'Compiler output (selected iteration)';
        const dslTitle = picked.is_valid === true
          ? 'LIRAS code (selected valid iteration, line numbered)'
          : 'LIRAS code (selected iteration, line numbered)';

        const compilerBody = (picked.compiler_output_text || '').trim();
        const dslBody = withLineNumbers(picked.dsl_text || '');
        const sections = [];

        sections.push('<span class="section-title">=== ' + escapeHtml(compilerTitle) + ' ===</span>');
        sections.push(escapeHtml(compilerBody || '[compiler output not available]'));
        sections.push('');
        sections.push('<span class="section-title">=== ' + escapeHtml(dslTitle) + ' ===</span>');
        sections.push(escapeHtml(dslBody || '[LIRAS code not available]'));

        if (lirasDiffMode) {
          const prevIter = selectedIndex > 0 ? versions[selectedIndex - 1] : null;
          sections.push('');
          sections.push('<span class="section-title">=== Diff vs previous iteration (LIRAS) ===</span>');
          if (prevIter && typeof prevIter.dsl_text === 'string') {
            const diffBody = buildLirasDiffHtml(prevIter.dsl_text, picked.dsl_text || '');
            sections.push(diffBody || '<span class="diff-line ctx">[no differences]</span>');
          } else {
            sections.push('<span class="diff-line ctx">[no previous iteration available]</span>');
          }
        }

        el.detail.innerHTML = sections.join('\\n');
      } else {
        el.detail.textContent = JSON.stringify(r, null, 2);
      }
    }

    function wireEvents() {
      const fns = [el.search, el.provider, el.model, el.scenario, el.status, el.sortBy];
      for (const x of fns) x.addEventListener('input', applyFilters);
      for (const x of fns) x.addEventListener('change', applyFilters);
    }

    function renderBootstrapError() {
      if (!bootstrapError) return;
      el.detail.textContent = [
        'Dashboard bootstrap error:',
        bootstrapError,
        '',
        'Try regenerating the HTML and reopening it.'
      ].join('\\n');
    }

    renderCards();
    setupFilters();
    wireEvents();
    renderBootstrapError();
    applyFilters();
  </script>
</body>
</html>
"""
    return html.replace("__DASHBOARD_DATA__", safe_data_json)


def build_dashboard(
    runs_dir: Path,
    output_html: Path,
    pretty: bool = False,
    vscode_open_mode: bool = False,
    artifacts_detail_mode: bool = False,
    embed_all_data: bool = False,
  liras_diff_mode: bool = False,
) -> None:
    records = _collect_records_with_options(runs_dir, embed_all_data=embed_all_data)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "root": str(ROOT),
        "runs_dir": str(runs_dir),
        "vscode_open_mode": bool(vscode_open_mode),
        "artifacts_detail_mode": bool(artifacts_detail_mode),
        "embed_all_data": bool(embed_all_data),
        "liras_diff_mode": bool(liras_diff_mode),
        "summary": _build_summary(records),
        "records": records,
    }

    html = _build_html(payload)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")

    if pretty:
        print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))

    print(f"[OK] Dashboard written: {output_html}")
    print("Open this file in your browser.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a standalone HTML dashboard for LIRAS runs.")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="Runs directory to scan")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output HTML path")
    parser.add_argument("--summary", action="store_true", help="Print compact summary to stdout")
    parser.add_argument(
      "--vscode-open-mode",
      action="store_true",
      help="Generate dashboard links/actions that open selected iteration artifacts with VS Code (vscode:// URIs)",
    )
    parser.add_argument(
      "--artifacts-detail-mode",
      action="store_true",
      help="Show selected iteration compiler output + line-numbered LIRAs in detail panel instead of JSON payload",
    )
    parser.add_argument(
      "--embed-all-data",
      action="store_true",
      help="Embed metadata and full artifacts text into the HTML to make it portable/shareable without external folders",
    )
    parser.add_argument(
      "--liras-diff-mode",
      action="store_true",
      help="Show Git-style LIRAS diff against the previous iteration in the detail panel",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    if not runs_dir.is_absolute():
        runs_dir = ROOT / runs_dir

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    build_dashboard(
      runs_dir=runs_dir,
      output_html=output,
      pretty=args.summary,
      vscode_open_mode=args.vscode_open_mode,
      artifacts_detail_mode=args.artifacts_detail_mode,
      embed_all_data=args.embed_all_data,
      liras_diff_mode=args.liras_diff_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
