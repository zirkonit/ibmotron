from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ibm650_it import REPO_ROOT
from ibm650_it.cloud.runpod import RunpodCtl
from ibm650_it.eval.locking import FINALIZE_STATE_FILENAME


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IBM 650 Job Dashboard</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #131a2b;
      --panel-2: #1a2338;
      --text: #edf2ff;
      --muted: #9fb0d1;
      --good: #38c172;
      --warn: #f6ad55;
      --bad: #f56565;
      --accent: #63b3ed;
      --border: #27324d;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #12213d 0%, var(--bg) 50%);
      color: var(--text);
    }
    header {
      padding: 24px;
      border-bottom: 1px solid var(--border);
      background: rgba(10, 15, 28, 0.75);
      backdrop-filter: blur(8px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0; font-size: 28px; }
    p.meta { margin: 6px 0 0; color: var(--muted); }
    main {
      padding: 20px;
      display: grid;
      gap: 20px;
    }
    .grid {
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }
    .panel {
      background: linear-gradient(180deg, rgba(26,35,56,0.95), rgba(19,26,43,0.95));
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }
    .panel h2 {
      margin: 0 0 14px;
      font-size: 18px;
    }
    .job-card {
      border-top: 1px solid var(--border);
      padding-top: 14px;
      margin-top: 14px;
    }
    .job-card:first-child {
      border-top: 0;
      margin-top: 0;
      padding-top: 0;
    }
    .row {
      display: grid;
      grid-template-columns: 132px 1fr;
      gap: 8px;
      margin: 6px 0;
      align-items: start;
    }
    .label { color: var(--muted); }
    .value { word-break: break-word; }
    .mono { font-family: var(--mono); font-size: 12px; }
    .chip {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      margin-right: 6px;
      margin-bottom: 6px;
      border: 1px solid transparent;
    }
    .good { background: rgba(56, 193, 114, 0.12); color: var(--good); border-color: rgba(56, 193, 114, 0.35); }
    .warn { background: rgba(246, 173, 85, 0.12); color: var(--warn); border-color: rgba(246, 173, 85, 0.35); }
    .bad { background: rgba(245, 101, 101, 0.12); color: var(--bad); border-color: rgba(245, 101, 101, 0.35); }
    .accent { background: rgba(99, 179, 237, 0.12); color: var(--accent); border-color: rgba(99, 179, 237, 0.35); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 8px 10px;
      border-top: 1px solid var(--border);
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 600; }
    .muted { color: var(--muted); }
    .empty {
      color: var(--muted);
      font-style: italic;
      padding: 8px 0 0;
    }
    .small { font-size: 12px; }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
    }
    .metric {
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(14, 20, 34, 0.65);
    }
    .metric .name { color: var(--muted); font-size: 12px; }
    .metric .number { font-size: 20px; font-weight: 700; margin-top: 4px; }
  </style>
</head>
<body>
  <header>
    <h1>IBM 650 Job Dashboard</h1>
    <p class="meta" id="meta">Loading…</p>
  </header>
  <main>
    <section class="panel">
      <h2>Active Jobs</h2>
      <div id="active-jobs"></div>
    </section>
    <section class="grid">
      <section class="panel">
        <h2>Active Pods</h2>
        <div id="pods"></div>
      </section>
      <section class="panel">
        <h2>Recent Runs</h2>
        <div id="recent-runs"></div>
      </section>
    </section>
  </main>
  <script>
    let refreshHandle = null;

    function badge(label, cls) {
      return `<span class="chip ${cls}">${label}</span>`;
    }

    function safe(value) {
      if (value === null || value === undefined || value === "") return '<span class="muted">n/a</span>';
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function renderProgress(progress) {
      if (!progress || Object.keys(progress).length === 0) return '<span class="muted">n/a</span>';
      return Object.entries(progress).map(([name, data]) => {
        const parts = [];
        if (data.lines !== undefined) parts.push(`${data.lines}/${data.expected ?? "?"}`);
        if (data.summary_exists) parts.push("summary");
        if (data.report_exists) parts.push("report");
        return `<div class="small"><span class="mono">${name}</span>: ${safe(parts.join(" · "))}</div>`;
      }).join("");
    }

    function renderActiveJobs(jobs) {
      const root = document.getElementById("active-jobs");
      if (!jobs.length) {
        root.innerHTML = '<div class="empty">No active local wrapper processes.</div>';
        return;
      }
      root.innerHTML = jobs.map((job) => `
        <div class="job-card">
          <div>
            ${badge(job.phase || "unknown", job.phase_class || "warn")}
            ${badge(job.run_mode || "job", "accent")}
            ${job.remote && job.remote.active_process ? badge("remote active", "good") : ""}
          </div>
          <div class="row"><div class="label">Name</div><div class="value mono">${safe(job.name)}</div></div>
          <div class="row"><div class="label">Local PID</div><div class="value mono">${safe(job.pid)} · ${safe(job.elapsed)}</div></div>
          <div class="row"><div class="label">Pod</div><div class="value mono">${safe(job.pod_id || "")} ${job.pod ? `· ${safe(job.pod.gpu || "")} · $${safe(job.pod.cost_per_hr)}/hr` : ""}</div></div>
          <div class="row"><div class="label">Remote</div><div class="value">${job.remote ? safe(job.remote.phase || "unknown") : '<span class="muted">unavailable</span>'}</div></div>
          <div class="row"><div class="label">Local</div><div class="value">${job.local ? safe(job.local.phase || "unknown") : '<span class="muted">n/a</span>'}</div></div>
          <div class="row"><div class="label">Progress</div><div class="value">${renderProgress(job.remote ? job.remote.progress : {})}</div></div>
          <div class="row"><div class="label">GPU</div><div class="value mono">${job.remote && job.remote.gpu ? safe(job.remote.gpu) : '<span class="muted">n/a</span>'}</div></div>
          <div class="row"><div class="label">Output</div><div class="value mono">${safe(job.remote_output)}</div></div>
          <div class="row"><div class="label">Local Output</div><div class="value mono">${safe(job.local_output)}</div></div>
          <div class="row"><div class="label">Command</div><div class="value mono">${safe(job.command)}</div></div>
        </div>
      `).join("");
    }

    function renderPods(pods) {
      const root = document.getElementById("pods");
      if (!pods.length) {
        root.innerHTML = '<div class="empty">No unmatched active pods.</div>';
        return;
      }
      root.innerHTML = `
        <table>
          <thead><tr><th>Name</th><th>Pod</th><th>GPU</th><th>Cost</th><th>Status</th></tr></thead>
          <tbody>
            ${pods.map((pod) => `
              <tr>
                <td class="mono">${safe(pod.name)}</td>
                <td class="mono">${safe(pod.id)}</td>
                <td>${safe(pod.gpu)}</td>
                <td>$${safe(pod.cost_per_hr)}/hr</td>
                <td>${safe(pod.status)}</td>
              </tr>`).join("")}
          </tbody>
        </table>`;
    }

    function renderRun(run) {
      const train = run.train || {};
      const evals = run.evaluations || {};
      const fine = run.fine_tuned || {};
      let metrics = "";
      if (fine.report) {
        metrics = `
          <div class="summary-grid">
            <div class="metric"><div class="name">Exact</div><div class="number">${(100 * fine.report.exact_match).toFixed(1)}%</div></div>
            <div class="metric"><div class="name">Assemblable</div><div class="number">${(100 * fine.report.assemblability).toFixed(1)}%</div></div>
            <div class="metric"><div class="name">Functional</div><div class="number">${(100 * fine.report.functional_equivalence).toFixed(1)}%</div></div>
          </div>`;
      } else if (evals.fine_tuned && evals.fine_tuned.report) {
        metrics = `
          <div class="summary-grid">
            <div class="metric"><div class="name">Few-shot Asm</div><div class="number">${(100 * evals.few_shot.report.assemblability).toFixed(1)}%</div></div>
            <div class="metric"><div class="name">Fine Exact</div><div class="number">${(100 * evals.fine_tuned.report.exact_match).toFixed(1)}%</div></div>
            <div class="metric"><div class="name">Fine Asm</div><div class="number">${(100 * evals.fine_tuned.report.assemblability).toFixed(1)}%</div></div>
          </div>`;
      }
      return `
        <div class="job-card">
          <div>
            ${badge(run.kind || "run", "accent")}
            ${badge(run.eval_mode || "unknown", run.eval_mode === "local_cpu_reevaluate" ? "good" : "warn")}
          </div>
          <div class="row"><div class="label">Path</div><div class="value mono">${safe(run.path)}</div></div>
          <div class="row"><div class="label">Updated</div><div class="value">${safe(run.updated_at)}</div></div>
          <div class="row"><div class="label">Backend</div><div class="value">${safe(train.backend)} ${train.qlora_bits !== undefined ? `· qlora_bits=${safe(train.qlora_bits)}` : ""}</div></div>
          ${metrics}
        </div>`;
    }

    function renderRecentRuns(runs) {
      const root = document.getElementById("recent-runs");
      if (!runs.length) {
        root.innerHTML = '<div class="empty">No local run summaries found.</div>';
        return;
      }
      root.innerHTML = runs.map(renderRun).join("");
    }

    async function refresh() {
      const response = await fetch("/api/status");
      const data = await response.json();
      document.getElementById("meta").textContent = `Updated ${data.generated_at} · refresh every ${data.refresh_seconds}s`;
      renderActiveJobs(data.active_jobs || []);
      renderPods(data.orphan_pods || []);
      renderRecentRuns(data.recent_runs || []);
      if (refreshHandle === null) {
        refreshHandle = setInterval(() => refresh().catch(() => {}), (data.refresh_seconds || 10) * 1000);
      }
    }

    refresh().catch((error) => {
      document.getElementById("meta").textContent = `Failed to load dashboard data: ${error}`;
    });
  </script>
</body>
</html>
"""


PS_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(.*)$")


@dataclass(slots=True)
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    refresh_seconds: int = 10
    cache_ttl_seconds: int = 5
    remote_timeout_seconds: int = 10


def _run_command(command: list[str], *, timeout_seconds: int = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def _parse_launcher_args(command: str) -> dict[str, Any]:
    argv = shlex.split(command)
    if "scripts/runpod_train_eval.py" not in argv:
        return {}
    script_index = argv.index("scripts/runpod_train_eval.py")
    flags = argv[script_index + 1 :]
    parsed: dict[str, Any] = {}
    index = 0
    while index < len(flags):
        token = flags[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        if index + 1 < len(flags) and not flags[index + 1].startswith("--"):
            value_tokens: list[str] = []
            next_index = index + 1
            while next_index < len(flags) and not flags[next_index].startswith("--"):
                value_tokens.append(flags[next_index])
                next_index += 1
            parsed[key] = " ".join(value_tokens)
            index = next_index
            continue
        parsed[key] = True
        index += 1
    return parsed


def collect_active_wrappers() -> list[dict[str, Any]]:
    proc = _run_command(["ps", "-axo", "pid=,etime=,command="], timeout_seconds=5)
    jobs: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        match = PS_RE.match(line)
        if not match:
            continue
        pid, elapsed, command = match.groups()
        if "scripts/runpod_train_eval.py" not in command:
            continue
        args = _parse_launcher_args(command)
        jobs.append(
            {
                "pid": int(pid),
                "elapsed": elapsed,
                "command": command,
                "name": args.get("name"),
                "run_mode": args.get("run_mode"),
                "gpu_id": args.get("gpu_id"),
                "dataset_name": args.get("dataset_name"),
                "dataset_index": args.get("dataset_index"),
                "remote_output": args.get("remote_output"),
                "local_output": args.get("local_output"),
                "limit": int(args["limit"]) if args.get("limit") else None,
                "example_count": int(args["example_count"]) if args.get("example_count") else None,
                "pod_id_arg": args.get("pod_id"),
            }
        )
    return jobs


def _read_archive_size(pid: int) -> int | None:
    proc = _run_command(["lsof", "-p", str(pid)], timeout_seconds=5)
    for line in proc.stdout.splitlines():
        if "ibmotron-runpod.tgz" not in line:
            continue
        parts = line.split()
        if len(parts) >= 7 and parts[6].isdigit():
            return int(parts[6])
    temp_archive = Path(tempfile_dir()) / "ibmotron-runpod.tgz"
    return temp_archive.stat().st_size if temp_archive.exists() else None


def tempfile_dir() -> str:
    return os.environ.get("TMPDIR", "/tmp")


def collect_pods() -> list[dict[str, Any]]:
    try:
        ctl = RunpodCtl()
        rows = ctl.list_pods(all_pods=True)
    except Exception as exc:
        return [{"id": "", "name": "runpod_error", "gpu": "", "cost_per_hr": "", "status": str(exc)}]
    pods: list[dict[str, Any]] = []
    for pod in rows:
        pods.append(
            {
                "id": str(pod.get("id", "")),
                "name": str(pod.get("name", "")),
                "gpu": str(pod.get("machine", {}).get("gpuDisplayName") or pod.get("gpuDisplayName") or pod.get("gpuTypeId") or ""),
                "cost_per_hr": pod.get("costPerHr"),
                "status": str(pod.get("desiredStatus", "")),
            }
        )
    return pods


def _expected_modes(run_mode: str) -> list[str]:
    if run_mode == "overfit-sanity":
        return ["fine_tuned"]
    if run_mode == "thinking-ablation":
        return ["thinking_on", "thinking_off"]
    return ["zero_shot", "few_shot", "fine_tuned"]


def _inspect_local_output(local_output: str | None, run_mode: str) -> dict[str, Any] | None:
    if not local_output:
        return None
    root = REPO_ROOT / local_output
    if not root.exists():
        return None

    state_path = root / FINALIZE_STATE_FILENAME
    state = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = None

    modes = _expected_modes(run_mode)
    reports = {
        mode: {
            "exists": (root / "reports" / f"{mode}.json").exists(),
        }
        for mode in modes
    }
    summary_exists = (root / "summary.json").exists()
    completed_reports = sum(1 for item in reports.values() if item["exists"])
    if state is not None:
        phase = "failed" if state.get("status") == "failed" else "local_reevaluate"
    elif summary_exists and completed_reports == len(modes):
        phase = "complete"
    elif summary_exists or completed_reports:
        phase = "local_reevaluate"
    else:
        phase = "pending"
    return {
        "root": str(root),
        "phase": phase,
        "summary_exists": summary_exists,
        "state": state,
        "reports": reports,
    }


def _ssh_remote_status(pod_id: str, remote_output: str, run_mode: str, expected_count: int | None, timeout_seconds: int) -> dict[str, Any]:
    ctl = RunpodCtl()
    info = ctl.ssh_info(pod_id)
    user, host, port = ctl._ssh_target(info)
    remote_script = f"""
import json
import subprocess
from pathlib import Path

root = Path("/workspace/ibmotron")
output = root / {json.dumps(remote_output)}
workspace = Path("/workspace")
data = {{
    "tgz_exists": any((workspace / name).exists() for name in ["ibmotron.tgz", "ibmotron-base.tgz", "ibmotron-dataset.tgz"]),
    "repo_exists": root.exists(),
    "output_exists": output.exists(),
    "progress": {{}},
    "active_process": False,
    "gpu": None,
}}
try:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    data["gpu"] = gpu or None
except Exception:
    pass
ps_text = subprocess.run(
    ["bash", "-lc", "ps -eo pid,etime,pcpu,pmem,command | egrep 'python -m ibm650_it.cli (train-eval|overfit-sanity)' | grep -v grep || true"],
    text=True,
    capture_output=True,
    check=False,
).stdout.strip()
data["process"] = ps_text
data["active_process"] = bool(ps_text)
if output.exists():
    if {json.dumps(run_mode)} == "overfit-sanity":
        modes = ["fine_tuned"]
    else:
        modes = ["zero_shot", "few_shot", "fine_tuned"]
    for mode in modes:
        pred_dir = output / "predictions" / mode
        pred_index = pred_dir / "predictions.jsonl"
        summary = pred_dir / "summary.json"
        report = output / "reports" / f"{{mode}}.json"
        lines = 0
        if pred_index.exists():
            lines = sum(1 for line in pred_index.read_text(encoding="utf-8").splitlines() if line.strip())
        data["progress"][mode] = {{
            "lines": lines,
            "expected": {expected_count if expected_count is not None else "None"},
            "summary_exists": summary.exists(),
            "report_exists": report.exists(),
        }}
print(json.dumps(data))
"""
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-i",
            str(Path.home() / ".ssh" / "id_ed25519"),
            "-p",
            port,
            f"{user}@{host}",
            f"python3 - <<'PY'\n{remote_script}\nPY",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or proc.stdout.strip() or f"ssh failed: {proc.returncode}"}
    return json.loads(proc.stdout)


def _phase_class(phase: str) -> str:
    if phase in {"complete"}:
        return "good"
    if phase in {"remote_bootstrap", "remote_train", "remote_generate", "local_reevaluate"}:
        return "accent"
    if phase in {"failed", "runpod_error"}:
        return "bad"
    return "warn"


def _derive_remote_subphase(remote: dict[str, Any] | None) -> str | None:
    if remote is None or remote.get("error"):
        return None
    progress = remote.get("progress", {})
    for mode in ["fine_tuned", "few_shot", "zero_shot"]:
        mode_progress = progress.get(mode, {})
        expected = mode_progress.get("expected")
        lines = mode_progress.get("lines")
        if lines and expected and lines < expected:
            return mode
        if expected and lines == expected:
            continue
    if remote.get("active_process"):
        return "training"
    return None


def _derive_remote_phase(remote: dict[str, Any] | None) -> str | None:
    if remote is None:
        return None
    if remote.get("error"):
        return "failed"
    if not remote.get("repo_exists"):
        return "remote_bootstrap"
    if not remote.get("output_exists"):
        return "remote_train"
    if remote.get("active_process"):
        return _derive_remote_subphase(remote) or "remote_train"
    if remote.get("output_exists"):
        return "remote_complete"
    return "remote_bootstrap"


def _derive_phase(job: dict[str, Any], remote: dict[str, Any] | None, local: dict[str, Any] | None) -> str:
    archive_size = _read_archive_size(job["pid"])
    if local is not None and local.get("phase") == "failed":
        return "failed"
    if local is not None and local.get("phase") == "local_reevaluate":
        return "local_reevaluate"
    if local is not None and local.get("phase") == "complete":
        return "complete"
    if remote is None:
        return "remote_bootstrap" if archive_size else "launching"
    if remote.get("error"):
        return "failed"
    if not remote.get("repo_exists"):
        return "remote_bootstrap"
    if not remote.get("output_exists"):
        return "remote_train"
    if remote.get("active_process"):
        return "remote_generate" if _derive_remote_subphase(remote) else "remote_train"
    if local is not None and local.get("summary_exists"):
        return "local_reevaluate"
    return "remote_generate"


def _match_pod(job: dict[str, Any], pods_by_id: dict[str, dict[str, Any]], pods_by_name: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    pod_id_arg = str(job.get("pod_id_arg") or "")
    if pod_id_arg:
        pod = pods_by_id.get(pod_id_arg)
        if pod is not None:
            return pod
    job_name = str(job.get("name") or "")
    if job_name:
        return pods_by_name.get(job_name)
    return None


def collect_recent_runs(limit: int = 8) -> list[dict[str, Any]]:
    eval_root = REPO_ROOT / "artifacts" / "eval_reports"
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(eval_root.rglob("summary.json"), key=lambda path: path.stat().st_mtime, reverse=True):
        if any(part in {"predictions", "reports", "failures", "model", "sft"} for part in summary_path.parts):
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        runs.append(
            {
                "path": str(summary_path.parent.relative_to(REPO_ROOT)),
                "updated_at": datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc).astimezone().isoformat(timespec="seconds"),
                "kind": "overfit" if "fine_tuned" in payload and "evaluations" not in payload else "train_eval",
                "eval_mode": payload.get("eval_mode"),
                "train": payload.get("train"),
                "evaluations": payload.get("evaluations"),
                "fine_tuned": payload.get("fine_tuned"),
            }
        )
        if len(runs) >= limit:
            break
    return runs


def collect_dashboard_status(config: DashboardConfig) -> dict[str, Any]:
    wrappers = collect_active_wrappers()
    pods = collect_pods()
    pods_by_id = {pod["id"]: pod for pod in pods if pod["id"]}
    pods_by_name = {pod["name"]: pod for pod in pods if pod["name"]}
    matched_pod_ids: set[str] = set()
    active_jobs: list[dict[str, Any]] = []
    for job in wrappers:
        pod = _match_pod(job, pods_by_id, pods_by_name)
        if pod and pod["id"]:
            matched_pod_ids.add(pod["id"])
        remote = None
        if pod and pod["id"]:
            expected = job.get("example_count") if job.get("run_mode") == "overfit-sanity" else job.get("limit")
            remote = _ssh_remote_status(
                pod["id"],
                str(job.get("remote_output") or ""),
                str(job.get("run_mode") or ""),
                expected,
                config.remote_timeout_seconds,
            )
            if remote is not None:
                remote["phase"] = _derive_remote_phase(remote)
        local = _inspect_local_output(str(job.get("local_output") or ""), str(job.get("run_mode") or ""))
        phase = _derive_phase(job, remote, local)
        active_jobs.append(
            {
                **job,
                "pod_id": pod["id"] if pod else None,
                "pod": pod,
                "remote": remote,
                "local": local,
                "phase": phase,
                "phase_class": _phase_class(phase),
            }
        )
    orphan_pods = [pod for pod in pods if pod["id"] and pod["id"] not in matched_pod_ids]
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "refresh_seconds": config.refresh_seconds,
        "active_jobs": active_jobs,
        "orphan_pods": orphan_pods,
        "recent_runs": collect_recent_runs(),
    }


class SnapshotCache:
    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._expires_at = 0.0
        self._snapshot: dict[str, Any] | None = None

    def get(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if self._snapshot is not None and now < self._expires_at:
                return self._snapshot
            self._snapshot = collect_dashboard_status(self.config)
            self._expires_at = now + self.config.cache_ttl_seconds
            return self._snapshot


def build_handler(cache: SnapshotCache) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/api/status"):
                payload = json.dumps(cache.get(), indent=2).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return DashboardHandler


def serve_dashboard(*, host: str = "127.0.0.1", port: int = 8765, refresh_seconds: int = 10) -> None:
    config = DashboardConfig(host=host, port=port, refresh_seconds=refresh_seconds)
    cache = SnapshotCache(config)
    server = ThreadingHTTPServer((host, port), build_handler(cache))
    print(f"IBM 650 dashboard listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh-seconds", type=int, default=10)
    args = parser.parse_args(argv)
    serve_dashboard(host=args.host, port=args.port, refresh_seconds=args.refresh_seconds)
