"""Replay real local Claude Code sessions through automode (rules + classifier).

The user agreed to send a SAMPLE of their history to the configured classifier API.
Before anything leaves the machine, credential-looking strings are replaced with <REDACTED_SECRET>.

Ground truth: these calls actually ran in the user's sessions, so almost all were wanted.
A deny/ask on one is, to first order, a false block; flagged calls are written out for review.

For each sampled call the classifier sees what the hook would have seen at that moment:
the user's messages so far and the agent's earlier tool calls.

Usage: python3 eval/replay_sessions.py --n 2000 --out eval/results/replay_v6.jsonl [--workers 8]
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plugin", "scripts"))

from automode import classifier, config, context, rules  # noqa: E402

EXCLUDE_DIR = re.compile(r"claude-auomode|scratch-workspaces|scratch-2026")
EXCLUDE_TOOL = re.compile(r"^mcp__ccd_")  # the desktop app's own UI tools never prompt
REDACT = re.compile(rules.SECRETISH.pattern + r"[^\s\"'`,;}]*", re.I)
EXTRA_REDACT = [
    (re.compile(r"(://[^:/\s@]+:)[^@/\s]+@"), r"\1<REDACTED_SECRET>@"),                  # user:pass@host
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]{12,}", re.I), r"\1<REDACTED_SECRET>"),
    (re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "<REDACTED_SECRET>"),  # JWT
    (re.compile(r"((?:token|passwd|pwd|pass|auth|access_key|secret_key|private_key)[\"']?\s*[=:]\s*[\"']?)"
                r"[^\s\"'`,;}&]{6,}", re.I), r"\1<REDACTED_SECRET>"),
    (re.compile(r"\b(PGPASSWORD|MYSQL_PWD|AWS_SECRET_ACCESS_KEY|[A-Z_]*TOKEN|[A-Z_]*SECRET)=\S+"), r"\1=<REDACTED_SECRET>"),
]


def redact(s: str) -> str:
    s = REDACT.sub("<REDACTED_SECRET>", s)
    for pat, rep in EXTRA_REDACT:
        s = pat.sub(rep, s)
    return s


def index_calls(files):
    """Every Bash / MCP tool_use: (file, line_no, block_no, name, input)."""
    out = []
    for fp in files:
        try:
            with open(fp, errors="ignore") as f:
                for ln, line in enumerate(f):
                    if '"tool_use"' not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("type") != "assistant" or e.get("isSidechain"):
                        continue
                    for bi, b in enumerate(e.get("message", {}).get("content") or []):
                        if not isinstance(b, dict) or b.get("type") != "tool_use":
                            continue
                        name = b.get("name", "")
                        if (name == "Bash" or name.startswith("mcp__")) and not EXCLUDE_TOOL.match(name):
                            out.append((fp, ln, bi, name, b.get("input") or {}))
        except OSError:
            pass
    return out


def context_at(fp, stop_line, stop_block, limit_users=12):
    """User messages, earlier tool calls and the agent's proposal before (stop_line, stop_block)."""
    users, prior = [], []
    last_agent_text, proposal = "", ""
    with open(fp, errors="ignore") as f:
        for ln, line in enumerate(f):
            if ln > stop_line:
                break
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("isSidechain"):
                continue
            content = e.get("message", {}).get("content")
            if e.get("type") == "user" and not e.get("isMeta"):
                if context._is_real_user_text(content):
                    proposal = last_agent_text
                if isinstance(content, str):
                    t = content.strip()
                    if t and not context._is_harness_text(t):
                        users.append(t[:context.MAX_MSG_CHARS])
                elif isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "text":
                            t = b.get("text", "").strip()
                            if t and not context._is_harness_text(t):
                                users.append(t[:context.MAX_MSG_CHARS])
            elif e.get("type") == "assistant" and isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                if any(t.strip() for t in texts):
                    last_agent_text = "\n".join(texts).strip()
                for bi, b in enumerate(content):
                    if ln == stop_line and bi >= stop_block:
                        break
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        prior.append(context.summarize_call(b.get("name", ""), b.get("input") or {}))
    if len(proposal) > context.MAX_PROPOSAL_CHARS:
        proposal = "…" + proposal[-context.MAX_PROPOSAL_CHARS:]
    return users[-limit_users:], prior[-8:], proposal


def judge(cfg, item):
    fp, ln, bi, name, tin = item
    tin = tin if isinstance(tin, dict) else {"value": tin}
    users, prior, proposal = context_at(fp, ln, bi)
    prompt = classifier.build_prompt([redact(u) for u in users], redact(context.render_action(name, tin)), [],
                                     rules.command_facts(name, tin), [redact(p) for p in prior], redact(proposal))
    try:
        d, why, stage = classifier.classify(cfg, prompt)
    except classifier.ClassifierError as e:
        d, why, stage = "ask", "error: %s" % e, "error"
    return {"project": os.path.basename(os.path.dirname(fp)), "session": os.path.basename(fp),
            "tool": name, "kind": "Bash" if name == "Bash" else "MCP",
            "action": redact(context.summarize_call(name, tin))[:400], "last_user": redact(users[-1])[:300] if users else "",
            "decision": d, "stage": stage, "reason": why}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many shuffled calls: --offset 2000 gives a sample disjoint from the first 2000")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit("refusing to overwrite existing results file: " + args.out)
    cfg = config.load()
    files = [f for f in glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")) if not EXCLUDE_DIR.search(f)]
    calls = index_calls(files)
    local = collections.Counter()
    to_model = []
    for c in calls:
        d, _ = rules.check(c[3], c[4] if isinstance(c[4], dict) else {})
        kind = "Bash" if c[3] == "Bash" else "MCP"
        local[(kind, d or "model")] += 1
        if d is None:
            to_model.append(c)
    random.Random(args.seed).shuffle(to_model)
    sample = to_model[args.offset:args.offset + args.n]
    print("sessions=%d calls=%d (excluded this project, scratch dirs and mcp__ccd_*)" % (len(files), len(calls)))
    for k in sorted(local):
        print("  rules: %-5s %-6s %6d" % (k[0], k[1], local[k]))
    print("sampling %d of %d model-bound calls (offset %d)" % (len(sample), len(to_model), args.offset))

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        res = list(pool.map(lambda c: judge(cfg, c), sample))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("elapsed=%.1fs model: %s errors=%d" % (time.time() - t0, cfg.slow_model, sum(r["stage"] == "error" for r in res)))
    for kind in ("Bash", "MCP"):
        rs = [r for r in res if r["kind"] == kind]
        c = collections.Counter(r["decision"] for r in rs)
        print("  %-4s n=%4d allow %4d  deny %3d (%.1f%%)  ask %3d (%.1f%%)" % (
            kind, len(rs), c["allow"], c["deny"], 100.0 * c["deny"] / max(1, len(rs)),
            c["ask"], 100.0 * c["ask"] / max(1, len(rs))))
    print("per-call results:", args.out)


if __name__ == "__main__":
    main()
