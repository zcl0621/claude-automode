"""Build what the classifier is allowed to see.

It sees ONLY what the user typed and the proposed action. Assistant text and tool
results are left out on purpose: both can carry injected instructions.
"""
import json
from typing import List

MAX_MSG_CHARS = 2000
MAX_INPUT_CHARS = 20000      # total rendered action
MAX_FIELD_CHARS = 4000       # any single string field; longer ones keep head + tail
TRUNCATED_MARK = "[automode: input truncated]"


def user_messages(transcript_path: str, limit: int) -> List[str]:
    msgs: List[str] = []
    try:
        with open(transcript_path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") != "user" or e.get("isMeta") or e.get("isSidechain"):
                    continue
                content = e.get("message", {}).get("content")
                texts = []
                if isinstance(content, str):
                    texts = [content]
                elif isinstance(content, list):
                    # tool_result blocks are tool output, not the user: skip them
                    texts = [b.get("text", "") for b in content if b.get("type") == "text"]
                for t in texts:
                    t = t.strip()
                    if t and not _is_harness_text(t):
                        msgs.append(t[:MAX_MSG_CHARS])
    except OSError:
        return []
    return msgs[-limit:]


MAX_PRIOR_ACTION_CHARS = 300


def prior_actions(transcript_path: str, limit: int = 8) -> List[str]:
    """The agent's earlier tool calls in this session: name + arguments, no results, no prose."""
    out: List[str] = []
    try:
        with open(transcript_path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") != "assistant" or e.get("isSidechain"):
                    continue
                content = e.get("message", {}).get("content")
                if not isinstance(content, list):
                    continue
                for b in content:
                    if b.get("type") == "tool_use":
                        out.append(summarize_call(b.get("name", ""), b.get("input") or {}))
    except OSError:
        return []
    return out[-limit:]


def summarize_call(tool_name: str, tool_input) -> str:
    shown = {k: v for k, v in tool_input.items() if k not in AGENT_PROSE_FIELDS} \
        if isinstance(tool_input, dict) else tool_input
    s = "%s %s" % (tool_name, json.dumps(shown, ensure_ascii=False))
    return s if len(s) <= MAX_PRIOR_ACTION_CHARS else s[:MAX_PRIOR_ACTION_CHARS] + "…"


HARNESS_PREFIXES = ("<system-reminder>", "<command-", "[Request interrupted", "<task-notification",
                    "<local-command", "<bash-input>", "<bash-stdout>", "<user-prompt-submit-hook>",
                    "[SYSTEM NOTIFICATION", "Caveat: The messages below")


def _is_harness_text(t: str) -> bool:
    # system reminders, background-task notifications, slash-command echoes, interrupt markers
    return t.startswith(HARNESS_PREFIXES)


MAX_PROPOSAL_CHARS = 1500


def agent_proposal(transcript_path: str) -> str:
    """The agent's last text right before the user's latest real message.

    Short replies ("ok", "清", "1 和 2", "跑") approve something the agent proposed; without the
    proposal the classifier cannot tell what was approved. This is the agent's text, so it is shown
    as what the reply refers to, never as authority.
    """
    last_agent_text, proposal = "", ""
    try:
        with open(transcript_path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("isSidechain"):
                    continue
                content = e.get("message", {}).get("content")
                if e.get("type") == "assistant" and isinstance(content, list):
                    texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    if any(t.strip() for t in texts):
                        last_agent_text = "\n".join(texts).strip()
                elif e.get("type") == "user" and not e.get("isMeta"):
                    if _is_real_user_text(content):
                        proposal = last_agent_text
    except OSError:
        return ""
    if len(proposal) > MAX_PROPOSAL_CHARS:
        proposal = "…" + proposal[-MAX_PROPOSAL_CHARS:]
    return proposal


def _is_real_user_text(content) -> bool:
    if isinstance(content, str):
        return bool(content.strip()) and not _is_harness_text(content.strip())
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
                   and not _is_harness_text(b.get("text", "").strip()) for b in content)
    return False


# Fields the agent writes to explain itself. They are not part of what runs, and an agent
# (or an injection steering it) could use them to argue with the classifier.
AGENT_PROSE_FIELDS = {"description"}


def render_action(tool_name: str, tool_input: dict) -> str:
    """Render the action so that nothing can hide past a cut-off.

    Long string fields are shortened individually (head + tail), so every key, every list item
    and every file path stays visible. A blind cut of the whole JSON once let a credential file
    at the end of a push_files call go unseen.
    """
    shown = {k: v for k, v in tool_input.items() if k not in AGENT_PROSE_FIELDS}
    clipped = [False]
    s = json.dumps(_shrink(shown, clipped), ensure_ascii=False, indent=1)
    header = "tool: %s\n" % tool_name
    paths = _paths(tool_input)
    if paths:
        header += "paths touched: %s\n" % ", ".join(paths)
    if len(s) > MAX_INPUT_CHARS:
        half = MAX_INPUT_CHARS // 2
        s = s[:half] + "\n...\n" + s[-half:]
        clipped[0] = True
    if clipped[0]:
        header += TRUNCATED_MARK + " parts of long fields are omitted; the paths above are complete.\n"
    return header + "input:\n" + s


def is_truncated(rendered: str) -> bool:
    return TRUNCATED_MARK in rendered


def _shrink(v, clipped):
    if isinstance(v, str) and len(v) > MAX_FIELD_CHARS:
        clipped[0] = True
        h = MAX_FIELD_CHARS // 2
        return v[:h] + "\n...[%d chars omitted]...\n" % (len(v) - 2 * h) + v[-h:]
    if isinstance(v, dict):
        return {k: _shrink(x, clipped) for k, x in v.items()}
    if isinstance(v, list):
        return [_shrink(x, clipped) for x in v]
    return v


PATH_KEYS = {"path", "file_path", "filepath", "filename", "notebook_path"}


def _paths(v, out=None):
    out = [] if out is None else out
    if isinstance(v, dict):
        for k, x in v.items():
            if k in PATH_KEYS and isinstance(x, str):
                out.append(x)
            else:
                _paths(x, out)
    elif isinstance(v, list):
        for x in v:
            _paths(x, out)
    return out
