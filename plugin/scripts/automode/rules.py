"""Static rule layer, checked before any model call.

Returns "allow", "deny", or None (no opinion -> go to the classifier).
Kept deliberately small: the classifier handles anything that depends on intent.
"""
import re
import shlex
from typing import Optional, Tuple

# Catastrophic and essentially never intended on a dev machine. Denied without asking a model.
# Matched on UNQUOTED text only: replaying the user's real sessions showed these patterns firing on
# `grep "mkfs"`, quoted test strings and sed expressions. Anything less certain (mkfs, curl | sh)
# is a COMMAND_FACT for the classifier instead, so a real SRE task is not hard-blocked.
HARD_DENY = [
    (r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/|~|\$HOME)(\s|/?$|/\*)", "recursive delete of / or home"),
    (r"\bdd\b[^|;&]*\bof=/dev/(disk|sd|nvme|hd)", "writes raw to a disk device"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"\bchmod\s+(-R\s+)?777\s+/(\s|$)", "chmod 777 on /"),
]

# Read-only commands that are safe as a whole command, with no shell operators.
# env / printenv are left out on purpose: they print secrets. echo is allowed only without `$`.
# cd / sleep / true change nothing outside the shell itself.
SAFE_FIRST_WORDS = {
    "ls", "pwd", "echo", "cat", "head", "tail", "wc", "grep", "rg", "find", "which", "file",
    "stat", "du", "df", "date", "whoami", "uname", "tree", "diff",
    "sort", "uniq", "basename", "dirname", "realpath", "cd", "sleep", "true", "type", "command",
    "sed",  # only `sed -n` without -i, checked below
}
ASSIGNMENT = re.compile(r"^(export\s+)?[A-Za-z_][A-Za-z0-9_]*=\S*$")
QUOTED = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"")
SAFE_GIT_SUBCOMMANDS = {"status", "log", "diff", "show", "rev-parse", "blame", "ls-files"}
# subcommands that are read-only only in their listing forms
GIT_LIST_FORMS = {
    "branch": {"-a", "-r", "-v", "-vv", "--list", "--all", "--show-current"},
    "remote": {"-v", "show", "get-url"},
}
SHELL_OPERATORS = re.compile(r"[;&|<>`]|\$\(")
SENSITIVE_PATHS = re.compile(r"(\.env\b|\.ssh/|id_rsa|id_ed25519|\.aws/|\.netrc|credentials|\.pem\b|\.key\b)")
FIND_ACTIONS = re.compile(r"\s-(delete|exec|execdir|ok|fprint)\b")
WRITES_OUTPUT_FLAG = {"sort", "tree"}  # `sort -o file`, `tree -o file`
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\s*(\n|$)", re.S)


# Facts about commands whose side effect is easy to miss. These are not decisions: they are
# appended to the action so the classifier knows what the command really does.
# (A holdout case: `crontab file.cron` was judged a read-only check; it installs the schedule.)
COMMAND_FACTS = [
    (r"\bmkfs(\.\w+)?\b", "`mkfs` formats a filesystem, destroying everything on that device."),
    (r"(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z)?sh\b",
     "This downloads a script and runs it immediately without review."),
    (r"(^|[;&|]\s*|\s)crontab\s+(?!-l\b)(-[ru]\s|\S+\.cron|\S+\s*($|[;&|]))",
     "`crontab <file>` installs that file as the user's cron schedule (replacing the current one); "
     "`crontab -r` deletes it. Only `crontab -l` is read-only."),
    (r"\blaunchctl\s+(load|bootstrap|enable|kickstart|submit)\b",
     "`launchctl load/bootstrap/enable` installs or starts a persistent background job."),
    (r"\bsystemctl\s+(--user\s+)?(enable|start|restart|stop|disable|mask)\b",
     "`systemctl enable/start/stop/disable` changes running or boot-time services."),
    (r"\bgit\s+(checkout\s+(--ours\s+|--theirs\s+)?--\s|restore\b(?!.*--staged\s*$)|clean\s+-[a-zA-Z]*f)",
     "`git checkout -- <path>`, `git restore <path>` and `git clean -f` discard uncommitted local changes."),
    (r"\bgit\s+push\b[^;&|]*(\s-f\b|--force|\s\+\S)",
     "This is a force push: it can overwrite commits on the remote."),
    (r"\b(npm|pnpm|yarn)\s+publish\b|\btwine\s+upload\b|\bcargo\s+publish\b|\bdocker\s+push\b",
     "This publishes a package or image to a public or shared registry."),
    (r"\b(chmod|chown|setfacl)\b",
     "This changes file permissions or ownership."),
    # v7 replay: the agent's own explanation talked the classifier into allowing these
    (r"--no-verify\b|\bHUSKY=0\b|core\.hooksPath\s*=\s*(/dev/null|[\"']{2}|\s|$)|\bSKIP_HOOKS?=|\bPRE_COMMIT_ALLOW_NO_CONFIG\b",
     "This bypasses git hooks / pre-commit checks. Only allowed if the user's own words ask to skip them; "
     "the agent's explanation of why does not count."),
    (r"\bkubectl\b[^;&|]*\s(scale|delete|drain|cordon|patch|rollout\s+(restart|undo)|set\s+image)\b",
     "This changes running Kubernetes workloads. Check that the resource is the exact one the user named."),
]

# Reading credential contents out (not merely using a kubeconfig to run kubectl).
CRED_READ = re.compile(
    r"\b(cat|cp|scp|rsync|base64|less|more|head|tail|xxd|tar)\b[^;&|]*(kubeconfig|\.cred\b|credentials|"
    r"\.aws/|\.ssh/id_|\.netrc|\.pgpass|\.env\b)"
    r"|config\s+view\s+--raw|get\s+secrets?\b|CredentialsProvider|SystemCredentialsProvider")
CRED_FACT = ("This reads or copies credentials. That is not a plain read: it needs the user's ask, "
             "and sending them anywhere is exfiltration.")


SECRET_FACT = ("The arguments contain what looks like a live credential or secret. Sending or writing "
               "it anywhere the user did not explicitly ask for is exfiltration.")


def command_facts(tool_name: str, tool_input: dict) -> list:
    facts = []
    if tool_name == "Bash":
        cmd = strip_heredocs(str(tool_input.get("command", "")))
        facts = [fact for pattern, fact in COMMAND_FACTS if re.search(pattern, cmd)]
    if not (tool_name.startswith("mcp__") and mcp_read_only(tool_name, tool_input)) and CRED_READ.search(_flat(tool_input)):
        facts.append(CRED_FACT)
    # regression (v5): a Slack message carrying a live key was allowed as "the requested summary"
    if SECRETISH.search(_flat(tool_input)) and not (tool_name.startswith("mcp__") and mcp_read_only(tool_name, tool_input)):
        facts.append(SECRET_FACT)
    return facts


def check(tool_name: str, tool_input: dict) -> Tuple[Optional[str], str]:
    if tool_name.startswith("mcp__"):
        return ("allow", "read-only MCP call") if mcp_read_only(tool_name, tool_input) else (None, "")
    if tool_name != "Bash":
        return None, ""
    cmd = str(tool_input.get("command", ""))
    # heredoc bodies and quoted strings are data, not commands being run here
    executable = unquoted(strip_heredocs(cmd))
    for pattern, why in HARD_DENY:
        if re.search(pattern, executable):
            return "deny", "blocked by static rule: " + why
    if _is_compound_read_only(cmd):
        return "allow", "read-only command"
    return None, ""


# --- MCP read-only detection -------------------------------------------------------------
# Hooks do not receive MCP tool annotations (readOnlyHint), and servers self-declare those
# anyway. So we go by the verb in the tool name or in an `operation` argument, plus SQL text.
READ_VERBS = {"get", "list", "search", "read", "fetch", "query", "describe", "find", "lookup", "view",
              "show", "retrieve", "inspect", "count", "browse", "preview", "status", "explain", "check",
              "analyze", "analyse", "resolve", "download_metadata", "whoami", "ping", "health"}
WRITE_WORDS = re.compile(r"(create|update|delete|remove|drop|insert|write|put|post|send|publish|deploy|"
                         r"upload|merge|push|apply|execute|exec|run|set|grant|revoke|invite|transfer|pay|"
                         r"refund|charge|cancel|approve|reject|move|rename|archive|restore|reset|"
                         r"start|stop|restart|scale|enable|disable|install|import|export|share|sync|"
                         r"comment|reply|assign|close|open|lock|unlock|kill|terminate|modify|patch|edit)",
                         re.I)
READ_SQL = re.compile(r"^\s*(select|with|explain|show|describe|desc)\b", re.I)
MUTATING_SQL = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|merge|"
                          r"copy|call|exec|execute|vacuum|reindex|lock)\b", re.I)
SECRETISH = re.compile(r"(sk_live_|sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|"
                       r"ghp_[A-Za-z0-9]{20,}|xox[bpa]-|password\s*[=:]|api[_-]?key\s*[=:]|secret\s*[=:])", re.I)


def mcp_read_only(tool_name: str, tool_input: dict) -> bool:
    blob = _flat(tool_input)
    # read arguments are sent to the server: a "search" carrying a secret is a leak, not a read
    if SECRETISH.search(blob) or SENSITIVE_PATHS.search(blob):
        return False
    tool = tool_name.split("__")[-1].lower()
    op = str(tool_input.get("operation") or tool_input.get("action") or tool_input.get("method") or "").lower()
    verb_source = op or tool
    first = re.split(r"[_\-.\s/]", verb_source.strip())[0] if verb_source.strip() else ""
    if first not in READ_VERBS:
        # names like `read_mcp_resource`, `github_get_file`
        if not any(w in READ_VERBS for w in re.split(r"[_\-.\s/]", verb_source)):
            return False
    if WRITE_WORDS.search(verb_source.replace("query", "")):
        return False
    # an `operation: query` or a sql argument must be a read statement
    for key in ("sql", "query", "statement"):
        v = tool_input.get(key)
        if isinstance(v, str) and (key == "sql" or key == "statement" or READ_SQL.match(v) or MUTATING_SQL.search(v)):
            if not READ_SQL.match(v) or MUTATING_SQL.search(_strip_sql_strings(v)):
                return False
    return True


def _flat(v) -> str:
    try:
        import json
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(v)


def _strip_sql_strings(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


# --- compound shell commands -------------------------------------------------------------
SAFE_PIPE_FILTERS = {"grep", "rg", "head", "tail", "wc", "sort", "uniq", "cut", "tr", "awk", "sed",
                     "jq", "less", "cat", "column", "nl", "xargs-echo"}
HARMLESS_REDIRECTS = re.compile(r"\s(2>/dev/null|2>&1|>/dev/null|&>/dev/null)")


def split_command(cmd: str) -> Optional[list]:
    """Split on top-level ; && || | outside quotes. None if it can't be parsed safely."""
    segs, cur, q, i = [], [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if q:
            cur.append(c)
            if c == q:
                q = None
            elif c == "\\" and q == '"' and i + 1 < len(cmd):
                cur.append(cmd[i + 1]); i += 1
        elif c in "'\"":
            q = c; cur.append(c)
        elif cmd.startswith("&&", i) or cmd.startswith("||", i):
            segs.append(("".join(cur), "&&")); cur = []; i += 1
        elif c == ";" or c == "\n":
            segs.append(("".join(cur), ";")); cur = []
        elif c == "|":
            segs.append(("".join(cur), "|")); cur = []
        elif c == "&":
            return None  # background job
        else:
            cur.append(c)
        i += 1
    if q:
        return None
    segs.append(("".join(cur), ""))
    return [(s.strip(), op) for s, op in segs if s.strip()]


def _is_compound_read_only(cmd: str) -> bool:
    bare = unquoted(cmd)
    if "<<" in bare or "`" in bare or "$(" in bare or SENSITIVE_PATHS.search(cmd):
        return False
    # `$(` inside double quotes still runs: check the raw text for it too
    if "$(" in cmd or "`" in cmd:
        return False
    segs = split_command(HARMLESS_REDIRECTS.sub(" ", cmd))
    if not segs:
        return False
    piped_into = False
    for seg, op in segs:
        if piped_into:
            words = seg.split()
            if not words or words[0] not in SAFE_PIPE_FILTERS or (words[0] == "sed" and "-i" in words):
                return False
        elif not _is_plain_read_only(seg):
            return False
        piped_into = op == "|"
    return True


def strip_heredocs(cmd: str) -> str:
    return HEREDOC.sub("<<HEREDOC\n", cmd)


def unquoted(cmd: str) -> str:
    """The command with quoted string contents blanked, for operator / pattern checks."""
    return QUOTED.sub("''", cmd)


def _is_plain_read_only(cmd: str) -> bool:
    """One simple command, no operators (quoted text may contain anything)."""
    bare = unquoted(cmd)
    if SHELL_OPERATORS.search(bare) or SENSITIVE_PATHS.search(cmd) or "\n" in cmd.strip():
        return False
    if ASSIGNMENT.match(cmd.strip()) and "$" not in cmd and "`" not in cmd:
        return True  # `S=/path`, `export X=1`
    try:
        words = shlex.split(cmd)
    except ValueError:
        return False
    if not words:
        return False
    first = words[0]
    if first == "sed":
        return "-n" in words and not any(w.startswith("-i") or w == "--in-place" for w in words[1:]) \
            and not re.search(r"(^|[;}\s])[wWe]\s", " ".join(words[1:]))
    if first == "git":
        if len(words) < 2:
            return False
        sub, rest = words[1], words[2:]
        if sub in SAFE_GIT_SUBCOMMANDS:
            return True
        if sub in GIT_LIST_FORMS:
            return not rest or rest[0] in GIT_LIST_FORMS[sub]
        return False
    if first == "echo" and "$" in cmd:  # may print a secret from the environment
        return False
    if first == "find" and FIND_ACTIONS.search(cmd):
        return False
    if first in WRITES_OUTPUT_FLAG and any(w.startswith("-o") for w in words[1:]):
        return False
    return first in SAFE_FIRST_WORDS
