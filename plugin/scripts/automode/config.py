"""Configuration, later sources win:

1. defaults below
2. ~/.claude-automode/config.json
3. the plugin's options in Claude Code's /config (passed to hooks as CLAUDE_PLUGIN_OPTION_<KEY>);
   empty options are ignored
4. AUTOMODE_* environment variables
"""
import json
import os
from dataclasses import dataclass, field

STATE_DIR = os.path.expanduser(os.environ.get("AUTOMODE_STATE_DIR", "~/.claude-automode"))


@dataclass
class Config:
    provider: str = "anthropic"          # "anthropic" (Messages API) or "openai" (chat completions)
    base_url: str = "https://api.anthropic.com"
    api_key: str = ""
    fast_model: str = ""                 # stage 1; empty -> skip stage 1, go straight to stage 2
    slow_model: str = ""                 # stage 2; required
    # Extra request-body fields per stage, merged into the API call. Reasoning models need this:
    # e.g. DeepSeek spends a tiny stage-1 budget on thinking unless {"thinking": {"type": "disabled"}}.
    fast_extra: dict = field(default_factory=dict)
    slow_extra: dict = field(default_factory=dict)
    slow_max_tokens: int = 4096          # room for reasoning tokens before the JSON verdict
    timeout_s: float = 30.0
    max_consecutive_denials: int = 3     # after this many denials in a row, hand the call to the human
    user_message_limit: int = 12         # how many recent user messages the classifier sees
    log_inputs: bool = True              # write tool inputs (truncated) to the decision log


ENV_MAP = {
    "provider": "AUTOMODE_PROVIDER",
    "base_url": "AUTOMODE_BASE_URL",
    "api_key": "AUTOMODE_API_KEY",
    "fast_model": "AUTOMODE_FAST_MODEL",
    "slow_model": "AUTOMODE_SLOW_MODEL",
    "fast_extra": "AUTOMODE_FAST_EXTRA",      # JSON object
    "slow_extra": "AUTOMODE_SLOW_EXTRA",      # JSON object
    "slow_max_tokens": "AUTOMODE_SLOW_MAX_TOKENS",
    "timeout_s": "AUTOMODE_TIMEOUT_S",
    "max_consecutive_denials": "AUTOMODE_MAX_DENIALS",
    "user_message_limit": "AUTOMODE_USER_MESSAGES",
    "log_inputs": "AUTOMODE_LOG_INPUTS",
}


# Fields settable from /config (plugin.json userConfig). disable_thinking is not a Config field:
# it expands into fast_extra / slow_extra.
PLUGIN_OPTIONS = ("provider", "base_url", "api_key", "fast_model", "slow_model")
NO_THINKING = {"thinking": {"type": "disabled"}}


def load() -> Config:
    cfg = Config()
    path = os.path.join(STATE_DIR, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    for name in PLUGIN_OPTIONS:
        v = os.environ.get("CLAUDE_PLUGIN_OPTION_" + name.upper(), "").strip()
        if v:
            setattr(cfg, name, v)
    if os.environ.get("CLAUDE_PLUGIN_OPTION_DISABLE_THINKING", "").lower() in ("1", "true", "yes"):
        cfg.fast_extra = {**cfg.fast_extra, **NO_THINKING}
        cfg.slow_extra = {**cfg.slow_extra, **NO_THINKING}
    for field, env in ENV_MAP.items():
        if env in os.environ:
            raw = os.environ[env]
            cur = getattr(cfg, field)
            if isinstance(cur, dict):
                setattr(cfg, field, json.loads(raw))
            elif isinstance(cur, bool):
                setattr(cfg, field, raw.lower() in ("1", "true", "yes"))
            elif isinstance(cur, int):
                setattr(cfg, field, int(raw))
            elif isinstance(cur, float):
                setattr(cfg, field, float(raw))
            else:
                setattr(cfg, field, raw)
    return cfg
