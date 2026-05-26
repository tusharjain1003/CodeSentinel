BOILERPLATE = {"thanks", "thank you", "lgtm", "looks good"}


def is_valid_comment(comment: dict) -> bool:
    body = (comment.get("body") or "").strip()
    lowered = body.lower()
    if len(body) < 15 or len(body) > 2000:
        return False
    if lowered in BOILERPLATE or lowered.startswith("lgtm"):
        return False
    if comment.get("user", {}).get("type") == "Bot":
        return False

    actionable_signals = [
        "should", "could", "consider", "instead",
        "bug", "issue", "error", "null", "crash",
        "race condition", "overflow", "injection", "vulnerability",
        "refactor", "rename", "extract", "missing",
        "fix", "wrong", "incorrect", "problem", "broken",
        "improve", "suggest", "need", "must", "check",
        "verify", "ensure", "prevent", "avoid", "handle",
        "add", "remove", "change", "update", "don't",
        "doesn't", "won't", "never", "always",
        "leak", "deadlock", "corrupt", "loss",
        "unsafe", "deprecated", "typo", "duplicate",
    ]
    return any(signal in lowered for signal in actionable_signals)


def clean_diff_hunk(hunk: str) -> str:
    lines = hunk.splitlines()
    clean_lines = [
        line
        for line in lines
        if not line.startswith("diff --git")
        and not line.startswith("index ")
        and not line.startswith("Binary")
    ]
    return "\n".join(clean_lines)[:3000]
