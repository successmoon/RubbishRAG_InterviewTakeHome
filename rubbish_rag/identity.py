"""Candidate identity: asked once, stamped on the README card + submission.

Stored in `.candidate.json` (repo root, packed into submission.zip).
The README report section is rendered from it by `rubbish bastesh`.
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDENT_PATH = os.path.join(BASE, ".candidate.json")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
GITHUB_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")


def load_identity():
    try:
        with open(IDENT_PATH, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("name") and EMAIL_RE.match(d.get("email", "")) \
                and GITHUB_RE.match(d.get("github", "")):
            return d
    except Exception:
        pass
    return None


def _ask(prompt, validate, err_hint):
    while True:
        try:
            v = input(prompt).strip()
        except EOFError:
            sys.exit("error: no input — run `python3 rubbish.py` interactively "
                     "once so we can stamp your report card.")
        if validate(v):
            return v
        print(err_hint)


def ensure_identity():
    ident = load_identity()
    if ident:
        return ident
    print("┌──────────────────────────────────────────────────┐")
    print("│  First run — who is taking this challenge?       │")
    print("│  This card goes on your README report + submission│")
    print("└──────────────────────────────────────────────────┘")
    name = _ask("  Full name: ",
                lambda v: len(v) >= 2, "  (please enter your name)")
    email = _ask("  Email: ",
                 lambda v: bool(EMAIL_RE.match(v)),
                 "  (that doesn't look like an email)")
    github = _ask("  GitHub username: ",
                  lambda v: bool(GITHUB_RE.match(v)),
                  "  (letters, digits, dashes — max 39 chars)")
    ident = {"name": name, "email": email, "github": github}
    with open(IDENT_PATH, "w", encoding="utf-8") as f:
        json.dump(ident, f, ensure_ascii=False, indent=1)
    print(f"  Saved ✓  — {name} (@{github})")
    return ident
