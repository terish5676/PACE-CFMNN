"""Check that every `python scripts/X.py ...` command in README.md exists.

For each command: the script file exists, `--help` runs, every `--flag` it uses is
in the help text, and every value given to a flag with fixed choices is one of them.
Read-only. Run from the repository root:
    python -B research_audit/scripts/check_readme_commands.py
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8")


def commands(text: str) -> list[str]:
    return sorted(set(re.findall(r"`(python (?:-m unittest[^`]*|scripts/[^`]+))`", text)))


def main() -> int:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    cmds = commands(text)
    helps: dict[str, str] = {}
    bad = 0
    for cmd in cmds:
        parts = shlex.split(cmd)
        if parts[1] == "-m":
            print(f"OK    {cmd}  (unittest; run separately)")
            continue
        script = parts[1]
        problems = []
        if not (ROOT / script).exists():
            problems.append("script missing")
        elif "argparse" not in (ROOT / script).read_text(encoding="utf-8"):
            # No parser: `--help` would run the whole script and rewrite its outputs. Check only
            # that the README passes it no flags.
            if any(p.startswith("-") for p in parts[2:]):
                problems.append("flags given to a script with no argument parser")
        else:
            if script not in helps:
                r = subprocess.run([sys.executable, "-B", script, "--help"], cwd=ROOT, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=300)
                helps[script] = r.stdout if r.returncode == 0 else ""
                if r.returncode != 0:
                    problems.append("--help failed: " + r.stderr.strip().splitlines()[-1])
            h = helps[script]
            for i, p in enumerate(parts[2:], start=2):
                if p.startswith("--"):
                    if p not in h:
                        problems.append(f"flag {p} not in --help")
                        continue
                    m = re.search(re.escape(p) + r" \{([^}]*)\}", h)
                    if m and i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                        if parts[i + 1] not in m.group(1).split(","):
                            problems.append(f"{p} {parts[i + 1]} not in choices {{{m.group(1)}}}")
        bad += bool(problems)
        print(("FAIL  " if problems else "OK    ") + cmd + ("  -> " + "; ".join(problems) if problems else ""))
    print(f"\n{len(cmds)} commands, {bad} with problems")
    return int(bad > 0)


if __name__ == "__main__":
    raise SystemExit(main())
