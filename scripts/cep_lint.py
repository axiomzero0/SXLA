# CEP:FILE: scripts/cep_lint.py
# CEP:WHAT: Mechanical CEP&CC comment-field checker for first-party sources.
# CEP:WHY: CEP&CC 2.3 (psychopathic tier) demands mechanical enforcement:
#          every first-party file needs a CEP:FILE header with the required
#          fields (10.2), every TODO needs an owner and ticket (10.10), and
#          stub/placeholder statuses must be honest (10.5). This linter is
#          deterministic, offline, and fails CI on violations.
# CEP:CLASS: CEP-2
# CEP:STATUS: complete
# CEP:FAILURE: exits 1 when any violation is found; exits 2 on I/O errors.
# CEP:ASSUMES: files are UTF-8 repository sources.
# CEP:COST: O(lines) per file; <1s on the whole workspace.
# CEP:EVIDENCE: CI workflow runs this gate; .cep/lint-config.md documents
#          the rule set.
# CEP:SECURITY: repository-trusted files only; no network.

import re
import sys
from pathlib import Path

REQUIRED_FILE_FIELDS = [
    "CEP:FILE",
    "CEP:WHAT",
    "CEP:WHY",
    "CEP:CLASS",
    "CEP:STATUS",
    "CEP:FAILURE",
    "CEP:ASSUMES",
    "CEP:COST",
    "CEP:EVIDENCE",
]

VALID_STATUS = {"complete", "partial", "stub", "placeholder"}

SEVERITIES = {1: "S1 (must fix before merge)", 2: "S2 (blocker)"}


def lint_file(path: Path) -> list:
    violations = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [(path, 0, f"S2: unreadable file: {e}")]
    lines = text.split("\n")
    header = "\n".join(lines[: min(len(lines), 40)])

    # 1. Required file-header fields.
    for field in REQUIRED_FILE_FIELDS:
        if field + ":" not in header:
            violations.append((path, 1, f"S2: missing {field}: in file header"))

    # 2. Status validity where a file-level status exists.
    m = re.search(r"// CEP:STATUS:\s*(\w+)", header)
    if m and m.group(1) not in VALID_STATUS:
        violations.append(
            (path, 1, f"S2: invalid CEP:STATUS '{m.group(1)}' (must be one of {sorted(VALID_STATUS)})")
        )

    # 3. TODO MARKERS must carry an owner and a ticket (10.10); prose
    # mentions of the word "TODO" are not markers and are not flagged.
    for i, line in enumerate(lines, start=1):
        if re.search(r"//\s*TODO\b[(:]", line):
            if not re.search(r"CEP:TODO\([-\w]+\):\s*CEP-\d+", line):
                violations.append(
                    (path, i, "S2: TODO without owner/ticket (CEP:TODO(owner): CEP-n: ...)")
                )

    # 4. Unsafe code must be documented (mechanical complement to clippy).
    for i, line in enumerate(lines, start=1):
        if "unsafe {" in line or line.strip().startswith("unsafe impl"):
            window = "\n".join(lines[max(0, i - 14) : i])
            if "CEP:UNSAFE" not in window and "Safety" not in window:
                violations.append((path, i, "S1: undocumented unsafe block (missing CEP:UNSAFE | Safety comment)"))

    # 5. Anonymous TODO markers of the banned forms.
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if "fixme" in low or "xxx hack" in low:
            violations.append((path, i, "S2: banned marker (FIXME / hack)"))

    return violations


def main(argv: list) -> int:
    roots = argv[1:] or ["crates", "tools", "benches", "tests"]
    files = []
    for root in roots:
        base = Path(root)
        if base.is_file():
            files.append(base)
            continue
        if not base.exists():
            print(f"cep_lint: no such path {root}", file=sys.stderr)
            return 2
        files.extend(sorted(base.rglob("*.rs")))
    if not files:
        print("cep_lint: no Rust sources found", file=sys.stderr)
        return 2
    total = 0
    for f in files:
        for path, line, msg in lint_file(f):
            print(f"{path}:{line}: {msg}")
            total += 1
    print(f"cep_lint: checked {len(files)} files, {total} violation(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
