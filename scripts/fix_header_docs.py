# CEP:FILE: scripts/fix_header_docs.py
# CEP:WHAT: Converts stray `///` continuation lines in file headers to plain
#           `//` comments so the `//!` module docs stay valid.
# CEP:WHY: The CEP:FILE header uses `//` comments; an accidental `///`
#          continuation makes the header an outer doc comment and rustc
#          rejects the following `//!` module docs (E0753).
# CEP:CLASS: CEP-2
# CEP:STATUS: complete
# CEP:FAILURE: exits nonzero if a file cannot be read.
# CEP:ASSUMES: headers precede the first item.
# CEP:COST: O(lines) per file.
# CEP:EVIDENCE: workspace compiles after runs.
# CEP:SECURITY: repository-trusted files only.

import sys

def fix(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    changed = 0
    for i, line in enumerate(lines):
        if line.startswith("use ") or line.startswith("pub "):
            break
        if line.startswith("///"):
            lines[i] = "//" + line[3:]
            changed += 1
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return changed

if __name__ == "__main__":
    total = 0
    for p in sys.argv[1:]:
        total += fix(p)
    print(f"fixed {total} header doc lines")
