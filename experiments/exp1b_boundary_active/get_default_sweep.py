"""Find `.get(k, default)` whose result is immediately sliced/indexed/attributed.

dict.get returns the default only when the key is ABSENT. A key present with
value None returns None, and the very next operation explodes -- which is the
bug that killed qa_runner's _print_summary while it was reporting an error.
"""
import ast, sys
from pathlib import Path

class V(ast.NodeVisitor):
    def __init__(s, p): s.p, s.hits = p, []
    def _isget(s, n):
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and len(n.args) == 2)
    def visit_Subscript(s, n):
        if s._isget(n.value): s.hits.append((n.lineno, "subscript"))
        s.generic_visit(n)
    def visit_Attribute(s, n):
        if s._isget(n.value): s.hits.append((n.lineno, f".{n.attr}"))
        s.generic_visit(n)

tot = 0
for p in sorted(Path(".").rglob("*.py")):
    if any(x in p.parts for x in ("venv", ".git", "__pycache__", "build")): continue
    try: t = ast.parse(p.read_text())
    except Exception: continue
    v = V(p); v.visit(t)
    for ln, kind in v.hits:
        src = p.read_text().splitlines()[ln-1].strip()[:95]
        print(f"{p}:{ln}  [{kind}]  {src}"); tot += 1
print(f"\n{tot} sites where a .get() default is immediately dereferenced")
