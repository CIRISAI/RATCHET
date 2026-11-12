"""The kindscope language (.ks): goals, stakes, misfits, diagnoses — parsed, emitted and checked.

A file is a sequence of blocks. A block starts at column 0 with `<kind> <ID>`; its fields follow as indented
`key: value` lines. `#` starts a comment. Kinds:

    goal G1
      line: map where parts of speech are carried in GoldWorm v6 base
      nouns: pos10, pos-ovr:*
      verbs: probe

    stake S0001
      goal: G1
      verb: probe                      # grammar: VERBS
      noun: pos-ovr:NOUN               # grammar: NOUNS
      at: pre/mlp2                     # <site>/<state>
      predict: carried = true          # <outcome field> <op> <value>
      from: S-V1-pre/mlp2              # bar provenance: a stake id | prior-art:<cite> | plant:<id>
      cites: M-PLANT-OBS, ...          # filled by the auditor from the registry, never by hand

    misfit KS-DIRECT-ONLY
      line: ...                        # one line
      rule: ...                        # what every later stake must do
      applies: verb in {attribute}     # predicate over stake fields: which stakes the rule binds
      requires: downstream             # engine obligations the bound stake must meet (comma list), or `cite`
      signature: verb=probe & V1=true & V2=false   # predicate over a result record: which misses it explains
      born: CAL-1 2026-11-11
      status: registered               # candidate | registered | refuted | adopted

    diagnosis D0001
      for: S0012, S0013                # the open misses it explains
      misfit: KS-LEXICAL               # new id (becomes a candidate) or an existing one
      line: / rule: / applies: / requires: / signature:   (for a new id)
      test: probe pos-ovr:ADP at last/attn0 predict V3 = false   # a discriminating stake; the misfit is
                                       # registered only if the test HITS, refuted (kept visible) if it misses

Predicates: clauses joined by `&`; a clause is `field op value` with op in = != < > <= >= in; `in {a,b}` takes a set.
Values: true/false, numbers, bare words. State order for comparisons: embed < attn0 < mlp0 < ... < mlp5.
"""

import re

STATES = ["embed"] + [f"{p}{i}" for i in range(6) for p in ("attn", "mlp")]
SITES = ["pre", "first", "last"]
VERBS = ["probe"]  # v1; amnesic/patch/steer/fork-delta join when each has its own plant
POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX", "CCONJ", "PROPN"]
NOUNS = ["pos10"] + [f"pos-ovr:{c}" for c in POS]
OUTCOME_FIELDS = ["carried", "V1", "V2", "V3", "acc", "acc_novel", "diff_novel"]


class LangError(ValueError):
    pass


def parse(text):
    blocks, cur = [], None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line[0].isspace():
            m = re.match(r"^(goal|stake|misfit|diagnosis)\s+(\S+)$", line)
            if not m:
                raise LangError(f"line {n}: expected `<goal|stake|misfit|diagnosis> <ID>`, got {line!r}")
            cur = {"kind": m.group(1), "id": m.group(2)}
            blocks.append(cur)
        else:
            if cur is None:
                raise LangError(f"line {n}: field outside a block")
            m = re.match(r"^\s+([a-z_]+):\s*(.*)$", line)
            if not m:
                raise LangError(f"line {n}: expected `key: value`, got {line!r}")
            cur[m.group(1)] = m.group(2).strip()
    return blocks


def emit(block):
    out = [f"{block['kind']} {block['id']}"]
    for k, v in block.items():
        if k not in ("kind", "id"):
            out.append(f"  {k}: {v}")
    return "\n".join(out) + "\n"


def _val(s):
    s = s.strip()
    if s in ("true", "false"):
        return s == "true"
    try:
        return float(s)
    except ValueError:
        return s


def _cmp_key(field, v):
    if field == "state" and isinstance(v, str) and v in STATES:
        return STATES.index(v)
    return v


def predicate(expr):
    """Compile `a=1 & b in {x,y} & state<mlp2` into a function of a dict. Unknown fields make a clause false."""
    clauses = []
    for c in [c.strip() for c in expr.split("&") if c.strip()]:
        m = re.match(r"^([A-Za-z_0-9]+)\s*(in|!=|<=|>=|=|<|>)\s*(.+)$", c)
        if not m:
            raise LangError(f"bad clause {c!r}")
        f, op, rhs = m.groups()
        if op == "in":
            s = rhs.strip()
            if not (s.startswith("{") and s.endswith("}")):
                raise LangError(f"`in` needs a set: {c!r}")
            rhs = {_val(x) for x in s[1:-1].split(",") if x.strip()}
        else:
            rhs = _val(rhs)
        clauses.append((f, op, rhs))

    def fn(rec):
        for f, op, rhs in clauses:
            if f not in rec:
                return False
            a = rec[f]
            if op == "in":
                if a not in rhs:
                    return False
                continue
            a, b = _cmp_key(f, a), _cmp_key(f, rhs)
            try:
                ok = {"=": a == b, "!=": a != b, "<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]
            except TypeError:
                return False
            if not ok:
                return False
        return True

    fn.expr = expr
    return fn


def stake_line(s):
    """`probe <noun> at <site>/<state> predict <field> <op> <value> [from <src>]` -> stake fields."""
    m = re.match(r"^(\w+)\s+(\S+)\s+at\s+(\w+)/(\w+)\s+predict\s+(\w+)\s*(=|!=|<=|>=|<|>)\s*(\S+)(?:\s+from\s+(\S+))?$",
                 s.strip())
    if not m:
        raise LangError(f"bad stake line {s!r}")
    verb, noun, site, state, f, op, v, src = m.groups()
    return {"verb": verb, "noun": noun, "at": f"{site}/{state}", "predict": f"{f} {op} {v}", "from": src or ""}


def check_stake(b):
    """Grammar check of a stake block; returns the normalised record or raises."""
    for k in ("verb", "noun", "at", "predict", "from"):
        if not b.get(k):
            raise LangError(f"stake {b.get('id')}: missing `{k}`")
    if b["verb"] not in VERBS:
        raise LangError(f"stake {b['id']}: verb {b['verb']!r} not in grammar {VERBS}")
    if b["noun"] not in NOUNS:
        raise LangError(f"stake {b['id']}: noun {b['noun']!r} not in grammar")
    site, _, state = b["at"].partition("/")
    if site not in SITES or state not in STATES:
        raise LangError(f"stake {b['id']}: at {b['at']!r} not <site>/<state>")
    m = re.match(r"^(\w+)\s*(=|!=|<=|>=|<|>)\s*(\S+)$", b["predict"])
    if not m or m.group(1) not in OUTCOME_FIELDS:
        raise LangError(f"stake {b['id']}: predict {b['predict']!r} must be `<{'|'.join(OUTCOME_FIELDS)}> op value`")
    if not re.match(r"^(S[-\w/]+|D\d+|prior-art:\S+|plant:\S+)$", b["from"]):
        raise LangError(f"stake {b['id']}: from {b['from']!r} must be a stake, a diagnosis, prior-art:<cite> or plant:<id>")
    if b.get("split", "explore") not in ("explore", "confirm"):
        raise LangError(f"stake {b['id']}: split must be explore or confirm")
    return {**b, "site": site, "state": state, "layer": STATES.index(state), "split": b.get("split", "explore")}
