#!/usr/bin/env bash
# Tune the h3ere arms locally, in a container, against real HE-300 questions.
#
# WHY LOCAL AND WHY DOCKER. Iterating on a 90-minute CI job is hopeless — three
# arms have now died on the job cap and produced ZERO rows, because the battery
# writes its results only at the end. And iterating on the bare scratch checkout
# is what produced the TWO FEDERATION IDENTITIES chase: CIRISAgent mints its
# identity per occurrence, and a directory wiped 63 times has 126 of them.
#
# docker-compose.research.yml is already the right shape: container-local
# CIRIS_HOME, data dir, log dir and port, so a run cannot collide with anything
# on the host and starts from one identity every time. It takes OVERRIDES (with
# CIRIS_TESTING_MODE, validated by the AGENT's own loader before startup) and
# QUESTIONS_FILE. Nothing here needs inventing.
#
# WHAT THIS IS FOR: making an h3ere arm answer an HE question at all, and
# measuring how long one takes. It runs `model_eval`, one question at a time —
# NOT the 10-turn arc. The arc is the experiment; this is the instrument check
# that has to pass before the experiment is worth dispatching.
#
#   ./tune_local.sh h3ere-blank        # one arm, 8 questions, all 4 categories
#   ./tune_local.sh h3ere-alt 4        # 4 questions
set -euo pipefail
cd "$(dirname "$0")"

ARM="${1:-h3ere-ciris}"
N="${2:-8}"
MODULE="${MODULE:-safety_battery}"   # battery writes results.jsonl; model_eval does not
STRATUM="${STRATUM:-axiotic_primary}"
# A PRISTINE CLONE, not the working checkout.
#
# Dockerfile.research does `COPY . /app` and .dockerignore excludes data/ but
# NOT identity/. The scratch checkout has been wiped ~63 times and holds 126
# accumulated bootstrap identities, so `--build` baked all of them into the
# image and the container inherited the exact federation-identity failure it
# exists to escape: the Engine and the compose process disagree about which is
# the node's, node_fold refuses, and the API server never binds port 8080.
#
# The first container runs worked because they used a PRE-EXISTING image built
# from a clean state. Rebuilding from a dirty tree is what broke it — the same
# lesson as "run it in CI", which clones fresh every time.
SRC="${SRC:-/tmp/a2911}"
AGENT="${AGENT:-/tmp/torque-agent}"
AGENT_REF="${AGENT_REF:-v2.9.11-stable}"
if [ ! -d "$AGENT/.git" ]; then
  echo "cloning $AGENT_REF into $AGENT (once)…"
  git clone --quiet --depth 1 --branch "$AGENT_REF" "$SRC" "$AGENT" 2>/dev/null \
    || git clone --quiet --depth 1 --branch "$AGENT_REF" \
         https://github.com/CIRISAI/CIRISAgent.git "$AGENT"
fi
# Never let host state into the build context, whatever the clone picked up.
rm -rf "$AGENT/identity" "$AGENT/data" "$AGENT/logs"

# The MODULE knob, applied to the clone. capture_traces.sh ships hardcoded to
# model_eval, which answers one question at a time and writes no results.jsonl —
# it proves the agent runs and cannot be scored. This is a real gap in the
# research tooling and belongs upstream; applied here so the local loop works
# now.
python3 - "$AGENT" <<'PATCH'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / "tools/research/capture_traces.sh"
t = p.read_text()
if "MODULE=" in t:
    sys.exit(0)
old = 'ARGS=(model_eval --live'
new = ('MODULE="${MODULE:-model_eval}"\n'
       'if [ "$MODULE" = "safety_battery" ]; then\n'
       '  ARGS=(safety_battery --live --live-key-file "$KEYFILE" --live-model "$MODEL"\n'
       '        --live-base-url "$BASE_URL" --live-provider openai\n'
       '        --safety-battery-lang "${LANGUAGES%%,*}"\n'
       '        --safety-battery-domain "${BATTERY_DOMAIN:?BATTERY_DOMAIN required}"\n'
       '        --safety-battery-template "${BATTERY_TEMPLATE:-he-300-benchmark}" --verbose)\n'
       'else\n'
       '  ARGS=(model_eval --live')
assert old in t, "capture_traces.sh shape changed"
t = t.replace(old, new, 1)
t = t.replace('[ -n "$QUESTIONS_FILE" ] && ARGS+=(--model-eval-questions-file "$QUESTIONS_FILE")',
              '  [ -n "$QUESTIONS_FILE" ] && ARGS+=(--model-eval-questions-file "$QUESTIONS_FILE")\nfi', 1)
p.write_text(t)
print("  MODULE knob applied to the clone")
PATCH
KEY="${KEY:-$HOME/.deepinfra_key}"
MODEL="${MODEL:-meta-llama/Llama-4-Scout-17B-16E-Instruct}"

[ -f "arms/$ARM.json" ] || { echo "no manifest: arms/$ARM.json"; exit 2; }

# Real questions, drawn from the same builder the campaign uses, so a pass here
# is a pass on the material that will run — not on a toy.
python3 - "$N" <<'PY'
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location("bh", "build_he300_arcs.py")
bh = importlib.util.module_from_spec(spec); spec.loader.exec_module(bh)
n = int(sys.argv[1])
out, per = [], max(1, n // 4)
for stratum in ("axiotic_primary", "axiotic_secondary", "deontic_held", "discriminant_control"):
    for it in bh.load_items(stratum)[:per]:
        out.append({
            "category": it["category"],
            "question": f"{bh.QUESTION[it['category']]}\n\n{it['text']}",
            "evaluates": f"concordance; gold={it['gold']}; {it['item_id']}",
            "translations": {"en": f"{bh.QUESTION[it['category']]}\n\n{it['text']}"},
        })
d = pathlib.Path("/tmp/a2911/docker/research-questions")
d.mkdir(parents=True, exist_ok=True)
(d / "he300_tune.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"wrote {len(out)} questions ({per} per stratum)")
PY

# One real arc, built by the campaign's own builder, mounted into the container.
# The battery reads cells from tests/safety inside the image, so the arc has to
# be baked in via a bind mount rather than passed as a file.
# Built INTO THE BUILD CONTEXT, not bind-mounted.
#
# Dockerfile.research does `COPY . /app` from the agent root, and the agent runs
# `verify_manifest_integrity` over its own file tree at startup. A directory
# bind-mounted into /app after the image was built is a file the integrity
# registry has never seen:
#     verify_manifest_integrity: HASH MISMATCH!
#     check_full: manifest integrity verification FAILED
# — the agent's own tamper detection, working, on a tamper I introduced. Writing
# the arcs into the checkout before `--build` puts them in the manifest instead.
python3 build_he300_arcs.py --n-arcs 1 --turns 10 \
  --seed 42 --stratum "$STRATUM" \
  --ethics /home/emoore/CIRISBench/engine/datasets/ethics \
  --safety-dir "$AGENT/tests/safety" >/dev/null
CELL="english_he300_${STRATUM}_a00"
DOMAIN="he300_${STRATUM}_a00"

mkdir -p "$AGENT/docker/manifests"
cp "arms/$ARM.json" "$AGENT/docker/manifests/manifest.json"

echo "── $ARM ─────────────────────────────────────────────"
# -e OVERRIDES, not an exported shell var: compose forwards only variables it
# names in its own `environment:` block, and OVERRIDES is not one of them. The
# first run set it in the parent shell, the container never saw it, and the arm
# ran with NO MANIFEST while reporting success — an unmodified agent wearing an
# arm's name, which is the most dangerous failure this campaign can have.
#
# And keep every assignment on the continuation chain: a comment line between
# `VAR=x \` and the command breaks the continuation, so the whole env prefix
# silently detaches and compose falls back to its openrouter defaults. That is
# how the second run got "HTTP 401 from openrouter" while being handed a
# deepinfra key.
# EVERY knob goes through -e, none through the environment prefix.
#
# Compose forwards only the variables its own `environment:` block names. OVERRIDES
# was not one, so the first run executed an unmodified agent wearing an arm's name
# and reported success. MODULE and BATTERY_DOMAIN are not either, so the next run
# silently ran model_eval instead of the battery — the wrong instrument, reporting
# its own failures as though they were the arm's.
#
# Same bug twice in one afternoon. Passing everything explicitly makes it
# structurally impossible rather than remembered.
cd "$AGENT/docker"
# --build every time. Dockerfile.research does `COPY . /app` at BUILD time, so
# an edit to capture_traces.sh on the host is invisible to a container started
# from a stale image — the MODULE knob was added, passed correctly, and ignored,
# because the image still held the pre-patch script hardcoding model_eval.
exec docker compose -f docker-compose.research.yml run --rm --build \
  -e PROVIDER=deepinfra \
  -e BASE_URL=https://api.deepinfra.com/v1/openai \
  -e MODEL="$MODEL" \
  -e MODULE="$MODULE" \
  -e BATTERY_DOMAIN="$DOMAIN" \
  -e BATTERY_TEMPLATE=he-300-benchmark \
  -e LANGUAGES=en \
  -e CONCURRENCY=1 \
  -e OVERRIDES=/manifests/manifest.json \
  -e API_KEY_FILE=/keys/key \
  -v "$KEY:/keys/key:ro" \
  -v "$AGENT/docker/manifests:/manifests:ro" \
  capture
