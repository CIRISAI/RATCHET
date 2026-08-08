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
AGENT="${AGENT:-/tmp/a2911}"
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
cd "$AGENT/docker"
exec env \
  PROVIDER=deepinfra \
  BASE_URL=https://api.deepinfra.com/v1/openai \
  MODEL="$MODEL" \
  QUESTIONS_FILE=/questions/he300_tune.json \
  CONCURRENCY=2 \
  docker compose -f docker-compose.research.yml run --rm \
    -e OVERRIDES=/manifests/manifest.json \
    -e API_KEY_FILE=/keys/key \
    -v "$KEY:/keys/key:ro" \
    -v "$AGENT/docker/manifests:/manifests:ro" \
    capture
