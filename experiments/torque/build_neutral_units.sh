#!/usr/bin/env bash
# Assemble the neutral arm's unit corpora.
#
# Neutral shares the ALT arm's substituted sources — the neutral arm is neutral
# on MEANINGS, not on NAMES — and differs from alt only on the 9 authored SWAP
# lines. D-aspdma and E-exemplars declare zero SWAP, so alt and neutral are
# byte-identical there by construction, which is a property worth asserting
# rather than a coincidence to notice later.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p corpora/values-neutral
for u in B-optveto B-epihum B-coherence C-pdma D-aspdma E-exemplars F-lg-axiotic; do
  src="partition/src/${u}-sub.txt"; [ -f "$src" ] || src="partition/src/${u}.txt"
  sw="partition/${u}-neutral_swaps.tsv"
  [ -f "$sw" ] || sw="/dev/null"      # zero-SWAP units need no swaps file
  python3 partition.py assemble "$src" "partition/${u}.tsv" "$sw" \
      --out "corpora/values-neutral/${u}-mechanical.txt" >/dev/null
  python3 partition.py verify "$src" "partition/${u}.tsv" \
      "corpora/values-neutral/${u}-mechanical.txt" \
      ${sw:+--swaps "$sw"} 2>&1 | tail -1 | sed "s/^/${u}: /"
done

# ---------------------------------------------------------------------------
# Cross-arm assertion. Alt and neutral share a partition, so they must differ on
# EXACTLY the declared SWAP lines and nowhere else. Fewer means a neutral line
# was copied from the alt arm verbatim — which is how "Anthropic's Guidelines"
# ended up inside the value-NEUTRAL control on first pass, carrying a named
# external authority into the arm whose whole job is to name none.
echo
echo "alt vs neutral — must differ on exactly the declared SWAP lines:"
for u in B-optveto B-epihum B-coherence C-pdma D-aspdma E-exemplars F-lg-axiotic; do
  n=$(diff "corpora/values-alt/${u}-mechanical.txt" "corpora/values-neutral/${u}-mechanical.txt" | grep -c "^<" || true)
  s=$(awk -F'\t' '$2=="SWAP"' "partition/${u}.tsv" | wc -l)
  if [ "$n" = "$s" ]; then printf '  %-14s %2s/%-2s ok\n' "$u" "$n" "$s"
  else printf '  %-14s %2s/%-2s MISMATCH — a declared SWAP is identical across arms\n' "$u" "$n" "$s"; fi
done
