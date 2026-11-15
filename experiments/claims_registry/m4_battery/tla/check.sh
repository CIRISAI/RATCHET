#!/usr/bin/env bash
# Model-check M4Episode.tla: every invariant under both verdict rules, and termination.
# Usage: TLA2TOOLS=/path/to/tla2tools.jar ./check.sh   (tla2tools 1.8.x; Java 11+)
set -u
cd "$(dirname "$0")"
JAR="${TLA2TOOLS:?set TLA2TOOLS to tla2tools.jar}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
INVARIANTS="TypeOK Bounded NoContaminatedVerdict NoFalseUptake NoMissedUptake PersistSound PersistComplete
            NoUnexplainedNo NoUnexplainedNotPersisted SpilloverReported"
run() {  # rules kind name
  local cfg="$WORK/$1-$3.cfg"
  printf 'CONSTANTS MaxAttempts = 3\n          PauseBudget = 3\n          Rules = "%s"\nSPECIFICATION Spec\n%s %s\n' \
    "$1" "$2" "$3" > "$cfg"
  local out
  out=$(java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -workers 1 -metadir "$WORK/meta-$1-$3" -config "$cfg" \
        M4Episode.tla 2>&1)
  local res
  res=$(echo "$out" | grep -oE 'No error has been found|Invariant [A-Za-z]+ is violated|Temporal properties were violated|Error: .*' | head -1)
  local st
  st=$(echo "$out" | grep -oE '[0-9]+ distinct states found' | head -1)
  printf '%-8s %-26s %s (%s)\n' "$1" "$3" "${res:-?}" "${st:-?}"
  if [ "${VERBOSE:-}" = 1 ] && ! echo "$out" | grep -q 'No error has been found'; then echo "$out" | tail -40; fi
}
for rules in fixed current; do
  for inv in $INVARIANTS; do run "$rules" INVARIANT "$inv"; done
  run "$rules" PROPERTY Terminates
done
