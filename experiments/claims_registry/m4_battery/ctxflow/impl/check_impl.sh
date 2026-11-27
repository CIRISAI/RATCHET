#!/usr/bin/env bash
# TLC on the as-implemented model: every property separately under each configuration. Needs TLA2TOOLS=tla2tools.jar.
set -u; cd "$(dirname "$0")"; : "${TLA2TOOLS:?set TLA2TOOLS to tla2tools.jar}"
PROPS="${PROPS:-TypeOK Termination GuidanceTerminates DeferNeverBlocked WAOverrule WAOverruleInTask Continuity ForcedDeferDispatched}"
for cfg in ${CFGS:-v2232 F1_ban_loop F2_retry_forced_ponder F2_retry_forced_ponder_noguard F3_veto_defer F4_inherit F5_unsure fix}; do
  for p in $PROPS ${LIVE:-}; do
    W=$(mktemp -d); cp H3EREImpl.tla "$W/"; cp "$cfg.cfg" "$W/c.cfg"
    if [ "$p" = TerminatesLive ]; then echo "PROPERTY $p" >> "$W/c.cfg"; else echo "INVARIANT $p" >> "$W/c.cfg"; fi
    out=$(cd "$W" && java ${TLC_JAVA_OPTS:--Xmx1200m} -XX:+UseParallelGC -cp "$TLA2TOOLS" tlc2.TLC -config c.cfg -workers ${TLC_WORKERS:-1} H3EREImpl.tla 2>&1)
    if echo "$out" | grep -q "No error has been found"; then r=holds
    elif echo "$out" | grep -q "is violated\|Temporal properties were violated"; then r=VIOLATED
    else r=ERROR; echo "$out" | tail -20; fi
    n=$(echo "$out" | grep -o '[0-9]* distinct states found' | tail -1)
    len=$(echo "$out" | grep -cE '^State [0-9]+:')
    printf '%-32s %-22s %-9s %-28s %s\n' "$cfg" "$p" "$r" "$n" "$([ "$r" = VIOLATED ] && echo "trace=$len states")"
    [ "$r" = VIOLATED ] && [ -n "${TRACES:-}" ] && echo "$out" | sed -n '/is violated/,/distinct states found/p' > "$TRACES/${cfg}__${p}.txt"
    rm -rf "$W"
  done
done
