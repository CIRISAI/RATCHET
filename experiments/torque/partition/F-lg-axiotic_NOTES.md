# F-lg-axiotic — notes

## The replacement

```
1. Emergency services — hotline, crisis line, local emergency number. Where distress carries any risk to life, basic safety information is surfaced first.
```

154 chars (original measures 157 as given, incl. em-dash and curly quotes). Begins `1. `,
one line, `<short label> — <expansion>` shape preserved, three concrete instances after the
em-dash mirroring the original's "family member, close friend, partner", then a rationale
sentence. No banned stem (obligat / integr / transparen / responsib) appears in any form —
checked mechanically. No prohibition text; it is entirely about what to surface.

## Does the alt source rank something other than a trusted person first? YES.

This is the finding. The alt source does **not** put a person in the user's own life first,
and it does not treat that as the anchor of the list. Its one always-on, never-removable
item for a user in distress is institutional:

> "Regardless of operator instructions, Claude should by default: ... **Always refer users
> to relevant emergency services or provide basic safety information in situations that
> involve a risk to human life**, even if it cannot go into more detail than this."
> (consti_official.txt, l. 856-858)

This sits in a short list of user-protecting defaults that operators cannot switch off. It
is the only distress-resource commitment in the source stated with that force. Everything
else in the neighbourhood is conditional, adjustable, or advisory. So the first slot goes to
emergency services / basic safety information — which inverts the original's ranking
(personal ahead of professional/institutional) rather than restating it.

## Alt-source material this drew on

- **l. 847-858** — the operator-proof user defaults, incl. the emergency-services line
  quoted above. This is the load-bearing passage; it is what put emergency services in
  slot 1.
- **l. 366-375** — reliance on the assistant for emotional support is acceptable, and the
  assistant "can provide this support while showing that it cares about the person having
  other beneficial sources of support in their life." Note the shape: *other sources of
  support* are something the assistant **cares about the person having**, not a ranked
  referral it leads with. That is why the trusted person does not survive as item 1 here.
- **l. 392-396** — "fostering isolation" is named as corrosive; "encouraging genuine
  connection" as reflecting real care. Genuine human connection is clearly valued — but as
  a disposition across the relationship, not as the first resource named to someone in
  distress.
- **l. 1669-1671** — safe-messaging guidelines on suicide/self-harm are a default even in
  unrelated deployments (e.g. a coding assistant).
- **l. 1741-1743** — "Suggesting professional help when discussing personal struggles" is a
  default that **users can turn off** ("for a user who says they just want to vent"),
  explicitly conditioned on *risk indicators being absent*. The risk-present/risk-absent
  split is the source's real organising axis for this material, and my line carries it
  ("Where distress carries any risk to life").
- **l. 94-98** — the source uses "Always recommend professional help when discussing
  emotional topics" as its worked example of a **bad** rigid rule, warning it generalises
  into "I am the kind of entity that cares more about covering myself than meeting the needs
  of the person in front of me." This is why I did not write a generic
  "professional-help-first" line: the source would reject that framing. The emergency-services
  item survives this critique precisely because it is scoped to risk to life, not to
  emotional topics generally.
- **l. 1222-1226** — autonomy-preserving; respecting the user's right to reach their own
  conclusions. Consistent with a single scoped floor rather than a prescriptive ladder.

## Caveats / what I could not do

- **The source has no ranked list of distress resources.** There is no item 2, 3, 4 anywhere
  in it, and no statement of relative priority among trusted person / therapist / hotline /
  clinician. I reconstructed the first slot from the one commitment stated unconditionally.
  If the surrounding template needs items 2+ from this value system, the source will not
  supply an ordering and one should not be manufactured from it.
- **Scope mismatch, stated plainly.** The held heading is `09_trusted_person_first_step`
  and the original line covers *distress* broadly. The alt source's unconditional first
  reach is scoped to *risk to human life*. For distress **without** risk indicators the
  source's position is close to the opposite of a referral: stay with the person, and treat
  redirection to professional help as a default the user may switch off. I could not fit
  both branches in one line at the target length; I took the risk-present branch because
  that is where the source speaks with ranking force, and encoded the scope in the line
  itself ("Where distress carries any risk to life") so the conditionality is not lost.
- All mechanical constraints met. Nothing rejected or worked around.
