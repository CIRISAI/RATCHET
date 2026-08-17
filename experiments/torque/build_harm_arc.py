#!/usr/bin/env python3
"""Emit HARM-1 items as v4 arcs so the SAME harness runs every arm.

Run in CI against a pinned agent checkout: `build_harm_arc.py <agent-root>`.
Items are frozen in `harm_items/` with digests in its MANIFEST.json, so the
arcs regenerate byte-identically from the repository alone — the earlier
version read them out of a scratchpad, which is not a provenance anyone else
can check.

Reusing qa_runner rather than calling the agent API directly keeps the pipeline
arm identical to MH-3's in every respect but the questions — same template, same
conscience path, same capture format.

Split into 6-turn arcs: these items are INDEPENDENT, unlike the mental-health
battery whose staged escalation is the instrument. Long arcs would add a
position effect for no benefit, and TORQUE already paid to learn that.
"""
import json,pathlib,sys
SP=pathlib.Path(__file__).resolve().parent / "harm_items"
AGENT=pathlib.Path(sys.argv[1])
tmpl=json.loads((AGENT/"tests/safety/english_mental_health/v4_english_mental_health_arc.json").read_text())
LET="ABCD"

def emit(name, questions):
    arc=json.loads(json.dumps(tmpl))          # deep copy of the shipped shape
    arc["battery_id"]=name
    arc["cell"]=dict(arc["cell"], domain=name, language="en")
    arc.pop("criteria_path",None); arc.pop("criteria_sha256",None)
    arc.pop("rubric_path",None);   arc.pop("rubric_sha256",None)
    arc["questions"]=questions
    d=AGENT/"tests/safety"/f"english_{name}"
    d.mkdir(parents=True,exist_ok=True)
    (d/f"v4_english_{name}_arc.json").write_text(json.dumps(arc,ensure_ascii=False,indent=1))
    return d

def q(qid,stage,text,extra):
    return {"question_id":qid,"question_version":1,"stage":stage,"category":"harm",
            "evaluates":"declines hazardous content while remaining helpful",
            "hard_fail_triggers":[],"soft_fail_triggers":[],"faculty_targets":[],
            "harm":extra,"translations":{"en":text}}

wmdp=json.loads((SP/"wmdp_bio.json").read_text())
xs=json.loads((SP/"xstest.json").read_text())
items=[]
for i,w in enumerate(wmdp):
    t=w["question"]+"\n"+"\n".join(f"{LET[j]}. {c}" for j,c in enumerate(w["choices"]))
    items.append(("wmdp",{"bench":"wmdp-bio","idx":i,"gold":w["answer"]},t))
for i,x in enumerate(xs):
    items.append(("xstest",{"bench":"xstest","idx":i,"label":x["label"],"type":x.get("type")},x["prompt"]))

n=0
for start in range(0,len(items),6):
    chunk=items[start:start+6]
    qs=[q(f"harm_a{start//6:02d}_t{j+1:02d}",f"Turn {j+1} of {len(chunk)}",txt,extra)
        for j,(_b,extra,txt) in enumerate(chunk)]
    emit(f"harm_a{start//6:02d}",qs); n+=1
print(f"wrote {n} arcs, {len(items)} items total")
