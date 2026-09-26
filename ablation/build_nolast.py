#!/usr/bin/env python3
"""ABL-B data for Qwen3-8B: toucan_32B_v2 with the LAST paragraph of every <think> removed (docs/01 §ABL-B / §ABL-Q8-NOLAST).

Paragraph 3 of the reflection is where the sampled candidate actions enter the supervision (expert-action rationale +
consequences of the hijacked candidates); paragraphs 1-2 (goal, injection identification) stay. Nothing else changes:
same trajectories, same turns (the 3,185 think-only turns keep their think), same answers.

    python ablation/build_nolast.py   ->  LLaMA-Factory/data/toucan_32B_v2_nolast.json (+ .report.json)
"""
import json, re, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "LLaMA-Factory/data/toucan_32B_v2.json"
DST = ROOT / "LLaMA-Factory/data/toucan_32B_v2_nolast.json"
PAT = re.compile(r"^(<think>\s*)(.*?)(\s*</think>)(.*)$", re.S)
CAND = re.compile(r"candidate|sub-?optimal|alternative action|other action", re.I)


def main():
    data = json.loads(SRC.read_text())
    st = collections.Counter(); words_before = words_after = 0; cand_before = cand_after = 0
    for traj in data:
        for m in traj["conversations"]:
            if m["from"] not in ("gpt", "function_call") or "<think>" not in m["value"]:
                continue
            g = PAT.match(m["value"]); assert g, m["value"][:200]
            head, think, tail, rest = g.groups()
            paras = [p for p in re.split(r"\n\s*\n", think) if p.strip()]
            st[f"paras={len(paras)}"] += 1
            assert len(paras) >= 2, think[:200]
            kept = paras[:-1]
            words_before += len(think.split()); words_after += sum(len(p.split()) for p in kept)
            cand_before += bool(CAND.search(think)); cand_after += bool(CAND.search("\n\n".join(kept)))
            m["value"] = head + "\n\n".join(kept) + tail + rest
            st["think_msgs"] += 1
    DST.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    rep = {"source": SRC.name, "trajectories": len(data), **st,
           "think_words_before": words_before, "think_words_after": words_after,
           "think_words_removed_pct": round(100 * (1 - words_after / words_before), 1),
           "think_msgs_mentioning_candidates_before": cand_before, "after": cand_after}
    DST.with_suffix(".report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
