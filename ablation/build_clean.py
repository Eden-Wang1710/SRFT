#!/usr/bin/env python3
"""aba-cleanSFT data: the SR-Agent-Llama training trajectories with the prompt injection REMOVED and no reflection.

  in : LLaMA-Factory/data/toucan_32B_v3_base.json                      (meta.injections[*].combined = the injected string)
       LLaMA-Factory/data/toucan_32B_v3_base_llama_local_nothink.json  (ABL-A data: same trajectories, Llama `local` tool
                                                                        format, every <think> block already stripped)
  out: LLaMA-Factory/data/toucan_32B_v3_base_llama_local_clean_nothink.json (+ .report.json)

The two inputs are index-aligned (asserted on the first human turn). Each trajectory carries exactly one injection, inserted
post hoc into tool observations in one of three ways (measured 2026-11-10 on all 5,852 occurrences):
    end of the observation   : <obs> + "\n" + INJ          (2,335)
    start of the observation : INJ + "\n" + <obs>          (1,799)
    inside the observation   : <left> + " " + INJ + " " + <right>   (1,716)
    the whole observation    : INJ                         (2)
so removing INJ together with that separator restores the original TOUCAN observation. 210 trajectories never received their
injection (meta.steps[*].has_injection_inserted_by_step all False); they are already clean and are copied unchanged.
Human turns, assistant tool calls and final answers are untouched: the expert actions were recorded on the clean
trajectory, so this is plain agentic SFT on expert demonstrations -- no injection, no reflection.
usage: python ablation/build_clean.py        (any env, CPU, ~1 min)
"""
import collections, hashlib, json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LLaMA-Factory", "data")
META = os.path.join(ROOT, "toucan_32B_v3_base.json")
SRC = os.path.join(ROOT, "toucan_32B_v3_base_llama_local_nothink.json")
OUT = os.path.join(ROOT, "toucan_32B_v3_base_llama_local_clean_nothink.json")


def strip_injection(text, inj, stat):
    """Remove every occurrence of `inj` from one observation together with the separator it was inserted with."""
    while True:
        i = text.find(inj)
        if i < 0:
            return text
        j = i + len(inj)
        if i == 0 and j == len(text):
            stat["whole"] += 1; text = ""
        elif j == len(text) and text[i - 1] == "\n":
            stat["end"] += 1; text = text[: i - 1]
        elif i == 0 and text[j] == "\n":
            stat["start"] += 1; text = text[j + 1:]
        elif text[i - 1] == " " and text[j] == " ":
            stat["inside"] += 1; text = text[: i - 1] + text[j + 1:]
        else:
            raise ValueError(f"unexpected insertion boundary: {text[max(0, i - 20):i]!r} | {text[j:j + 20]!r}")


def main():
    meta, src = json.load(open(META)), json.load(open(SRC))
    assert len(meta) == len(src) == 3698, (len(meta), len(src))
    stat, traj = collections.Counter(), collections.Counter()
    out = []
    for m, s in zip(meta, src):
        assert m["conversations"][0]["value"] == s["conversations"][0]["value"], "inputs are not index-aligned"
        injs = list(m["meta"]["injections"].values())
        assert len(injs) == 1
        inj = injs[0]
        conv, changed = [], 0
        for turn in s["conversations"]:
            v = turn["value"]
            if turn["from"] == "observation" and inj["combined"] in v:
                v = strip_injection(v, inj["combined"], stat); changed += 1
            conv.append({"from": turn["from"], "value": v})
        flagged = any(st.get("has_injection_inserted_by_step") for st in m["meta"]["steps"].values())
        assert flagged == (changed > 0), m["meta"]["trajectory"]
        traj["injection removed" if changed else "already clean (injection never inserted)"] += 1
        # nothing of the injection may survive in what the model reads from tools, and nothing else may change
        for a, b in zip(s["conversations"], conv):
            if a["from"] == "observation":
                assert inj["combined"] not in b["value"] and inj["trigger"] not in b["value"] and inj["task"] not in b["value"]
            else:
                assert a["value"] == b["value"]
            assert "<think>" not in b["value"]
        out.append({"conversations": conv, "system": s["system"]})
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    roles = collections.Counter(t["from"] for e in out for t in e["conversations"])
    report = {
        "out": os.path.basename(OUT), "sha256": hashlib.sha256(open(OUT, "rb").read()).hexdigest(),
        "trajectories": len(out), "turns_by_role": dict(roles),
        "trajectories_by_kind": dict(traj), "occurrences_removed_by_position": dict(stat),
        "assistant_turns_changed": 0, "human_turns_changed": 0,
    }
    json.dump(report, open(OUT + ".report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
