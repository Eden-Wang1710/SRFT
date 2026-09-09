"""Shared judge/pick helpers (used by generate_paraphrase.py, refilter.py, rescue_paraphrase.py)."""
from paraphrase import check
def quotes_injection(s, text):
    if s.situation != "injected" or not s.injection_snippet: return True
    q = s.injection_snippet.split(); o = text.lower()
    return any(" ".join(q[i:i+6]).lower() in o for i in range(max(1, len(q) - 5)))
def pick(ok, s=None):
    return min(ok, key=lambda x: (not quotes_injection(s, x["text"]) if s else False, x["metrics"].get("pasted_json", False), abs(x["metrics"]["len_ratio"] - 1.0), x["metrics"]["overlap8"]))
def judge_attempts(s, cands, round_id, tok):
    atts, ok = [], []
    for k, o in enumerate(cands):
        fails, m = check(s, o, tok); att = {"round": round_id, "k": k, "text": o.strip(), "fails": fails, "metrics": m}; atts.append(att)
        if not fails: ok.append(att)
    return atts, ok
