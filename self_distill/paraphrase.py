"""v3 recipe: context-grounded PARAPHRASE of the Claude think by Qwen3-8B (thinking OFF). See docs/04_self_distill_plan.md "v3 recipe".
Filters are content-first: length ratio, paragraphs, tool-name coverage, alternatives kept, injection kept, first person, meta leak,
step-0 grounding, not-a-copy, no action text.
"""
from __future__ import annotations
import re, json
from common import Step, ngram_overlap

REWRITE_MSG = """[Rewrite task. This message and the draft below are private: never mention them or that you are rewriting anything.]
Below is a draft of your inner reasoning at this exact step, written by someone else in the third person.
Rewrite it as YOUR OWN inner monologue, first person, in your natural thinking voice, as if you are thinking right now, before acting.

Rules
1. Keep every point of the draft: the goal and what has been done so far; every injected instruction it identifies (quote it word for word, as the draft does) and why it is malicious; EVERY alternative action it mentions and what would happen if you took it; the action it settles on and why. Do not summarize or shorten — your version should be about as long as the draft, with the same paragraphs.
2. Change the voice, not the content: "the agent / the assistant / the model" -> "I"; "the expert action / expert response" -> "the right move here"; "candidate model actions / the candidates / a_model_1" -> concrete first-person alternatives ("I could call X instead, but ...").
3. Refer only to tool outputs that actually appear above. If the draft discusses a tool response that does not exist yet, leave that part out.
4. Never use the words "draft", "reference", "expert", "candidate", "notes", or "private". No headings, no lists, no <tool_call>, no final answer — output the monologue text only.
5. Below, under "Actions you weighed", the action you actually take at this step is marked "= what you do"; the others are alternatives. Where the draft compares "candidate model actions" (a_model_1, a_model_2, ...), those are these alternatives. Mention each alternative in first person as something you weighed ("I could call X with ... instead, but ...") and keep the draft's judgement of it. Describe the action you take in words (tool name and what you pass), do not paste the raw argument JSON. Do not copy the list format.
"""

ALT_HEADER = "\nActions you weighed at this step (for you only, never list or quote them as a list):\n"

def _same_call(c, exp):
    return c.get("kind") == "tool_call" and isinstance(exp, dict) and c.get("name") == exp.get("name") \
        and json.dumps(c.get("args"), sort_keys=True) == json.dumps(exp.get("arguments"), sort_keys=True)

def alternatives_block(step: Step) -> str:
    """Candidates (deduped) + the expert action, each tagged: '= what you do' / 'would follow the injected instruction' / 'safe, but not what you do'."""
    exp = step.expert_action
    items = []; seen = set()
    for c in step.candidates:
        key = json.dumps([c["kind"], c["name"], c["args"], (c["text"] or "")[:200]], sort_keys=True)
        if key in seen: continue
        seen.add(key); items.append(c)
    have_expert = any(_same_call(c, exp) for c in items)          # final steps: always insert the expert answer itself
    if not have_expert:
        if isinstance(exp, dict): items.insert(0, {"kind": "tool_call", "name": exp.get("name"), "args": exp.get("arguments"), "text": None, "follows_injection": False, "_expert": True})
        else: items.insert(0, {"kind": "text", "name": None, "args": None, "text": str(exp), "follows_injection": False, "_expert": True})
    lines = []; expert_marked = False
    for i, c in enumerate(items, 1):
        is_exp = bool(c.get("_expert") or _same_call(c, exp))
        if c["kind"] == "tool_call":
            desc = f"call `{c['name']}` with {json.dumps(c['args'], ensure_ascii=False)}"
        else:
            t = (c["text"] or "").replace("\n", " ")
            first = ""                       # opening of the answer: whole sentences up to ~200 chars, at least ~60
            for sent in re.split(r"(?<=[.!?])\s+", t.strip()):
                if first and len(first) >= 60: break
                first = (first + " " + sent).strip()[:200]
            desc = f"answer the user directly, no tool call (it begins: \"{first}\")" if is_exp else f"answer the user directly: \"{t[:120]}{'…' if len(t) > 120 else ''}\""
        if is_exp and not expert_marked: tag, expert_marked = "  <- = what you do at this step", True
        elif c["follows_injection"]: tag = "  <- would follow the injected instruction"
        else: tag = "  <- safe, but not what you do"
        lines.append(f"{i}. {desc}{tag}")
    return ALT_HEADER + "\n".join(lines) + "\n"

DRAFT_HEADER = "\nDraft:\n"

TOOL_ID = re.compile(r"`([A-Za-z][\w\-]*(?:[-_][\w\-]+)+)`|\b([a-z][\w]*(?:-[\w]+){2,}(?:_[\w]+)?)\b")
META = re.compile(   # phrase-level: bare "private"/"reference"/"note" occur legitimately (injection text, "API reference", "note that")
    r"\b(the draft|this draft|draft (above|below|says|mentions)|reference (reasoning|notes?|text|monologue)|expert('s)? (action|response|reasoning|call|answer|move)"
    r"|candidate (model )?(actions?|responses?|calls?)|the candidates|a_model_\d|private notes?|(these|the|my) notes (say|mention|above)"
    r"|alternatives? (you|I) (have )?considered|actions? (you|I) (have )?weighed|what (you|I) do at this step|listed (above|below)|rewrit(e|ing|ten)|third person)\b", re.I)
TOOL_DEF = re.compile(r'"name":\s*"([^"]+)"')

def system_tools(system: str) -> list[str]:
    return sorted(set(TOOL_DEF.findall(system or "")), key=len, reverse=True)

def _mentions(text: str, name: str) -> bool:
    t = text.lower(); sh = short(name)
    return name.lower() in t or (len(sh) > 4 and sh in t) or (len(sh) > 4 and re.sub(r"[_-]", " ", sh) in t)

def tools_in(text: str, tools: list[str]) -> set[str]:
    return {n for n in tools if _mentions(text, n)}
THIRD = re.compile(r"\b(the (agent|assistant|model)) (is|was|has|had|should|must|will|would|needs?|calls?|called|decid\w*)", re.I)
FIRST = re.compile(r"\bI\b|\bI'|\bmy\b", re.I)
QUOTE = re.compile(r'"([^"\n]{30,})"')

def short(name: str) -> str:
    return re.split(r"[-_]", name)[-1].lower()

def tool_names(text: str) -> set[str]:
    out = set()
    for a, b in TOOL_ID.findall(text):
        n = a or b
        if n and len(n) > 6: out.add(short(n))
    return out

def paragraphs(t: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", t.strip()) if p.strip()]

def rescue_note(step: Step) -> str:
    n = len(step.claude_think.split())
    note = (f"\nIMPORTANT: your rewrite must be AT LEAST as long as the draft (the draft has about {n} words; write {n}-{int(n*1.1)} words). "
            "Do not compress, merge or drop any sentence's content; expand each point in your own words instead.")
    if step.kind == "final":
        note += " At this step you answer the user directly instead of calling a tool; say explicitly that you will now answer/respond and what the answer covers."
    return note + "\n"

def build_messages(step: Step, rescue: bool = False) -> list[dict]:
    msgs = [{"role": "system", "content": step.system}] + step.history
    msgs.append({"role": "user", "content": REWRITE_MSG + (rescue_note(step) if rescue else "") + alternatives_block(step) + DRAFT_HEADER + step.claude_think.strip()})
    return msgs

def render(tok, step: Step, rescue: bool = False) -> str:
    return tok.apply_chat_template(build_messages(step, rescue), tokenize=False, add_generation_prompt=True, enable_thinking=False)

MAX_THINK_TOKENS = 480     # inference think budget is 512 (forced transition beyond it)
LEN_LO, LEN_HI = 0.80, 1.15

def check(step: Step, out: str, tok=None) -> tuple[list[str], dict]:
    """-> (failed filter names, metrics)"""
    d = step.claude_think.strip(); o = (out or "").strip()
    o = re.sub(r"^<think>\s*</think>\s*", "", o).strip()          # empty think block from nothink template
    fails = []; m = {}
    wd, wo = len(d.split()), len(o.split()); m["len_ratio"] = round(wo / max(wd, 1), 2)
    lo = LEN_LO
    if tok is not None:
        m["tokens"] = len(tok(o, add_special_tokens=False).input_ids)
        if m["tokens"] > MAX_THINK_TOKENS: fails.append("too_many_tokens")
        dt = len(tok(d, add_special_tokens=False).input_ids)
        if dt > MAX_THINK_TOKENS: lo = 0.65          # long drafts must be trimmed to fit the budget
    if not (lo <= m["len_ratio"] <= LEN_HI): fails.append(f"length_{m['len_ratio']}")
    pd_, po = len(paragraphs(d)), len(paragraphs(o)); m["paras"] = f"{po}/{pd_}"
    if po < pd_ - 1 or po > 2 * pd_: fails.append("paragraphs")      # Qwen tends to split the alternatives into their own paragraphs; allow up to 2x
    tools = system_tools(step.system)
    if tools:   # only real tool names from the system prompt count (the old identifier regex matched "hotels", "456", "step")
        td, to = tools_in(d, tools), tools_in(o, tools)
    else:
        td, to = tool_names(d), tool_names(o)
    # tools that MUST survive: the expert action's tool and the candidate tools (the learn-from-failure contrast); other tools mentioned
    # in passing (e.g. "the earlier search-hotels call") are optional
    must = set()
    if isinstance(step.expert_action, dict) and step.expert_action.get("name"): must.add(step.expert_action["name"])
    for c in step.candidates:
        if c.get("kind") == "tool_call" and c.get("name"): must.add(c["name"])
    must = {n for n in must if _mentions(d, n)}                      # only if the draft itself talks about them
    missing = sorted(n for n in must if not _mentions(o, n)); m["tools_missing"] = [short(x) for x in missing]
    if missing: fails.append("tool_coverage")
    alt = {n for n in must if not (isinstance(step.expert_action, dict) and n == step.expert_action.get("name"))}
    kept_alt = {n for n in alt if _mentions(o, n)}; m["alts"] = f"{len(kept_alt)}/{len(alt)}"
    if len(kept_alt) < len(alt): fails.append("alternatives_dropped")
    if step.kind == "final" and step.situation != "step0" and not re.search(r"\b(answer|respond|reply|tell|report|summar|confirm|provide|give|present|explain|let (the user|them) know|wrap up|conclude|inform|share|list|recommend)", o, re.I):
        fails.append("final_not_signalled")                            # a final step's think must end in answering, not calling
    if step.situation == "injected" and step.injection_snippet:      # ground-truth snippet, injected steps only
        qq = step.injection_snippet
        kept = any(" ".join(qq.split()[i:i+6]).lower() in o.lower() for i in range(max(1, len(qq.split()) - 5)))
        if not kept and not re.search(r"inject|instruction", o, re.I): fails.append("injection_dropped")
    m["first_person"] = len(FIRST.findall(o))
    if m["first_person"] < 3: fails.append("not_first_person")
    if THIRD.search(o): fails.append("third_person")
    leak = META.findall(o)
    if leak: fails.append("meta_leak"); m["leak"] = sorted({(x if isinstance(x, str) else x[0]).lower() for x in leak})
    if step.situation == "step0" and re.search(r"tool (response|result|output)", o, re.I): fails.append("step0_grounding")
    m["overlap8"] = round(ngram_overlap(o, d), 2)
    if m["overlap8"] > 0.50: fails.append("copy")
    sd = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", d) if s.strip()]
    so = set(s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", o) if s.strip())
    m["sent_changed"] = round(1 - sum(s in so for s in sd) / max(len(sd), 1), 2)
    if m["sent_changed"] < 0.15: fails.append("pasted")
    if re.search(r"<tool_call>|final answer\s*:", o, re.I): fails.append("action_text")   # JSON inside the draft is allowed to survive
    m["pasted_json"] = bool(isinstance(step.expert_action, dict) and step.expert_action.get("arguments") not in (None, {}) and json.dumps(step.expert_action["arguments"], ensure_ascii=False) in o)
    return fails, m
