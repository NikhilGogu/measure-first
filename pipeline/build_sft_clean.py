"""Cleaned, packed SFT data: verified runs cut at their first successful submit_patch.

usage: [TOKENIZER=dir] build_sft_clean.py OUT.jsonl RUN_DIR [RUN_DIR ...]
Consecutive calls of a task are chained into one sequence with one loss span per call. Per task, on the last attempt of a
unanimously verified task:
- terminal call T = first call whose completion calls submit_patch and whose tool response reports patch_size > 0 and
  files_changed > 0; a task without one, or with any edit command after T, is excluded (its success may depend on it);
- calls after T are dropped, except call T+1 when it is text only (the closing summary, supervised with its own end of turn);
- a call whose tool calls all repeat a call made earlier in the same stretch without an edit in between keeps its tokens
  as context but gets no loss span.
Prints counts and writes boundary samples to OUT.boundaries.txt for inspection.
"""
import glob
import json
import os
import re
import sys

from transformers import AutoTokenizer

out_path, runs = sys.argv[1], sys.argv[2:]
tok = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER", os.path.expanduser("~/g4/models/gemma-4-31b-it-qat-w4a16-ct")))
TR = tok.convert_tokens_to_ids("<|tool_response>")
CALL_RE = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.S)
EDIT_MARKERS = ("apply.py", "cx.py replace", "cx.py insert", "cx.py blocks", "sed -i", "write_file", "edit_file")
stats = dict(tasks_verified=0, kept=0, no_successful_submit=0, edit_after_submit=0, calls_kept=0, calls_dropped_tail=0,
             calls_masked_repeat=0, closing_summaries=0, rows=0)
rows, samples = [], []


def calls_in(ids):
    return CALL_RE.findall(tok.decode(ids, skip_special_tokens=False))


def is_edit(sigs):
    return any(name in ("edit_file", "write_file") or any(m in args for m in EDIT_MARKERS) for name, args in sigs)


def submit_ok(resp_ids):
    text = tok.decode(resp_ids, skip_special_tokens=False)
    ps, fc = re.search(r"patch_size\D{0,8}(\d+)", text), re.search(r"files_changed\D{0,8}(\d+)", text)
    return bool(ps and fc and int(ps.group(1)) > 0 and int(fc.group(1)) > 0)


for run in runs:
    outcomes = {}
    for f in glob.glob(os.path.join(run, "results_shard*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                outcomes.setdefault(r["id"], set()).add(bool(r.get("resolved")))
    resolved = {t for t, o in outcomes.items() if o == {True}}
    calls = {}
    for f in glob.glob(os.path.join(run, "cap", "calls_*.jsonl")) + glob.glob(os.path.join(run, "calls.jsonl")):
        for l in open(f):
            if not l.strip():
                continue
            try:
                c = json.loads(l)
            except json.JSONDecodeError:  # a line cut short when a collector was stopped
                continue
            if c.get("status") == 200 and c.get("prompt_token_ids") and c.get("completion_token_ids") and c.get("task") in resolved:
                calls.setdefault(c["task"], []).append(c)
    for task, cs in calls.items():
        cs.sort(key=lambda c: c["ts"])
        starts = [i for i, c in enumerate(cs) if c["prompt_token_ids"] == cs[0]["prompt_token_ids"]]
        cs = cs[starts[-1]:]
        stats["tasks_verified"] += 1
        n = len(cs)
        sigs = [calls_in(c["completion_token_ids"]) for c in cs]
        # tool response of call i = tokens of call i+1's prompt after call i's re-rendered completion (when the chain extends)
        resp = [None] * n
        for i in range(n - 1):
            p, comp, nxt = cs[i]["prompt_token_ids"], cs[i]["completion_token_ids"], cs[i + 1]["prompt_token_ids"]
            if comp[-1:] == [TR] and len(nxt) > len(p) and nxt[:len(p)] == p and TR in nxt[len(p):]:
                resp[i] = nxt[nxt.index(TR, len(p)) + 1:]
        T = next((i for i in range(n) if any(nm == "submit_patch" for nm, _ in sigs[i]) and resp[i] is not None and submit_ok(resp[i])), None)
        if T is None:
            stats["no_successful_submit"] += 1
            continue
        if any(is_edit(sigs[i]) for i in range(T + 1, n)):
            stats["edit_after_submit"] += 1
            continue
        last = T + 1 if T + 1 < n and not sigs[T + 1] else T
        stats["closing_summaries"] += last == T + 1
        stats["calls_dropped_tail"] += n - 1 - last
        cs = cs[:last + 1]
        supervise, seen = [], set()
        for i, s in enumerate(sigs[:last + 1]):
            key = [(nm, a) for nm, a in s]
            repeat = bool(key) and all(k in seen for k in key)
            supervise.append(not repeat)
            stats["calls_masked_repeat"] += repeat
            seen.update(key)
            if is_edit(s):
                seen = set()
        if not supervise[T]:
            supervise[T] = True
            stats["calls_masked_repeat"] -= 1
        stats["kept"] += 1
        chain_spans = []
        for i, c in enumerate(cs):
            p, comp = c["prompt_token_ids"], c["completion_token_ids"]
            nxt = cs[i + 1]["prompt_token_ids"] if i + 1 < len(cs) else None
            if nxt is not None and comp[-1:] == [TR] and len(nxt) > len(p) and nxt[:len(p)] == p and TR in nxt[len(p):]:
                if supervise[i]:
                    chain_spans.append([len(p), nxt.index(TR, len(p)) + 1])
                continue
            if supervise[i]:
                chain_spans.append([len(p), len(p) + len(comp)])
            if chain_spans:
                rows.append({"input_ids": p + comp, "loss_spans": chain_spans, "task": task, "calls": len(chain_spans)})
                stats["calls_kept"] += len(chain_spans)
            chain_spans = []
        if len(samples) < 6:
            r = rows[-1]
            s0, e0 = r["loss_spans"][-1]
            samples.append(f"== {task}: T={T} last={last} of {n} calls; final supervised span:\n" + tok.decode(r["input_ids"][s0:e0], skip_special_tokens=False)[-600:])

with open(out_path, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
stats["rows"] = len(rows)
open(out_path + ".boundaries.txt", "w").write("\n\n".join(samples))
print(json.dumps(stats))
