"""Project per-task wall time on the scorer's 4x L4 from a run's traces (forum-measured 31B L4 costs:
1.4 s per model request + 28.7 ms per generated token, thinking included) plus a per-tool-call allowance.
usage: project_l4.py RUN [RUN ...]   (run dirs under ~/g4/runs; sound public ids only)
Reports mean/p90 projected seconds, tasks over the 8-minute cap, resolved as run and resolved within the cap."""
import ast
import glob
import json
import os
import sys

PER_CALL, PER_TOK, PER_TOOL, CAP = 1.4, 0.0287, 1.0, 480
ids = open(os.path.expanduser("~/g4/runs/sound_public_ids.txt")).read().strip().split(",")


def lit(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return {}
    return v or {}


for run in sys.argv[1:]:
    d = os.path.expanduser(f"~/g4/runs/{run}")
    res = {}
    for f in glob.glob(f"{d}/results_shard*.jsonl"):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            res[r["id"]] = r
    rows = []
    for i in ids:
        tp = f"{d}/traces/trace_{i}.json"
        if i not in res or not os.path.exists(tp):
            continue
        t = json.load(open(tp))
        calls = toks = tools = 0
        for s in t["steps"]:
            if s.get("source") != "agent":
                continue
            m = lit(s.get("metrics"))
            calls += int(s.get("llm_call_count") or 1)
            toks += int(m.get("completion_tokens") or 0)
            tools += len(lit(s.get("tool_calls")) or [])
        proj = PER_CALL * calls + PER_TOK * toks + PER_TOOL * tools
        rows.append((i, res[i]["resolved"], proj, res[i].get("dur") or 0, calls, toks))
    if not rows:
        print(run, "no traces")
        continue
    n = len(rows)
    pj = sorted(r[2] for r in rows)
    print(f"{run:24s} n={n:3d} resolved {sum(r[1] for r in rows):3d} | within L4 cap {sum(r[1] and r[2] <= CAP for r in rows):3d} | "
          f"proj L4 mean {sum(pj) / n:5.0f}s p90 {pj[int(.9 * n) - 1]:4.0f}s over-cap {sum(p > CAP for p in pj):2d} | "
          f"H100 mean {sum(r[3] for r in rows) / n:4.0f}s | mean model calls {sum(r[4] for r in rows) / n:4.1f} tokens {sum(r[5] for r in rows) / n:5.0f} | "
          f"120 tasks at mean: {120 * sum(pj) / n / 3600:4.1f} h")
