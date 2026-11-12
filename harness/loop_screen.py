"""Loop screen from traces: per task, calls, max repeats of one identical tool call, first edit step, empty patch.
A task is an abort when any identical call (same tool, same arguments) occurs >= 5 times.
usage: loop_screen.py RUN [RUN ...]"""
import ast
import collections
import glob
import json
import os
import sys

EDIT_MARKERS = ("apply.py", "cx.py replace", "cx.py insert", "cx.py blocks", "sed -i", "write_file", "edit_file")
for run in sys.argv[1:]:
    d = os.path.expanduser(f"~/g4/runs/{run}")
    res = {}
    for f in glob.glob(d + "/results_shard*.jsonl"):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            res[r["id"]] = r
    aborts = 0
    for tid in sorted(res):
        tp = f"{d}/traces/trace_{tid}.json"
        if not os.path.exists(tp):
            continue
        calls, first_edit, n = collections.Counter(), None, 0
        for s in json.load(open(tp))["steps"]:
            tcs = s.get("tool_calls")
            tcs = ast.literal_eval(tcs) if isinstance(tcs, str) else (tcs or [])
            for c in tcs:
                n += 1
                key = (c.get("function_name"), json.dumps(c.get("arguments"), sort_keys=True))
                calls[key] += 1
                if first_edit is None and any(m in key[0] + key[1] for m in EDIT_MARKERS):
                    first_edit = n
        rep = max(calls.values()) if calls else 0
        abort = rep >= 5
        aborts += abort
        r = res[tid]
        print(f"{run} {tid:45s} resolved={int(r['resolved'])} calls={n:3d} max_repeat={rep:2d} first_edit={first_edit} empty_patch={int((r.get('patch_chars') or 0) == 0)} {'ABORT' if abort else ''}")
    print(f"{run}: {aborts} aborts of {len(res)}")
