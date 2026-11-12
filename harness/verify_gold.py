"""Check environment soundness: gold patch must pass, empty patch must fail, for every task. usage: verify_gold.py OUT [concurrency] [tasks.jsonl] [wheels_dir]"""
import asyncio, json, pathlib, sys, time
from adk_submission import ModelRegistry
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.models import load_tasks
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
conc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tasks_path = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else pathlib.Path(str(pathlib.Path.home() / "g4/data/tasks.jsonl"))
wheels = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else pathlib.Path(str(pathlib.Path.home() / "g4/data/wheels"))
snaps = pathlib.Path(str(pathlib.Path.home() / "g4/data/snapshots"))
limits, gen = build_submission_limits()
cfg = EvalConfig(tasks_path=tasks_path, snapshots_dir=snaps, results_dir=out, submission_dir=pathlib.Path(str(pathlib.Path.home() / "g4/data/sample_submission")),
                 models=ModelRegistry(), sandbox="docker", timeout_seconds=300, verification_timeout_seconds=300, max_time_minutes=1.0,
                 max_tool_calls=1, limits=limits, generation_constraints=gen, wheels_dir=wheels, verbose=False)
ev = Evaluator(cfg)
tasks = load_tasks(tasks_path)
sem = asyncio.Semaphore(conc)
res_path = out / "gold.jsonl"
done = {json.loads(l)["id"] + "|" + json.loads(l)["kind"] for l in res_path.read_text().splitlines()} if res_path.exists() else set()
async def one(t, kind):
    if f"{t.instance_id}|{kind}" in done:
        return
    async with sem:
        sp, bp, pp = resolve_task_snapshot_paths(snaps, t.instance_id, t.repo)
        t0 = time.time()
        try:
            r = await verify_task(ev.docker, cfg, t, sp, base_snapshot_path=bp, patch_path=pp, agent_patch=(t.patch if kind == "gold" else ""))
            rec = dict(id=t.instance_id, kind=kind, resolved=bool(r.resolved), exit=r.test_exit_code, err=str(r.error or "")[:400], s=round(time.time() - t0, 1))
        except Exception as e:
            rec = dict(id=t.instance_id, kind=kind, resolved=None, exit=None, err=f"EXC {e!r}"[:400], s=round(time.time() - t0, 1))
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
async def main():
    await asyncio.gather(*[one(t, k) for t in tasks for k in ("gold", "empty")])
asyncio.run(main())
recs = [json.loads(l) for l in res_path.read_text().splitlines()]
g = [r for r in recs if r["kind"] == "gold"]; e = [r for r in recs if r["kind"] == "empty"]
print("gold pass", sum(bool(r["resolved"]) for r in g), "/", len(g), "| empty pass", sum(bool(r["resolved"]) for r in e), "/", len(e))
