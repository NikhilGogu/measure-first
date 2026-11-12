"""Run swegemma tasks against an already running vLLM server, mirroring the official notebook EvalConfig.
usage: run_eval.py --sub DIR --out DIR [--tasks tasks.jsonl] [--ids a,b] [--shard k --nshards M] [--sandbox docker]"""
import argparse, asyncio, json, os, pathlib, time
import yaml
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True"); os.environ.setdefault("OTEL_SDK_DISABLED", "true")
import litellm
litellm.drop_params = True
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.models.lite_llm import LiteLlm
from adk_submission import ModelRegistry, discover_adapters
from adk_submission.discovery import discover_declared_models
from swegemma.config import EvalConfig, build_submission_limits, ALLOWED_ADAPTER_EXTENSIONS
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

ap = argparse.ArgumentParser()
ap.add_argument("--sub", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--tasks", default=str(pathlib.Path.home() / "g4/data/tasks.jsonl")); ap.add_argument("--snapshots", default=str(pathlib.Path.home() / "g4/data/snapshots"))
ap.add_argument("--wheels", default=str(pathlib.Path.home() / "g4/data/wheels_fixed")); ap.add_argument("--ids", default="")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--sandbox", default="docker"); ap.add_argument("--api", default="http://127.0.0.1:8000/v1"); ap.add_argument("--tag-file", default=""); ap.add_argument("--time-scale", type=float, default=1.0)
a = ap.parse_args()
sub = pathlib.Path(a.sub); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
server = json.loads(pathlib.Path(str(pathlib.Path.home() / "g4/runs/server.json")).read_text())
api = a.api
models = ModelRegistry()
for m in discover_declared_models(sub):
    models.register(m, LiteLlm(model=f"openai/{server["served"]}", api_base=api, api_key="EMPTY", num_retries=3))
adapters = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
for name in (getattr(adapters, "adapters", None) or {}):
    for key in (name, f"adapter:{name}"):
        models.register(key, LiteLlm(model=f"openai/{name}", api_base=api, api_key="EMPTY", num_retries=3))
ec = next((c for n in ("eval_config.yaml", "eval_config.yml") if (c := sub / n).is_file()), None)
raw = (yaml.safe_load(ec.read_text()) or {}) if ec else {}
ev = raw.get("evaluation", raw)
turns = ev.get("max_turns", ev.get("max_llm_calls"))
limits, gen = build_submission_limits()
cfg = EvalConfig(tasks_path=pathlib.Path(a.tasks), snapshots_dir=pathlib.Path(a.snapshots), results_dir=out, submission_dir=sub,
                 models=models, sandbox=a.sandbox, timeout_seconds=int(ev.get("timeout_seconds", 300)),
                 verification_timeout_seconds=int(ev.get("verification_timeout_seconds", 300)),
                 max_time_minutes=float(ev.get("max_time_minutes", 60.0)) * a.time_scale, max_tool_calls=int(ev.get("max_tool_calls", 100)),
                 max_turns=int(turns) if turns is not None else None, limits=limits, generation_constraints=gen, adapter_manifest=adapters,
                 context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                 events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
                 graph_dir=str(pathlib.Path.home() / "g4/data/graphs"), embeddings_dir=str(pathlib.Path.home() / "g4/data/embeddings"), wheels_dir=pathlib.Path(a.wheels), verbose=False)
ev_ = Evaluator(cfg)
tasks = load_tasks(pathlib.Path(a.tasks))
if a.ids:
    want = set(a.ids.split(",")); tasks = [t for t in tasks if t.instance_id in want]
tasks = [t for i, t in enumerate(tasks) if i % a.nshards == a.shard]
done = set()
res_path = out / f"results_shard{a.shard}.jsonl"
if res_path.exists():
    done = {json.loads(l)["id"] for l in res_path.read_text().splitlines() if l.strip()}

async def main():
    for i, t in enumerate(tasks, 1):
        if t.instance_id in done:
            continue
        t0 = time.time()
        if a.tag_file:
            pathlib.Path(a.tag_file).write_text(t.instance_id)
        try:
            r = await ev_.evaluate_task(task=t, task_index=i, total_tasks=len(tasks))
            (out / "patches").mkdir(exist_ok=True); (out / "patches" / f"{t.instance_id}.patch").write_text(r.agent_patch or ""); (out / "test_outputs").mkdir(exist_ok=True); (out / "test_outputs" / f"{t.instance_id}.log").write_text((r.test_output or "")[-20000:])
            rec = dict(id=t.instance_id, resolved=bool(r.resolved), exit=r.test_exit_code, patch_chars=len((r.agent_patch or "").strip()),
                       tool_calls=r.tool_calls, dur=round(r.duration_seconds, 1), wall=round(time.time() - t0, 1), err=str(getattr(r, "error", "") or "")[:300])
        except Exception as e:
            rec = dict(id=t.instance_id, resolved=False, exit=None, patch_chars=0, tool_calls=None, dur=None, wall=round(time.time() - t0, 1), err=f"EXC {e!r}"[:300])
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

asyncio.run(main())
