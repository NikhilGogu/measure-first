"""Run the scorer's agent loop (swegemma 0.2.11, unchanged) on SWE-rebench tasks, or check their soundness.

Only repository preparation and Phase 2 verification are replaced (simplified):
- each task runs in a container from its own SWE-rebench image through swegemma's ContainerManager
  (same 4 GiB / 2 vCPU / no-network limits); /testbed is moved to /workspace and git history is rebuilt
  at base_commit so the fix cannot leak;
- verification applies the agent patch (protected test/config paths stripped), resets the test-patch
  files, applies the test patch and requires every FAIL_TO_PASS and PASS_TO_PASS node to be PASSED in
  pytest -rA output.

usage:
  rebench_eval.py --tasks rebench.jsonl --out DIR --sub BUNDLE [--shard k --nshards M] [--api URL] [--tag-file F]
  rebench_eval.py --tasks rebench.jsonl --out DIR --gold          # gold must pass, empty must fail
"""
import argparse
import asyncio
import base64
import json
import os
import pathlib
import re
import shlex
import time
import uuid

import yaml

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
import litellm

litellm.drop_params = True
from adk_submission import ModelRegistry, discover_adapters
from adk_submission.discovery import discover_declared_models
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.models.lite_llm import LiteLlm

import swegemma.evaluate as ev
import swegemma.harness.agent_runner as ar
import swegemma.harness.container_setup as cs
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
from swegemma.harness.verification import save_trace_artifact, strip_protected_files_from_patch
from swegemma.models import Task, TaskResult

ENVPATH = "/opt/conda/envs/testbed/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
RECORDS: dict[str, dict] = {}
VERIFIED_AT: dict[str, float] = {}  # epoch time each task's verification outcome was fixed (deadline filter of the self-trained round)
SENTINEL = pathlib.Path(__file__)


def patch_paths(diff: str) -> list[str]:
    out = set()
    for m in re.finditer(r"^(?:---|\+\+\+) (?:a/|b/)?(\S+)", diff or "", re.M):
        p = m.group(1)
        if p != "/dev/null" and not p.startswith("/") and ".." not in p.split("/"):
            out.add(p)
    return sorted(out)


def run(docker, cid, cmd, timeout=300):
    r = docker.exec(cid, cmd, timeout=timeout)
    return r.exit_code, (r.stdout or ""), (r.stderr or "")


def put_file(docker, cid, path, text):
    b64 = base64.b64encode(text.encode()).decode()
    chunks = [b64[i:i + 60000] for i in range(0, len(b64), 60000)] or [""]
    run(docker, cid, f"rm -f {path}.b64")
    for c in chunks:
        run(docker, cid, f"printf %s {shlex.quote(c)} >> {path}.b64")
    rc, out, err = run(docker, cid, f"base64 -d {path}.b64 > {path} && rm -f {path}.b64")
    if rc:
        raise RuntimeError(f"put_file failed: {err or out}")


def prepare_repo(docker, cid, *args, **kwargs):
    """Replaces swegemma's snapshot extraction: /testbed becomes /workspace, git rebuilt at base_commit."""
    rec = RECORDS[getattr(docker, "_rb_task", "")]
    base = shlex.quote(rec["base_commit"])
    script = (
        "set -e; cd /; "
        "if [ -d /workspace ] && [ ! -L /workspace ] && [ -d /testbed ] && [ ! -L /testbed ]; then rm -rf /workspace; fi; "
        "if [ ! -L /testbed ]; then mv /testbed /workspace && ln -s /workspace /testbed; fi; "
        "cd /workspace; git config --global --add safe.directory '*'; "
        "git config --global user.email agent@eval; git config --global user.name Agent; "
        f"if [ \"$(git rev-parse HEAD)\" != {base} ]; then git checkout -q -f {base}; fi; "
        f"git reset -q --hard {base}; git ls-files -z > /tmp/rb_tracked; "
        "rm -rf .git && git init -q && git config core.hooksPath /dev/null && "
        "xargs -0 -r git add -f -- < /tmp/rb_tracked; rm -f /tmp/rb_tracked"
    )
    rc, out, err = run(docker, cid, script, timeout=600)
    if rc:
        raise RuntimeError(f"prepare_repo failed: {(err or out)[-800:]}")


def _noop(*args, **kwargs):
    return None


def apply_patch(docker, cid, text, name):
    if not text.strip():
        return 0, ""
    put_file(docker, cid, f"/tmp/{name}.patch", text.rstrip("\n") + "\n")
    p = f"/tmp/{name}.patch"
    rc, out, err = run(docker, cid, f"cd /workspace && (git apply --whitespace=nowarn {p} || git apply -3 {p} || "
                                    f"git apply --ignore-whitespace --recount {p} || patch -p1 --batch --forward -i {p})")
    return rc, (out + err)[-2000:]


def parse_passed(log: str) -> dict[str, str]:
    status = {}
    for line in re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log).splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S.*?)(?: - .*)?$", line)
        if m:
            node = m.group(2).strip()
            if status.get(node) != "FAILED":
                status[node] = m.group(1)
    return status


def verify_sync(docker, config, task, agent_patch, start_time, trace=None, agent_error=None):
    rec = RECORDS[task.instance_id]
    log, rc, ok, error = "", -1, False, agent_error
    cid = None
    try:
        cid = docker.start()
        prepare_repo(docker, cid)
        cs.setup_git_exclude(docker, cid)
        cs.setup_baseline_commit(docker, cid, "eval_baseline")
        test_files = patch_paths(rec["test_patch"])
        clean = strip_protected_files_from_patch(agent_patch or "", test_files)
        arc, aout = apply_patch(docker, cid, clean, "agent")
        if arc:
            raise RuntimeError(f"agent patch failed to apply: {aout}")
        for p in test_files:
            q = shlex.quote(p)
            run(docker, cid, f"cd /workspace && (git cat-file -e HEAD:{q} 2>/dev/null && git checkout -q HEAD -- {q} || rm -f -- {q})")
        trc, tout = apply_patch(docker, cid, rec["test_patch"], "test")
        if trc:
            raise RuntimeError(f"test patch failed to apply: {tout}")
        required = list(rec["FAIL_TO_PASS"]) + list(rec["PASS_TO_PASS"])
        targets = sorted({n.split("::", 1)[0] for n in required if "::" in n} |
                         {p for p in test_files if p.endswith(".py") and not p.endswith("conftest.py")})
        cmd = rec["install_config"]["test_cmd"]
        full = f"cd /workspace && {cmd} {' '.join(shlex.quote(t) for t in targets)} > /tmp/rb_tests.log 2>&1; echo $?"
        rcx, out, _ = run(docker, cid, full, timeout=900)
        rc = int(out.strip().splitlines()[-1]) if out.strip() else -1
        _, log, _ = run(docker, cid, "tail -c 400000 /tmp/rb_tests.log")
        st = parse_passed(log)
        missing = [n for n in required if st.get(n) != "PASSED"]
        ok = rc in (0, 1) and not missing
        if not ok:
            error = error or f"{len(missing)} required tests not PASSED (exit {rc}); first: {missing[:3]}"
    except Exception as e:
        error = f"Verification error: {e}"[:2000]
    finally:
        VERIFIED_AT[task.instance_id] = time.time()
        if cid is not None:
            try:
                docker.stop(cid)
            except Exception:
                pass
    json_path = save_trace_artifact(trace, config.results_dir, task.instance_id) if trace is not None else None
    return TaskResult(instance_id=task.instance_id, repo=task.repo, resolved=ok, agent_patch=agent_patch or "",
                      test_output=log, test_exit_code=rc, duration_seconds=time.perf_counter() - start_time,
                      error=None if ok else error, trace=trace, trace_json_path=json_path)


async def verify(docker, config, task, snapshot_path, **kw):
    return await asyncio.to_thread(verify_sync, docker, config, task, kw.get("agent_patch", ""),
                                   kw.get("start_time", time.perf_counter()), kw.get("trace"), kw.get("agent_error"))


def load(path):
    tasks = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            if isinstance(r.get(k), str):
                r[k] = json.loads(r[k])
        RECORDS[r["instance_id"]] = r
        tasks.append(Task(instance_id=r["instance_id"], repo=r["repo"], base_commit=r["base_commit"],
                          problem_statement=r["problem_statement"], hints_text=r.get("hints_text") or "",
                          patch="", test_patch=r["test_patch"], created_at=str(r.get("created_at", "")),
                          FAIL_TO_PASS=tuple(r["FAIL_TO_PASS"]), PASS_TO_PASS=tuple(r["PASS_TO_PASS"])))
    return tasks


def install_hooks():
    ev.resolve_task_snapshot_paths = lambda *a, **k: (SENTINEL, None, None)
    ev.verify_task = verify
    ev.Evaluator._hydrate_task_from_secret = lambda self, t: t
    ar.extract_snapshot = prepare_repo
    for n in ("setup_container_wheels", "install_editable_package", "install_test_dependencies", "setup_workspace_test_config"):
        setattr(ar, n, _noop)


def make_config(a, sub, out, image, models=None, adapters=None):
    ev_cfg = {}
    if sub is not None:
        ec = next((c for n in ("eval_config.yaml", "eval_config.yml") if (c := sub / n).is_file()), None)
        raw = (yaml.safe_load(ec.read_text()) or {}) if ec else {}
        ev_cfg = raw.get("evaluation", raw)
    turns = ev_cfg.get("max_turns", ev_cfg.get("max_llm_calls"))
    limits, gen = build_submission_limits()
    return EvalConfig(tasks_path=pathlib.Path(a.tasks), snapshots_dir=SENTINEL.parent, results_dir=out,
                      submission_dir=sub or pathlib.Path(str(pathlib.Path.home() / "g4/data/sample_submission")),
                      models=models or ModelRegistry(), sandbox="docker", image=image,
                      timeout_seconds=int(ev_cfg.get("timeout_seconds", 300)), verification_timeout_seconds=900,
                      max_time_minutes=float(ev_cfg.get("max_time_minutes", 60.0)) * a.time_scale,
                      max_tool_calls=int(ev_cfg.get("max_tool_calls", 100)), max_turns=int(turns) if turns is not None else None,
                      limits=limits, generation_constraints=gen, adapter_manifest=adapters,
                      context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                      events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
                      verbose=False)


def evaluator_for(cfg, task_id):
    e = ev.Evaluator(cfg)
    e.docker.config.environment = {"TEST_TMPDIR": "/tmp", "PATH": ENVPATH, "CONDA_PREFIX": "/opt/conda/envs/testbed",
                                   "CONDA_DEFAULT_ENV": "testbed", "PYTHONDONTWRITEBYTECODE": "1"}
    e.docker._rb_task = task_id
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True); ap.add_argument("--sub")
    ap.add_argument("--gold", action="store_true"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1"); ap.add_argument("--tag-file", default=""); ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--rm-image", action="store_true"); ap.add_argument("--prefetch", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="agent mode: run each task N times in a row on one pulled image; rows get kind r0..r{N-1} and the tag file ID#rK")
    a = ap.parse_args()
    assert a.repeat >= 1 and not (a.gold and a.repeat > 1), "--repeat is for agent mode"
    KINDS = ("gold", "empty") if a.gold else (tuple(f"r{k}" for k in range(a.repeat)) if a.repeat > 1 else ("",))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    install_hooks()
    tasks = [t for i, t in enumerate(load(a.tasks)) if i % a.nshards == a.shard]
    res_path = out / (f"gold_shard{a.shard}.jsonl" if a.gold else f"results_shard{a.shard}.jsonl")
    pattern = "gold_shard*.jsonl" if a.gold else "results_shard*.jsonl"
    done = {json.loads(l)["id"] + "|" + json.loads(l).get("kind", "") for f in out.glob(pattern) for l in f.read_text().splitlines() if l.strip()}
    # drop finished tasks before the prefetcher sees them: --rm-image deleted their images, so a restart would re-pull every one
    tasks = [t for t in tasks if any(f"{t.instance_id}|{k}" not in done for k in KINDS)]
    sub = pathlib.Path(a.sub) if a.sub else None
    models = adapters = None
    if sub is not None:
        server = json.loads(pathlib.Path(str(pathlib.Path.home() / "g4/runs/server.json")).read_text())
        models = ModelRegistry()
        for m in discover_declared_models(sub):
            models.register(m, LiteLlm(model=f"openai/{server['served']}", api_base=a.api, api_key="EMPTY", num_retries=3))
        adapters = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        for name in (getattr(adapters, "adapters", None) or {}):
            for key in (name, f"adapter:{name}"):
                models.register(key, LiteLlm(model=f"openai/{name}", api_base=a.api, api_key="EMPTY", num_retries=3))

    import subprocess, threading
    def image_of(t):
        r = RECORDS[t.instance_id]
        return r.get("docker_image") or r["image_name"]
    def pull(img):
        subprocess.run(["docker", "pull", "-q", img], capture_output=True)

    async def go():
        pulls = {}
        for i, t in enumerate(tasks, 1):
            rec = RECORDS[t.instance_id]
            image = image_of(t)
            for nxt in tasks[i:i + a.prefetch]:
                im = image_of(nxt)
                if im not in pulls:
                    th = threading.Thread(target=pull, args=(im,), daemon=True); th.start(); pulls[im] = th
            if image in pulls:
                await asyncio.to_thread(pulls[image].join)
            cfg = make_config(a, sub, out, image, models, adapters)
            e = evaluator_for(cfg, t.instance_id)
            try:
                if a.gold:
                    for kind, p in (("gold", rec["patch"]), ("empty", "")):
                        if f"{t.instance_id}|{kind}" in done:
                            continue
                        r = await verify(e.docker, cfg, t, SENTINEL, agent_patch=p, start_time=time.perf_counter())
                        with open(res_path, "a") as f:
                            f.write(json.dumps(dict(id=t.instance_id, kind=kind, resolved=bool(r.resolved), exit=r.test_exit_code,
                                                    err=str(r.error or "")[:300])) + "\n")
                    continue
                for kind in KINDS:  # ("",) unless --repeat
                    if f"{t.instance_id}|{kind}" in done:
                        continue
                    tag = t.instance_id + (f"#{kind}" if kind else "")
                    if kind != KINDS[0]:
                        e = evaluator_for(cfg, t.instance_id)  # a fresh evaluator (and container) per replicate
                    attempt_id = uuid.uuid4().hex
                    VERIFIED_AT.pop(t.instance_id, None)
                    if a.tag_file:  # the capture proxy joins calls to this attempt by attempt_id
                        tag_path = pathlib.Path(a.tag_file)
                        tag_tmp = tag_path.with_name(tag_path.name + ".tmp")
                        tag_tmp.write_text(f"{tag}@{attempt_id}")
                        tag_tmp.replace(tag_path)
                    t0 = time.time()
                    try:
                        r = await e.evaluate_task(task=t, task_index=i, total_tasks=len(tasks))
                        (out / "patches").mkdir(exist_ok=True)
                        (out / "patches" / f"{tag}.patch").write_text(r.agent_patch or "")
                        (out / "test_outputs").mkdir(exist_ok=True)
                        (out / "test_outputs" / f"{tag}.log").write_text((r.test_output or "")[-20000:])
                        rec_out = dict(id=t.instance_id, kind=kind, resolved=bool(r.resolved), exit=r.test_exit_code,
                                       patch_chars=len((r.agent_patch or "").strip()), tool_calls=r.tool_calls,
                                       dur=round(r.duration_seconds, 1), wall=round(time.time() - t0, 1), err=str(getattr(r, "error", "") or "")[:300],
                                       verified_at=VERIFIED_AT.pop(t.instance_id, None))
                    except Exception as ex:
                        rec_out = dict(id=t.instance_id, kind=kind, resolved=False, exit=None, patch_chars=0, tool_calls=None, dur=None,
                                       wall=round(time.time() - t0, 1), err=f"EXC {ex!r}"[:300], verified_at=VERIFIED_AT.pop(t.instance_id, None))
                    rec_out["attempt_id"] = attempt_id
                    rec_out["started_at"] = t0
                    with open(res_path, "a") as f:
                        f.write(json.dumps(rec_out) + "\n")
                    print(json.dumps(rec_out), flush=True)
            finally:  # every task frees its image, gold controls included
                if a.rm_image:
                    subprocess.run(["docker", "rmi", "-f", image], capture_output=True)

    asyncio.run(go())


if __name__ == "__main__":
    main()
