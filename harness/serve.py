"""Start vLLM exactly like the scorer (setup_vllm_server in the official notebook) and keep it alive.
usage: serve.py <model_alias> [submission_dir_with_adapters]"""
import os, sys, time, json, pathlib
for k, v in dict(VLLM_NO_USAGE_STATS="1", VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_ENGINE_READY_TIMEOUT_S="1200",
                 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                 NCCL_DEBUG="WARN", TRANSFORMERS_OFFLINE="1").items():
    os.environ[k] = v
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
alias = sys.argv[1]
sub = sys.argv[2] if len(sys.argv) > 2 else None
adapters = discover_adapters(sub, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS) if sub else None
amap = (getattr(adapters, "adapters", None) or {}) if adapters else {}
ranks = [r for i in amap.values() if (r := getattr(i, "rank", None)) is not None and r > 0]
extra = ("--hf-overrides", json.dumps({"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"})) if alias == "gemma-4-12b-it-qat-w4a16-ct" else ()
cfg = VllmConfig(model=str(pathlib.Path.home() / "g4/models" / alias), port=8000, host="127.0.0.1",
                 tool_call_parser="gemma4", reasoning_parser="gemma4", max_model_len=32768, dtype="bfloat16",
                 gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=bool(amap),
                 max_loras=max(1, len(amap)), max_lora_rank=max(ranks) if ranks else 128,
                 scheduler_reserve_full_isl=False, tensor_parallel_size=int(os.environ.get("TP", "4")), startup_timeout=1200, extra_args=extra)
t0 = time.time()
srv = VllmServer(cfg, adapter_manifest=adapters).start()
info = {"alias": alias, "served": getattr(srv, "effective_model_path", cfg.model), "adapters": list(amap), "startup_s": round(time.time() - t0, 1)}
pathlib.Path(str(pathlib.Path.home() / "g4/runs/server.json")).write_text(json.dumps(info))
print("READY", info, flush=True)
while True:
    time.sleep(3600)
