# Scorer environment fixes

The local scorer uses the competition's own packages: swegemma 0.2.11, adk-submission 0.2.13, adk-eval-core 0.1.0 and
google-adk 1.36.1, with vLLM 0.19.1 (`mkenv.sh` builds that environment from a local wheelhouse). Three faults in the task
environment made correct fixes fail. None of them is in the agent. The first was reported on the competition forum (thread
744370); another participant's rebuilt notebook sandbox passes 127 of the 129 reference patches (thread 751487).

1. **Test dependencies are installed with `--no-deps`.** pydantic then cannot import `typing_inspection`, so fastapi's
   reference fixes fail at test collection (66 of 67 before the fix). Fix: add these wheels, with their own dependencies,
   to the wheels directory: typing_inspection, inline_snapshot, dirty_equals, annotated_doc, pytest_httpbin.
2. **Every wheel under 20 MB is injected into every task container** (`_LARGE_WHEEL_THRESHOLD_BYTES` in
   `swegemma/harness/container_setup.py`). The inline_snapshot pytest plugin then loads into rich's tests and breaks them.
   Fix: `python3 pad_wheel.py <inline_snapshot wheel>` pads it past 20 MB, so it is injected only into tasks that import it.
3. **Unpacked wheels are cached by name, not content** (tar files under the system temp directory, in
   `swegemma_sp_cache_v8`). A changed wheel keeps its old contents. Fix: delete that directory after changing any wheel.

Serving: vLLM 0.19.1's health check fails with FastAPI 0.141; pin fastapi 0.119.1 and starlette 0.48.0 in the serving
environment.

After the fixes, 50 of the 129 public tasks pass their reference fix and fail an empty patch (`tasks/public50.txt`): 38 of
48 from rich and 12 of 67 from fastapi. Most remaining fastapi tasks need an older starlette per task.
