# A local test bench for small Gemma 4 coding agents

Code and task lists from the paper "Measure first: local scoring and training for small Gemma 4 coding agents" by
Nikhil Reddy Gogu (Gemma 4 Developer Agent Competition, paper track). The paper is in [`paper/paper.pdf`](paper/paper.pdf).
Everything runs against the competition's own agent loop and packages; nothing here replaces the scorer.

## Contents

| Path | What it is |
|---|---|
| `env/ENV.md`, `env/mkenv.sh`, `env/pad_wheel.py` | The scorer environment and the three fixes that make reference patches pass locally. |
| `harness/serve.py` | Starts vLLM with the scorer's settings (gemma4 tool and reasoning parsers, 32,768-token context, LoRA adapters found in a submission directory). Prints `READY` and writes `~/g4/runs/server.json`. |
| `harness/run_eval.py` | Runs the competition's public tasks with the unchanged agent loop against that server. |
| `harness/verify_gold.py` | Soundness controls for public tasks: the reference patch must pass, an empty patch must fail. |
| `harness/rebench_eval.py` | Runs the unchanged agent loop inside SWE-rebench task images (repository moved to `/workspace`, history rebuilt at the base commit, protected tests reset, verified by FAIL_TO_PASS and PASS_TO_PASS). `--gold` runs the controls. |
| `harness/loop_screen.py` | Flags runs in which one identical tool call is made five or more times. |
| `harness/project_l4.py` | Projects wall time on the scorer's 4x L4 from a run's traces. |
| `cx/` | cx (find, show, outline, callers, replace and insert by reference, compile-checked), the SEARCH/REPLACE editor `apply.py`, `find.py`, `t.py`, the skill as a submission installs it, and the regression tests. |
| `pipeline/tokproxy.py` | Proxy between the agent and vLLM that logs the served prompt and completion token IDs of every call. |
| `pipeline/build_sft_clean.py` | Packs verified runs into training sequences (one loss span per call, cut at the first successful `submit_patch`). |
| `pipeline/train_lora.py` | LoRA trainer for the QAT checkpoints: chunked head and loss, per-call weighting, several GPUs. |
| `tasks/ood86.jsonl` | 86 SWE-rebench tasks, one per repository, created 24 Jan to 30 Apr 2025, each passing both controls in this harness. Licence per task. |
| `tasks/public50.txt` | The 50 of the competition's 129 public tasks that pass both controls after the environment fixes. |
| `tasks/train_pools.jsonl` | 3,046 training tasks in four pools. No repository is shared with `ood86.jsonl`. Licence per task. |
| `MANIFEST.json` | Each file's source and md5, and whether the release copy differs from the source. |
| `paper/` | The paper as PDF and its LaTeX source (`paper.tex`, `refs.bib`, `figs/`). |

## Layout

The scripts expect a working directory `~/g4` (`models/` with the QAT checkpoints, `data/`, `runs/`, `src/`) and the
scorer environment in `~/venv-score`. The cx tests expect the tools in `~/g4/src/tools` and task snapshots in
`~/g4/snapcache`.

## Task licences

`tasks/ood86.jsonl` and `tasks/train_pools.jsonl` carry SWE-rebench's `license_name` for every task. The 86 evaluation
tasks are MIT (33), Apache 2.0 (25), BSD variants (27) and one dual Apache 2.0 and MIT. Task contents (issues, patches,
tests) are not redistributed here; they come from SWE-rebench and its images.

## Citation

```bibtex
@misc{gogu2026measurefirst,
  author = {Gogu, Nikhil Reddy},
  title = {Measure first: local scoring and training for small Gemma 4 coding agents},
  year = {2026},
  howpublished = {Gemma 4 Developer Agent Competition, paper track},
  url = {https://github.com/NikhilGogu/measure-first}
}
```

## Licence of this code

Apache License, Version 2.0. The full text is in `LICENSE`.
