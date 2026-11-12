"""LoRA SFT for Gemma 4 QAT checkpoints on per-call examples (exact prompt token IDs -> raw completion token IDs).

usage: [torchrun --nproc_per_node G] train_lora.py --model DIR --data train.jsonl [--val val.jsonl] --out ADAPTER_DIR
           [--rank 16] [--lr 1e-4] [--epochs 2] [--accum 8] [--max-len 32000] [--limit N] [--chunk 1024]
           [--check] [--grad-dump PATH]
Each JSONL row: {"prompt_token_ids": [...], "completion_token_ids": [...], "weight": 1.0 (optional), "task": "..."}
or a merged row {"input_ids": [...], "loss_spans": [[start, end), ...]}.
Loss is mean cross-entropy over completion tokens only; the head and loss run in chunks over the final hidden states,
so long prompts cost no vocabulary-sized logits. Base = dequantized QAT weights (frozen).
Multi-GPU: every rank builds the same shuffled order; each optimizer step's `accum` sequences are split across ranks
by token count, and LoRA gradients are summed across ranks before clipping, so G GPUs take the same steps as one.
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText, CompressedTensorsConfig

SUFFIXES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            "per_layer_input_gate", "per_layer_projection", "per_layer_model_projection"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--val")
ap.add_argument("--out", required=True); ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=int, default=0)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--accum", type=int, default=8); ap.add_argument("--max-len", type=int, default=32000)
ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=1024); ap.add_argument("--check", action="store_true")
ap.add_argument("--grad-dump", help="save the summed gradient of optimizer step --grad-step to this path and exit")
ap.add_argument("--grad-step", type=int, default=2, help="step whose gradient --grad-dump saves (step 1 has zero LoRA-A gradients)")
ap.add_argument("--perturb-b", action="store_true", help="test only: set LoRA-B to identical small random values so step-1 LoRA-A gradients are nonzero")
ap.add_argument("--save-every", type=int, default=20, help="steps between resumable checkpoints (adapter + optimizer + step)")
ap.add_argument("--save-epochs", default="", help="comma-separated epochs after which to save the adapter to OUT/epochK")
ap.add_argument("--weighting", choices=["token", "call", "row", "task"], default="call",
                help="token: mean over all supervised tokens in a step; call: every model call weighted equally (as unmerged per-call SFT); row: per-row token mean, rows averaged (pilot); task: token loss weighted by each row's weight field (set to equalise tasks), normalised by weighted tokens")
ap.add_argument("--psd", help="per-call rows with hinted-teacher prompts; adds lambda * sum_t A_t * CE_t / task tokens inside the same updates")
ap.add_argument("--psd-scores", help="per-token log-probabilities and advantages aligned with the --psd rows")
ap.add_argument("--psd-lambda", type=float, default=0.1)
ap.add_argument("--psd-check", action="store_true", help="sign check of PSD gradients against plain CE, then exit")
ap.add_argument("--stop-step", type=int, default=0, help="stop after this optimizer step (the schedule keeps its full horizon)")
a = ap.parse_args()

WORLD, RANK, LOCAL = (int(os.environ.get(k, d)) for k, d in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
torch.cuda.set_device(LOCAL)
DEV = torch.device(f"cuda:{LOCAL}")
if WORLD > 1:
    dist.init_process_group("nccl")
random.seed(a.seed); torch.manual_seed(a.seed)


def log(*x):
    if RANK == 0:
        print(*x, flush=True)


def load(path):
    rows, dropped = [], [0, 0, 0]
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if "input_ids" in r:
            ids, spans = r["input_ids"], [tuple(x) for x in r["loss_spans"]]
        else:
            p, c = r["prompt_token_ids"], r["completion_token_ids"]
            ids, spans = p + c, [(len(p), len(p) + len(c))]
        if not ids or not spans or len(ids) > a.max_len:
            dropped[0] += 1; dropped[1] += len(spans); dropped[2] += sum(e0 - s0 for s0, e0 in spans)
            continue
        rows.append((ids, spans, float(r.get("weight", 1.0))))
    if dropped[0]:
        log(f"{path}: dropped {dropped[0]} rows over --max-len ({dropped[1]} calls, {dropped[2]} supervised tokens)")
    return rows


train = load(a.data)
if a.limit:
    train = train[: a.limit]
val = load(a.val) if a.val else []
log(f"train sequences {len(train)}, val {len(val)}, mean tokens {sum(len(i) for i, _, _ in train) / max(len(train), 1):.0f}, supervised calls {sum(len(s) for _, s, _ in train)}, gpus {WORLD}")

psd_by_step, psd_tokens = {}, {}
if a.psd:
    import hashlib
    sc = {(s["task"], s["call"]): s for s in map(json.loads, open(a.psd_scores))}
    rows_by_task = {}
    for r in map(json.loads, open(a.psd)):
        s = sc[(r["task"], r["call"])]
        digest = hashlib.sha256(json.dumps([r["student_ids"], r["teacher_ids"], r["completion_ids"]]).encode()).hexdigest()
        assert digest == r["row_sha"] == s["row_sha"] and s["n"] == len(r["completion_ids"]) == len(s["adv"]), (r["task"], r["call"])
        ids = r["student_ids"] + r["completion_ids"]
        rows_by_task.setdefault(r["task"], []).append((ids, [(len(r["student_ids"]), len(ids))], s["adv"]))
    # Placement is fixed by the epoch-1 boundary alone, so stopping early reproduces a prefix of the full run.
    e1 = round(len(train) / a.accum)
    assert len(rows_by_task) <= e1, (len(rows_by_task), e1)
    for j, task in enumerate(sorted(rows_by_task)):
        k = j * e1 // len(rows_by_task)
        assert k not in psd_by_step
        psd_by_step[k] = rows_by_task[task]
        psd_tokens[k] = sum(len(w) for _, _, w in rows_by_task[task])
    log(f"psd: {len(rows_by_task)} tasks, {sum(len(v) for v in rows_by_task.values())} rows, {sum(psd_tokens.values())} tokens, lambda {a.psd_lambda}, steps {sorted(psd_by_step)[:4]}...")

model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": LOCAL},
                                                    quantization_config=CompressedTensorsConfig(run_compressed=False))
model.requires_grad_(False)
targets = [n for n, m in model.named_modules() if n.startswith("model.language_model.") and n.rsplit(".", 1)[-1] in SUFFIXES]
cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha or 2 * a.rank, lora_dropout=0.0, bias="none", target_modules=targets, task_type="CAUSAL_LM")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
pm = get_peft_model(model, cfg)
if RANK == 0:
    pm.print_trainable_parameters()
params = [p for p in pm.parameters() if p.requires_grad]
if WORLD > 1:
    for p in params:
        dist.broadcast(p.data, 0)
if a.perturb_b:
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if p.requires_grad and "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=gen) * 1e-3)
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)

inner = pm.base_model.model
CAP = inner.config.get_text_config().final_logit_softcapping


def head_ce(h, t):
    logits = inner.lm_head(h)
    if CAP is not None:
        logits = torch.tanh(logits / CAP) * CAP
    return F.cross_entropy(logits.float(), t, reduction="none")


def token_losses(ids, spans):
    # Merged rows can supervise thousands of tokens; full-vocabulary logits for all of them at once do not fit,
    # so the head and loss run in checkpointed chunks over the final hidden states (same math as the model's head).
    keep = torch.tensor([t - 1 for s0, e0 in spans for t in range(s0, e0)], device=DEV)
    tgt = torch.tensor([ids[t] for s0, e0 in spans for t in range(s0, e0)], device=DEV)
    h = inner.model(input_ids=torch.tensor([ids], device=DEV), use_cache=False).last_hidden_state[0, keep]
    parts = []
    for i in range(0, len(tgt), a.chunk):
        if torch.is_grad_enabled():
            parts.append(checkpoint(head_ce, h[i:i + a.chunk], tgt[i:i + a.chunk], use_reentrant=False))
        else:
            parts.append(head_ce(h[i:i + a.chunk], tgt[i:i + a.chunk]))
    return torch.cat(parts)


def units(spans, w=1.0):
    if a.weighting == "task":
        return w * sum(e0 - s0 for s0, e0 in spans)
    return {"token": sum(e0 - s0 for s0, e0 in spans), "call": len(spans), "row": 1}[a.weighting]


def numerator(tl, spans):
    # Loss summed in the chosen unit; dividing by the units of the whole step gives that unit's mean.
    if a.weighting in ("token", "task"):
        return tl.sum()
    if a.weighting == "row":
        return tl.mean()
    out, off = 0.0, 0
    for s0, e0 in spans:
        out = out + tl[off:off + e0 - s0].mean()
        off += e0 - s0
    return out


@torch.no_grad()
def evaluate(rows):
    if not rows:
        return None
    pm.eval()
    mine = rows[RANK::WORLD]
    t = torch.tensor([sum(w * numerator(token_losses(i, s), s).item() for i, s, w in mine), sum(units(s, w) for _, s, w in mine)],
                     device=DEV, dtype=torch.float64)
    if WORLD > 1:
        dist.all_reduce(t)
    pm.train()
    return (t[0] / t[1]).item()


def assign(chunk):
    # Longest-first greedy split of one step's sequences across ranks by token count; identical on every rank.
    loads, mine = [0] * WORLD, []
    for i in sorted(chunk, key=lambda i: (-len(train[i][0]), i)):
        r = min(range(WORLD), key=lambda r: (loads[r], r))
        loads[r] += len(train[i][0])
        if r == RANK:
            mine.append(i)
    return mine


def sync_grads():
    if WORLD == 1:
        return
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().reshape(-1) for p in params])
    dist.all_reduce(flat)
    off = 0
    for p in params:
        n = p.numel()
        p.grad = flat[off:off + n].view_as(p).to(p.dtype)
        off += n


if a.check:
    # Chunked loss must equal the model's own head on a short row; then one backward on the row with most supervised tokens.
    ids, spans, _ = min(train, key=lambda r: len(r[0]))
    with torch.no_grad():
        keep = [t - 1 for s0, e0 in spans for t in range(s0, e0)]
        out = pm(input_ids=torch.tensor([ids], device=DEV), logits_to_keep=torch.tensor(keep, device=DEV), use_cache=False)
        ref = F.cross_entropy(out.logits[0].float(), torch.tensor([ids[t] for s0, e0 in spans for t in range(s0, e0)], device=DEV)).item()
        new = token_losses(ids, spans).mean().item()
    log(f"check short row: model head {ref:.5f} chunked {new:.5f}")
    assert abs(ref - new) <= 1e-4 * max(1.0, abs(ref)), "chunked loss differs from the model head"
    # Gradients: perturb LoRA-B (zero at init, which would zero every LoRA-A gradient), then compare backward passes.
    pm.train()
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if p.requires_grad and "lora_B" in n:
                p.normal_(0, 1e-3)
    out = pm(input_ids=torch.tensor([ids], device=DEV), logits_to_keep=torch.tensor(keep, device=DEV), use_cache=False)
    F.cross_entropy(out.logits[0].float(), torch.tensor([ids[t] for s0, e0 in spans for t in range(s0, e0)], device=DEV)).backward()
    g_ref = torch.cat([p.grad.float().reshape(-1) for p in params]); pm.zero_grad(set_to_none=True)
    out = pm(input_ids=torch.tensor([ids], device=DEV), logits_to_keep=torch.tensor(keep, device=DEV), use_cache=False)
    F.cross_entropy(out.logits[0].float(), torch.tensor([ids[t] for s0, e0 in spans for t in range(s0, e0)], device=DEV)).backward()
    g_ref2 = torch.cat([p.grad.float().reshape(-1) for p in params]); pm.zero_grad(set_to_none=True)
    token_losses(ids, spans).mean().backward()
    g_new = torch.cat([p.grad.float().reshape(-1) for p in params]); pm.zero_grad(set_to_none=True)
    rel = ((g_ref - g_new).norm() / g_ref.norm()).item()
    floor = ((g_ref - g_ref2).norm() / g_ref.norm()).item()
    log(f"check gradients: chunked vs model head {rel:.2e}; model head vs itself {floor:.2e} (norm {g_ref.norm().item():.3e})")
    assert rel < 1e-2, "chunked gradients differ from the model head"
    ids, spans, _ = max(train, key=lambda r: sum(e0 - s0 for s0, e0 in r[1]))
    pm.train(); torch.cuda.reset_peak_memory_stats(); t1 = time.time()
    token_losses(ids, spans).mean().backward()
    log(f"check heaviest row: {len(ids)} tokens, {sum(e0 - s0 for s0, e0 in spans)} supervised, {time.time() - t1:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    raise SystemExit

order = []
while len(order) < len(train) * a.epochs:
    o = list(range(len(train))); random.shuffle(o); order.extend(o)
order = order[: int(len(train) * a.epochs)]
chunks = [order[j:j + a.accum] for j in range(0, len(order), a.accum)]
total_steps = len(chunks)


def lr_at(step):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    prog = (step - a.warmup) / max(1, total_steps - a.warmup)
    return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


step = 0
ckpt = os.path.join(a.out, "resume.pt")
if os.path.exists(ckpt) and not a.grad_dump:
    st = torch.load(ckpt, map_location=DEV, weights_only=False)
    for p, v in zip(params, st["params"]):
        p.data.copy_(v)
    opt.load_state_dict(st["opt"]); step = st["step"]
    log(f"resumed from {ckpt} at step {step}")
if not a.grad_dump and step == 0:
    log("val loss before:", evaluate(val))
pm.train()
t0 = time.time()
run = torch.zeros(2, device=DEV, dtype=torch.float64)
psd_run = torch.zeros(2, device=DEV, dtype=torch.float64)
if a.psd_check:
    flat = lambda: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().reshape(-1) for p in params])
    # Constant weights through the PSD loss path: A = +1 must follow plain summed CE, A = -1 must oppose it.
    ids, spans, adv = min((r for v in psd_by_step.values() for r in v), key=lambda r: len(r[0]))
    # Two plain CE backward passes differ slightly in bf16 with recomputation; their cosine is the noise floor, and the PSD
    # path must match CE (A = +1) or oppose it (A = -1) to within that floor plus 1e-3.
    grads = {}
    for name, wts in (("ce", None), ("ce2", None), ("plus", [1.0] * len(adv)), ("minus", [-1.0] * len(adv))):
        opt.zero_grad(set_to_none=True)
        tl = token_losses(ids, spans)
        (tl.sum() if wts is None else (tl * torch.tensor(wts, device=DEV)).sum()).backward()
        grads[name] = flat()
    opt.zero_grad(set_to_none=True)
    floor = F.cosine_similarity(grads["ce2"], grads["ce"], dim=0).item()
    cp = F.cosine_similarity(grads["plus"], grads["ce"], dim=0).item()
    cm = F.cosine_similarity(grads["minus"], grads["ce"], dim=0).item()
    # signed relative errors catch scaling bugs that cosine cannot: A = +1 must equal g_CE, A = -1 must equal -g_CE
    n = grads["ce"].norm()
    noise = ((grads["ce2"] - grads["ce"]).norm() / n).item()
    rp = ((grads["plus"] - grads["ce"]).norm() / n).item()
    rm = ((grads["minus"] + grads["ce"]).norm() / n).item()
    log(f"psd check on a {len(ids)}-token row: noise cos {floor:+.6f} rel {noise:.5f} | A=+1 cos {cp:+.6f} rel {rp:.5f} | A=-1 cos {cm:+.6f} rel {rm:.5f}")
    tol = max(3 * noise, 0.02)
    raise SystemExit(0 if floor > 0.99 and cp >= floor - 1e-3 and cm <= -(floor - 1e-3) and rp <= tol and rm <= tol else 1)
for chunk in chunks[step:]:
    denom = sum(units(train[i][1], train[i][2]) for i in chunk)
    for i in assign(chunk):
        ids, spans, w = train[i]
        num = numerator(token_losses(ids, spans), spans) * w
        (num / denom).backward()
        run[0] += num.item(); run[1] += units(spans, w)
    if step in psd_by_step:
        rows = psd_by_step[step]
        loads, mine = [0] * WORLD, []
        for j in sorted(range(len(rows)), key=lambda j: (-len(rows[j][0]), j)):
            r = min(range(WORLD), key=lambda r: (loads[r], r))
            loads[r] += len(rows[j][0])
            if r == RANK:
                mine.append(rows[j])
        for ids, spans, adv in mine:
            psd = (token_losses(ids, spans) * torch.tensor(adv, device=DEV)).sum() / psd_tokens[step]
            (a.psd_lambda * psd).backward()
            psd_run[0] += psd.item(); psd_run[1] += 1
    sync_grads()
    if a.grad_dump and step + 1 == a.grad_step:
        if RANK == 0:
            torch.save(torch.cat([p.grad.float().reshape(-1) for p in params]).cpu(), a.grad_dump)
            log(f"grad dump {a.grad_dump}: {len(chunk)} sequences, {time.time() - t0:.1f}s")
        break
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    opt.step(); opt.zero_grad(set_to_none=True); step += 1
    for ep in (int(x) for x in a.save_epochs.split(",") if x):
        if step == round(ep * len(train) / a.accum) and RANK == 0 and not a.grad_dump:
            pm.save_pretrained(os.path.join(a.out, f"epoch{ep}"))
            log(f"saved epoch {ep} at step {step}")
    if a.save_every and step % a.save_every == 0 and step < total_steps and not a.grad_dump and RANK == 0:
        os.makedirs(a.out, exist_ok=True)
        torch.save({"params": [p.detach().cpu() for p in params], "opt": opt.state_dict(), "step": step}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
    if step % 5 == 0 or step == total_steps or step == a.stop_step:
        if WORLD > 1:
            dist.all_reduce(run); dist.all_reduce(psd_run)
        psd_txt = f" psd {psd_run[0].item():.4f} over {int(psd_run[1].item())} rows" if a.psd else ""
        log(f"step {step}/{total_steps} loss {run[0].item() / max(run[1].item(), 1):.4f}{psd_txt} lr {lr_at(step):.2e} {time.time() - t0:.0f}s")
        run.zero_(); psd_run.zero_()
    if a.stop_step and step == a.stop_step:
        log(f"stopped at step {step} of a {total_steps}-step schedule")
        break
if not a.grad_dump:
    log("val loss after:", evaluate(val))
    if RANK == 0:
        pm.save_pretrained(a.out)
        if os.path.exists(ckpt):
            os.remove(ckpt)
        log("saved", a.out)
if WORLD > 1:
    dist.barrier()
    dist.destroy_process_group()
