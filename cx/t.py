"""Run targeted tests quickly. usage: python3 /tmp/t/t.py <test file or dir or source file> [more...] [-k EXPR]
Source files are mapped to likely test files. Prints pass/fail counts and the first failure, max ~60 lines."""
import os, re, subprocess, sys
WS = "/workspace"
args = sys.argv[1:]; k = None
if "-k" in args:
    i = args.index("-k"); k = args[i + 1] if i + 1 < len(args) else None; args = args[:i] + args[i + 2:]
targets = []
for a in args:
    rel = re.sub(r"^/workspace/", "", a)
    if re.search(r"(^|/)(tests?|testing)/|test_[^/]*\.py|_test\.py", rel) or os.path.isdir(os.path.join(WS, rel)):
        targets.append(rel); continue
    stem = os.path.splitext(os.path.basename(rel))[0]
    cands = []
    for root, dirs, fs in os.walk(WS):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "build", "dist")]
        for f in fs:
            if f.endswith(".py") and (f == f"test_{stem}.py" or f == f"{stem}_test.py" or (f.startswith("test_") and stem in f)):
                cands.append(os.path.relpath(os.path.join(root, f), WS))
    targets.extend(sorted(cands, key=len)[:2] or [])
if not targets:
    print("No test files found for", args, "- pass a test file path."); sys.exit(2)
cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--no-header", "-rfE"] + targets + (["-k", k] if k else [])
try:
    p = subprocess.run(cmd, cwd=WS, capture_output=True, text=True, timeout=150)
    out = p.stdout + p.stderr
except subprocess.TimeoutExpired as e:
    out = (e.stdout or "") + "\nTIMEOUT after 150s"
    p = None
lines = out.strip().split("\n")
print("CMD:", " ".join(cmd[2:]))
if len(lines) > 60:
    head = [l for l in lines if re.search(r"(Error|assert|FAILED|^E )", l)][:25]
    print("\n".join(head + ["..."] + lines[-25:]))
else:
    print("\n".join(lines))
sys.exit(0 if p is not None and p.returncode == 0 else 1)
