"""Rank where to look in /workspace for an issue. usage: python3 /tmp/t/find.py <keywords or pasted issue text> (or stdin)
Scores Python functions/classes with BM25 over identifier tokens plus exact-identifier and path boosts. Prints top symbols with line ranges, top files, and related test files."""
import ast, math, os, re, sys, collections
WS = "/workspace"
q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
SKIP = {".git", "node_modules", "build", "dist", "__pycache__", ".tox", ".venv", "venv", ".mypy_cache", ".pytest_cache", "site-packages"}
def toks(s):
    out = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s):
        out.append(w.lower())
        for p in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", w):
            if len(p) > 1: out.append(p.lower())
        for p in w.split("_"):
            if len(p) > 1 and p.lower() != w.lower(): out.append(p.lower())
    return out
STOP = set("the a an and or of to in is it for on with be this that by as are was not if from at but when should can will would we i you use using used has have get set add new fix bug issue error test tests self none true false return def class import".split())
qt = [t for t in toks(q) if t not in STOP and len(t) > 2]
exact_names = set()
for m in re.finditer(r"`([^`\n]+)`|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z]+_[A-Za-z0-9_]+|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*", q):
    for p_ in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(0)):
        if len(p_) > 2:
            exact_names.add(p_)
paths_in_q = set(re.findall(r"[\w/.-]+\.py", q))
docs, files, tests = [], collections.Counter(), []
for root, dirs, fs in os.walk(WS):
    dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
    for f in fs:
        if not f.endswith(".py"): continue
        p = os.path.join(root, f); rel = os.path.relpath(p, WS)
        is_test = bool(re.search(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$", rel))
        try:
            src = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        if is_test:
            tests.append((rel, src)); continue
        try:
            tree = ast.parse(src)
        except Exception:
            docs.append((rel, "<module>", 1, src.count("\n") + 1, toks(rel + " " + src))); continue
        lines = src.split("\n"); seen_spans = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                s, e = node.lineno, getattr(node, "end_lineno", node.lineno)
                body = "\n".join(lines[s - 1:e]) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "\n".join(lines[s - 1:min(e, s + 30)])
                docs.append((rel, node.name, s, e, toks(rel + " " + node.name + " " + body)))
        docs.append((rel, "<module>", 1, len(lines), toks(rel + " " + "\n".join(l for l in lines if not l.startswith((" ", "\t")))[:20000])))
N = len(docs) or 1
avgdl = sum(len(d[4]) for d in docs) / N
df = collections.Counter()
for d in docs:
    df.update(set(d[4]))
qc = collections.Counter(qt)
res = []
for rel, name, s, e, tk in docs:
    tf = collections.Counter(tk); dl = len(tk); sc = 0.0
    for t, w in qc.items():
        if t in tf:
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            sc += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl)) * min(w, 3)
    if name in exact_names: sc *= 2.5
    if any(rel.endswith(p) or p.endswith(rel) for p in paths_in_q): sc *= 2.0
    elif re.match(r"(scripts|docs|benchmarks?|examples?|tools)/", rel): sc *= 0.5
    if name != "<module>" and e - s > 150: sc *= 0.85
    res.append((sc, rel, name, s, e))
res.sort(reverse=True)
print("TOP SYMBOLS (file:lines symbol score)")
for sc, rel, name, s, e in res[:15]:
    if sc <= 0: break
    print(f"  {rel}:{s}-{e} {name} {sc:.1f}")
best = collections.defaultdict(list)
for sc, rel, name, s, e in res:
    if sc > 0 and len(best[rel]) < 2: best[rel].append(sc)
for rel, v in best.items():
    files[rel] = v[0] + 0.3 * (v[1] if len(v) > 1 else 0)
print("TOP FILES")
for rel, sc in files.most_common(6):
    print(f"  {rel} {sc:.1f}")
mods = {os.path.splitext(os.path.basename(r))[0] for r, _ in files.most_common(4)}
names = {n for _, _, n, _, _ in res[:8] if n != "<module>"}
rt = []
for rel, src in tests:
    hit = sum(1 for m in mods if m in rel or re.search(r"\b" + re.escape(m) + r"\b", src[:3000])) + sum(1 for n in names if n in src)
    if hit: rt.append((hit, rel))
print("RELATED TESTS")
for h, rel in sorted(rt, reverse=True)[:6]:
    print(f"  {rel}")
