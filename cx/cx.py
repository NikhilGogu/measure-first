"""cx: code explorer with refs for /workspace (like a browser snapshot, but for code). Python 3.7+, no dependencies.

  python3 /tmp/t/cx.py find <identifiers or issue summary>   ranked functions/classes, each with a ref like @3
  python3 /tmp/t/cx.py outline <file.py>                     every function/class in a file, with refs
  python3 /tmp/t/cx.py show @3                               numbered source of a symbol (also Class.method or file.py:120)
  python3 /tmp/t/cx.py show file.py:120-170                  numbered lines of a file range
  python3 /tmp/t/cx.py replace @3 <<'EOF'                    replace the WHOLE symbol with a complete new definition
  (complete new definition)
  EOF
  python3 /tmp/t/cx.py insert @3 after <<'EOF'               add code after (or before) a symbol, same indentation
  python3 /tmp/t/cx.py insert @5 into <<'EOF'                add a method at the end of class @5
  python3 /tmp/t/cx.py blocks <<'EOF'                        apply blocks "REPLACE @3" / "INSERT AFTER @3" / "INSERT INTO @5", each ended by "END"
  python3 /tmp/t/cx.py callers @3                            where the symbol is used
  python3 /tmp/t/cx.py tests @3 [--run] [-k expr]            related test files; --run runs them and reports PASSED/FAILED
  python3 /tmp/t/cx.py diff                                  current changes
Edits are compile-checked and keep decorators and indentation. Symbols longer than 120 lines are edited with apply.py."""
import ast, collections, difflib, hashlib, io, json, math, os, re, subprocess, sys, textwrap, tokenize

WS = "/workspace"
STATE = "/tmp/t/cx_refs.json"
MAX_REPLACE = 120
SKIP = {".git", "node_modules", "build", "dist", "__pycache__", ".tox", ".venv", "venv", ".mypy_cache", ".pytest_cache", "site-packages"}
TEST_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$")
_SYM_CACHE = {}
LAST_EDIT = {}


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"n": 0, "refs": {}}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(st, open(STATE, "w"))


def py_files():
    for root, dirs, fs in os.walk(WS):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith("."))
        for f in sorted(fs):
            if f.endswith(".py"):
                p = os.path.join(root, f)
                yield os.path.relpath(p, WS), p


def read(rel):
    return open(os.path.join(WS, rel), encoding="utf-8", errors="replace").read()


def logical_ends(src):
    """Sorted line numbers where logical lines end (NEWLINE tokens); used when end_lineno is missing (Python 3.7)."""
    ends = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.NEWLINE:
                ends.append(tok.start[0])
    except Exception:
        pass
    return ends


def node_end(node, ends):
    end = getattr(node, "end_lineno", None)
    if end:
        return end
    last = max(getattr(n, "lineno", 0) for n in ast.walk(node))
    for e in ends:
        if e >= last:
            return e
    return last


def symbols(rel):
    """[(qual, kind, start_incl_decorators, end, def_line, body_line, fingerprint)]; duplicates get #2, #3 (e.g. property setters)."""
    if rel in _SYM_CACHE:
        return _SYM_CACHE[rel]
    try:
        src = read(rel)
        tree = ast.parse(src)
    except SyntaxError as ex:
        _SYM_CACHE[rel] = []
        print("note: %s does not parse (line %s: %s); use sed -n to read it" % (rel, ex.lineno, ex.msg))
        return []
    except Exception:
        _SYM_CACHE[rel] = []
        return []
    ends = logical_ends(src) if not hasattr(tree.body[0] if tree.body else tree, "end_lineno") else []
    out, seen = [], collections.Counter()

    def visit(stmts, prefix):
        for ch in stmts:
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                base = prefix + ch.name
                seen[base] += 1
                q = base if seen[base] == 1 else "%s#%d" % (base, seen[base])
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                kind = "class" if isinstance(ch, ast.ClassDef) else ("async def" if isinstance(ch, ast.AsyncFunctionDef) else "def")
                body_line = ch.body[0].lineno if ch.body else ch.lineno
                lines = src.split("\n")
                end = node_end(ch, ends)
                fp = hashlib.md5("\n".join(lines[start - 1:end]).encode("utf-8", "replace")).hexdigest()[:10]
                bl = lines[body_line - 1] if body_line - 1 < len(lines) else ""
                body_inline = bool(ch.body) and ch.body[0].col_offset > len(bl) - len(bl.lstrip())
                out.append((q, kind, start, end, ch.lineno, body_line, fp, body_inline))
                if isinstance(ch, ast.ClassDef):
                    visit(ch.body, base + ".")
            elif isinstance(ch, (ast.If, ast.Try, ast.With, ast.For, ast.While)) or type(ch).__name__ in ("TryStar", "AsyncWith"):
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(ch, field, []) or [], prefix)
                for h in getattr(ch, "handlers", []) or []:
                    visit(h.body, prefix)
    visit(tree.body, "")
    _SYM_CACHE[rel] = out
    return out


def new_ref(st, rel, q, kind, s, e):
    x = by_qual(rel, q)
    fp = x[6] if x else ""
    base = q.split("#")[0]
    unique_fp = fp and sum(1 for y in symbols(rel) if y[6] == fp) == 1
    if unique_fp:
        for k, v in st["refs"].items():
            if not v.get("stale") and v["file"] == rel and v.get("fp") == fp and v["qual"].split("#")[0] == base:
                v.update(qual=q, start=s, end=e, kind=kind)
                return k
    for k, v in st["refs"].items():
        if v.get("stale"):
            continue
        if v["file"] == rel and v["qual"] == q and (not unique_fp or v.get("fp") in ("", None, "stale", fp)):
            v.update(start=s, end=e, kind=kind, fp=fp)
            return k
    st["n"] += 1
    k = "@%d" % st["n"]
    st["refs"][k] = {"file": rel, "qual": q, "kind": kind, "start": s, "end": e, "fp": fp}
    return k


def refresh_refs(st, rel, edit, edited_ref=None):
    """Re-anchor refs in `rel` after our own edit of lines (a, b] replaced by n new lines: shift by position, then
    bind each ref to the symbol that now starts at its shifted line (identical twin definitions stay distinct)."""
    a, b, n = edit
    delta = n - (b - a)
    syms = symbols(rel)
    by_start = {}
    for x in syms:
        by_start.setdefault(x[2], []).append(x)
    for k, v in st["refs"].items():
        if v["file"] != rel:
            continue
        base = v["qual"].split("#")[0]
        if k == edited_ref:
            want = a + 1
        elif v["start"] > b:
            want = v["start"] + delta
        elif a < v["start"] <= b:
            v["stale"] = True
            continue
        else:
            want = v["start"]
        cands = [x for x in by_start.get(want, []) if x[0].split("#")[0] == base]
        if cands:
            x = cands[0]
            v.update(qual=x[0], kind=x[1], start=x[2], end=x[3], fp=x[6])
        else:
            v["fp"] = "stale"


def by_qual(rel, q):
    for x in symbols(rel):
        if x[0] == q:
            return x
    return None


def resolve(st, target):
    """@ref, file.py, file.py:LINE, file.py:A-B, file.py:qual or qualname -> (ref or None, entry)."""
    target = target.strip().strip("`").replace("/workspace/", "")
    if target.startswith("@"):
        if target not in st["refs"]:
            sys.exit("unknown ref %s; run find or outline first" % target)
        r = st["refs"][target]
        if r.get("stale"):
            sys.exit("ref %s pointed inside code that has since been replaced; run outline %s for fresh refs" % (target, r["file"]))
        x = by_qual(r["file"], r["qual"])
        if r.get("fp") and (not x or x[6] != r["fp"]):
            base = r["qual"].split("#")[0]
            same = [y for y in symbols(r["file"]) if y[0].split("#")[0] == base and y[6] == r["fp"]]
            if sum(1 for y in symbols(r["file"]) if y[6] == r["fp"]) != 1:
                same = []
            if len(same) == 1:
                x = same[0]
                r["qual"] = x[0]
            elif not x:
                sys.exit("ref %s (%s %s) no longer exists; run outline %s" % (target, r["file"], r["qual"], r["file"]))
            else:
                print("note: the signature of %s changed since it was listed; using %s %s" % (target, r["file"], x[0]))
        if not x:
            sys.exit("ref %s (%s %s) no longer exists; run outline %s" % (target, r["file"], r["qual"], r["file"]))
        r.update(start=x[2], end=x[3], kind=x[1], fp=x[6])
        return target, r
    m = re.match(r"^([\w./-]+\.py)(?::(\d+)(?:-(\d+))?|:([\w.#]+))?$", target)
    if m:
        rel = m.group(1)
        if not os.path.exists(os.path.join(WS, rel)):
            sys.exit("no such file: " + rel)
        if m.group(3):
            return None, {"file": rel, "range": (int(m.group(2)), int(m.group(3)))}
        if m.group(2):
            ln = int(m.group(2))
            inside = [x for x in symbols(rel) if x[2] <= ln <= x[3]]
            if inside:
                q, kind, s, e = min(inside, key=lambda x: x[3] - x[2])[:4]
                k = new_ref(st, rel, q, kind, s, e)
                return k, st["refs"][k]
            return None, {"file": rel, "range": (max(1, ln - 20), ln + 20)}
        if m.group(4):
            hits = [x for x in symbols(rel) if x[0] == m.group(4) or x[0].endswith("." + m.group(4))]
            if len(hits) == 1:
                q, kind, s, e = hits[0][:4]
                k = new_ref(st, rel, q, kind, s, e)
                return k, st["refs"][k]
            if hits:
                ambiguous(st, [(rel,) + h for h in hits])
            sys.exit("%s not found in %s" % (m.group(4), rel))
        return None, {"file": rel}
    hits = []
    for rel, _ in py_files():
        if TEST_RE.search(rel):
            continue
        for x in symbols(rel):
            if x[0] == target or x[0].endswith("." + target):
                hits.append((rel,) + x)
    if len(hits) == 1:
        rel, q, kind, s, e = hits[0][:5]
        k = new_ref(st, rel, q, kind, s, e)
        return k, st["refs"][k]
    if hits:
        ambiguous(st, hits)
    sys.exit("nothing named %s; try find" % target)


def ambiguous(st, hits):
    print("ambiguous, use one of these refs:")
    for rel, q, kind, s, e, *_ in hits[:12]:
        print("  %s %s:%d-%d %s" % (new_ref(st, rel, q, kind, s, e), rel, s, e, q))
    save_state(st)
    sys.exit(1)


# ---------- find (BM25 over identifier tokens) ----------
STOP = set("the a an and or of to in is it for on with be this that by as are was not if from at but when should can will would we i you use using used has have get set add new fix bug issue error test tests self none true false return def class import".split())


def toks(s):
    out = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s):
        out.append(w.lower())
        for p in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", w):
            if len(p) > 1:
                out.append(p.lower())
        for p in w.split("_"):
            if len(p) > 1 and p.lower() != w.lower():
                out.append(p.lower())
    return out


def exact_names(q):
    names = set()
    for m in re.finditer(r"`([^`\n]+)`|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z]+_[A-Za-z0-9_]+|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*", q):
        for p in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(0)):
            if len(p) > 2:
                names.add(p)
    return names


def cmd_find(st, q):
    qt = [t for t in toks(q) if t not in STOP and len(t) > 2]
    exact = exact_names(q)
    paths_in_q = set(re.findall(r"[\w/.-]+\.py", q))
    docs, tests = [], []
    for rel, p in py_files():
        try:
            src = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        if TEST_RE.search(rel):
            tests.append((rel, src))
            continue
        lines = src.split("\n")
        for qn, kind, s, e, *_ in symbols(rel):
            body = "\n".join(lines[s - 1:e] if kind != "class" else lines[s - 1:min(e, s + 30)])
            docs.append((rel, qn, kind, s, e, toks(rel + " " + qn + " " + body)))
        docs.append((rel, "<module>", "module", 1, len(lines), toks(rel + " " + "\n".join(l for l in lines if not l.startswith((" ", "\t")))[:20000])))
    N = len(docs) or 1
    avgdl = sum(len(d[5]) for d in docs) / N
    df = collections.Counter()
    for d in docs:
        df.update(set(d[5]))
    qc = collections.Counter(qt)
    res = []
    for rel, qn, kind, s, e, tk in docs:
        tf = collections.Counter(tk)
        dl = len(tk)
        sc = 0.0
        for t, w in qc.items():
            if t in tf:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                sc += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl)) * min(w, 3)
        if qn.split(".")[-1].split("#")[0] in exact:
            sc *= 2.5
        if any(rel.endswith(p) or p.endswith(rel) for p in paths_in_q):
            sc *= 2.0
        elif re.match(r"(scripts|docs|benchmarks?|examples?|tools)/", rel):
            sc *= 0.5
        if kind != "module" and e - s > 150:
            sc *= 0.85
        res.append((sc, rel, qn, kind, s, e))
    res.sort(key=lambda x: -x[0])
    print("TOP SYMBOLS (ref file:lines name)")
    shown = 0
    for sc, rel, qn, kind, s, e in res:
        if sc <= 0 or shown >= 14:
            break
        if qn == "<module>":
            continue
        print("  %s %s:%d-%d %s%s" % (new_ref(st, rel, qn, kind, s, e), rel, s, e, qn, " (class)" if kind == "class" else ""))
        shown += 1
    best_file = collections.OrderedDict()
    for sc, rel, qn, kind, s, e in res:
        if sc > 0 and rel not in best_file:
            best_file[rel] = sc
    print("TOP FILES")
    for rel in list(best_file)[:6]:
        print("  " + rel)
    mods = {os.path.splitext(os.path.basename(r))[0] for r in list(best_file)[:4]}
    names = {qn.split(".")[-1].split("#")[0] for sc, rel, qn, kind, s, e in res[:8] if qn != "<module>"}
    rt = sorted(((sum(1 for m in mods if m in rel or re.search(r"\b" + re.escape(m) + r"\b", src[:3000])) + sum(1 for n in names if n in src), rel) for rel, src in tests), reverse=True)
    print("RELATED TESTS")
    for h, rel in rt[:5]:
        if h:
            print("  " + rel)


def numbered(rel, s, e, cap=140):
    lines = read(rel).split("\n")
    s, e = max(1, s), min(e, len(lines))
    if e - s + 1 > cap:
        for i in range(s, s + cap - 20):
            print("%5d  %s" % (i, lines[i - 1]))
        print("  ...  lines %d-%d hidden; view them with show %s:%d-%d" % (s + cap - 20, e - 16, rel, s + cap - 20, e - 16))
        for i in range(e - 15, e + 1):
            print("%5d  %s" % (i, lines[i - 1]))
    else:
        for i in range(s, e + 1):
            print("%5d  %s" % (i, lines[i - 1]))


def cmd_show(st, target):
    k, r = resolve(st, target)
    if k is None:
        if "range" in r:
            print("%s lines %d-%d" % (r["file"], r["range"][0], r["range"][1]))
            return numbered(r["file"], r["range"][0], r["range"][1], cap=200)
        return cmd_outline(st, r["file"])
    print("%s %s %s lines %d-%d" % (k, r["file"], r["qual"], r["start"], r["end"]))
    if r["kind"] == "class" and r["end"] - r["start"] > 60:
        numbered(r["file"], r["start"], min(r["end"], r["start"] + 25))
        print("  methods:")
        for q, kind, s, e, *_ in symbols(r["file"]):
            if q.startswith(r["qual"] + ".") and q.count(".") == r["qual"].count(".") + 1:
                print("    %s %s lines %d-%d" % (new_ref(st, r["file"], q, kind, s, e), q, s, e))
    else:
        numbered(r["file"], r["start"], r["end"])
    if r["kind"] != "class" and r["end"] - r["start"] + 1 > MAX_REPLACE:
        print("note: %d lines; edit inside it with apply.py SEARCH/REPLACE, not cx replace" % (r["end"] - r["start"] + 1))


def cmd_outline(st, rel):
    rel = rel.replace("/workspace/", "")
    if not os.path.exists(os.path.join(WS, rel)):
        sys.exit("no such file: " + rel)
    syms = symbols(rel)
    n = len(read(rel).split("\n"))
    print("%s (%d lines)" % (rel, n))
    if not syms:
        print("  no functions or classes; read it with show %s:1-%d" % (rel, min(n, 200)))
    for q, kind, s, e, *_ in syms:
        print("  %s%s %s lines %d-%d%s" % ("  " * q.count("."), new_ref(st, rel, q, kind, s, e), q, s, e, " (class)" if kind == "class" else ""))


def compiles(text, path):
    try:
        compile(text, path, "exec")
        return None
    except SyntaxError as ex:
        return "line %s: %s" % (ex.lineno, ex.msg)


def splice_checked(rel, lines, a, b, new_lines, note):
    """Replace lines[a:b] with new_lines; refuse if the file would not compile."""
    path = os.path.join(WS, rel)
    old = "\n".join(lines)
    text = "\n".join(lines[:a] + new_lines + lines[b:])
    err = compiles(text, path)
    if err:
        print("NOT APPLIED: the file would not compile (%s). Check brackets, quotes and indentation; use real line breaks." % err)
        sys.exit(1)
    open(path, "w").write(text)
    _SYM_CACHE.pop(rel, None)
    LAST_EDIT[rel] = (a, b, len(new_lines))
    d = "".join(difflib.unified_diff(old.splitlines(True), text.splitlines(True), "a/" + rel, "b/" + rel, n=2))
    print("OK " + note)
    print(d[:3000] + ("\n...diff truncated" if len(d) > 3000 else ""))


def string_spans_text(src):
    """[(start, end)] character offsets of string literals (multi-line aware), via the same scanner as code_positions."""
    spans, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == "#":
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        if c in "\"'":
            q3 = src[i:i + 3]
            if q3 in ('"""', "\'\'\'"):
                j = i + 3
                while j < n and not src.startswith(q3, j):
                    j += 2 if src[j] == "\\" else 1
                spans.append((i, j + 3)); i = j + 3
                continue
            j = i + 1
            while j < n and src[j] not in (c, "\n"):
                j += 2 if src[j] == "\\" else 1
            spans.append((i, j + 1)); i = j + 1
            continue
        i += 1
    return spans


def prepare_code(code, indent):
    """Re-indent new code to `indent`, leaving lines that continue a multi-line string exactly as written."""
    code = code.replace("\r\n", "\n").strip("\n")
    code = re.sub(r"^[ \t]*```[a-zA-Z]*\n|\n[ \t]*```\s*$", "", code)
    lines = code.split("\n")
    spans = string_spans_text(code)
    starts, off = [], 0
    for l in lines:
        starts.append(off)
        off += len(l) + 1
    inside = [any(a < st_ < b for a, b in spans) for st_ in starts]
    widths = [len(l) - len(l.lstrip()) for l, ins in zip(lines, inside) if l.strip() and not ins]
    d = min(widths) if widths else 0
    out = []
    for l, ins in zip(lines, inside):
        if ins:
            out.append(l)
        elif l.strip():
            out.append(" " * indent + l[d:])
        else:
            out.append("")
    return out


def code_positions(src):
    """Indexes of characters that are code: outside string literals and comments (multi-line strings aware)."""
    pos, i, n = set(), 0, len(src)
    while i < n:
        c = src[i]
        if c == "#":
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        if c in "\"'":
            q3 = src[i:i + 3]
            if q3 in ('"""', "\'\'\'"):
                j = i + 3
                while j < n and not src.startswith(q3, j):
                    j += 2 if src[j] == "\\" else 1
                i = j + 3
                continue
            j = i + 1
            while j < n and src[j] not in (c, "\n"):
                j += 2 if src[j] == "\\" else 1
            i = j + 1
            continue
        pos.add(i)
        i += 1
    return pos


def repaired(code):
    """If the new code does not compile but would with literal \\n (outside strings) turned into line breaks, fix it."""
    probe = textwrap.dedent(code).strip("\n")
    if compiles(probe, "<new>") is None or "\\n" not in code:
        return code, ""
    ok = code_positions(code)
    out, i = [], 0
    while i < len(code):
        if code.startswith("\\n", i) and i in ok and (i + 2 == len(code) or code[i + 2] in " \t\n"):
            out.append("\n")
            i += 2
            continue
        out.append(code[i])
        i += 1
    fixed = "".join(out)
    if compiles(textwrap.dedent(fixed).strip("\n"), "<new>") is None:
        return fixed, " (literal \\n in your code turned into line breaks)"
    return code, ""


def defined(code):
    """(kind, name, has_decorators) of the first top-level statement of the new code, or Nones."""
    try:
        tree = ast.parse(textwrap.dedent(code).strip("\n"))
    except SyntaxError:
        m = re.search(r"^(?:@[^\n]*\n)*(async[ \t]+def|def|class)[ \t]+([A-Za-z_][A-Za-z0-9_]*)", textwrap.dedent(code).strip("\n"))
        return ((("async def" if m.group(1).startswith("async") else m.group(1)), m.group(2), None) if m else (None, None, None))
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None, None, None
    n = tree.body[0]
    kind = "class" if isinstance(n, ast.ClassDef) else ("async def" if isinstance(n, ast.AsyncFunctionDef) else "def")
    return kind, n.name, bool(n.decorator_list)


def cmd_replace(st, target, code):
    k, r = resolve(st, target)
    if k is None:
        sys.exit("replace needs a function or class ref (see find/outline)")
    rel = r["file"]
    if r["end"] - r["start"] + 1 > MAX_REPLACE and r["kind"] != "class":
        sys.exit("NOT APPLIED: %s is %d lines; change only the lines you need with apply.py SEARCH/REPLACE" % (r["qual"], r["end"] - r["start"] + 1))
    code, note = repaired(code)
    kind, name, has_dec = defined(code)
    old_name = r["qual"].split(".")[-1].split("#")[0]
    if kind is None:
        sys.exit("NOT APPLIED: the new code must start with the complete definition (def/class %s), not other statements" % old_name)
    if kind != r["kind"] or name != old_name:
        sys.exit("NOT APPLIED: %s is a %s named %s, but your code defines a %s named %s. Use the right ref (outline %s), or insert to add new code." % (k, r["kind"], old_name, kind, name, rel))
    lines = read(rel).split("\n")
    def_line = by_qual(rel, r["qual"])[4]
    indent = len(lines[def_line - 1]) - len(lines[def_line - 1].lstrip())
    new = prepare_code(code, indent)
    old_decorators = lines[r["start"] - 1:def_line - 1]
    if old_decorators and not has_dec:
        new = old_decorators + new
        note += " (kept the existing decorators)"
    splice_checked(rel, lines, r["start"] - 1, r["end"], new, "replaced %s %s (lines %d-%d)%s" % (k, r["qual"], r["start"], r["end"], note))
    refresh_refs(st, rel, LAST_EDIT[rel], edited_ref=k)


def cmd_insert(st, target, where, code):
    k, r = resolve(st, target)
    if k is None:
        sys.exit("insert needs a function or class ref")
    rel = r["file"]
    code, note = repaired(code)
    lines = read(rel).split("\n")
    def_line = by_qual(rel, r["qual"])[4]
    def_indent = len(lines[def_line - 1]) - len(lines[def_line - 1].lstrip())
    if where == "into":
        if r["kind"] != "class":
            sys.exit("insert into needs a class ref")
        xx = by_qual(rel, r["qual"])
        body_line = xx[5]
        if body_line == def_line or xx[7]:
            sys.exit("NOT APPLIED: class %s is written on one line; replace it with cx replace instead" % r["qual"])
        indent = len(lines[body_line - 1]) - len(lines[body_line - 1].lstrip())
        at = r["end"]
    else:
        indent = def_indent
        at = r["end"] if where == "after" else r["start"] - 1
    new = [""] + prepare_code(code, indent) + ([""] if where != "into" else [])
    splice_checked(rel, lines, at, at, new, "inserted %s %s%s" % (where, r["qual"], note))
    refresh_refs(st, rel, LAST_EDIT[rel])


def cmd_blocks(st, text, argv):
    text = text.replace("\r\n", "\n")
    pat = re.compile(r"^[ \t]*(REPLACE|INSERT AFTER|INSERT BEFORE|INSERT INTO)[ \t]+([^\n]+?)[ \t]*\n(.*?)(?:\n[ \t]*END[ \t]*$|\Z)", re.S | re.M)
    found = [(m.group(1), m.group(2), m.group(3)) for m in pat.finditer(text)]
    if not found:
        words = [w for w in argv if w.upper() not in ("REPLACE", "INSERT", "AFTER", "BEFORE", "INTO")]
        up = [w.upper() for w in argv]
        kind = "INSERT AFTER" if "AFTER" in up else ("INSERT BEFORE" if "BEFORE" in up else ("INSERT INTO" if "INTO" in up else "REPLACE"))
        code = text.strip("\n")
        target = " ".join(words) or (defined(code)[1] or "")
        if not target or not code.strip():
            sys.exit('no blocks found; use a line "REPLACE @3" (or "INSERT AFTER @3", "INSERT INTO @5"), then the code, then "END"')
        found = [(kind, target, code)]
    for kind, target, code in found:
        target = target.strip().strip("`")
        m = re.match(r"^([\w./-]+\.py)[ \t:]+(\d+|[\w.#]+)$", target)
        if m:
            target = "%s:%s" % (m.group(1), m.group(2))
        if kind == "REPLACE":
            cmd_replace(st, target, code)
        else:
            cmd_insert(st, target, kind.split()[1].lower(), code)


def cmd_callers(st, target):
    k, r = resolve(st, target)
    name = r["qual"].split(".")[-1].split("#")[0] if k else os.path.splitext(os.path.basename(r["file"]))[0]
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    n = 0
    for rel, p in py_files():
        try:
            for i, l in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                if pat.search(l) and not re.match(r"\s*(async\s+def|def|class)\s+" + re.escape(name) + r"\b", l):
                    print("  %s:%d  %s" % (rel, i, l.strip()[:120]))
                    n += 1
                    if n >= 25:
                        print("  ...")
                        return
        except Exception:
            pass
    if not n:
        print("  no uses found")


def cmd_tests(st, target, run, kexpr):
    k, r = resolve(st, target)
    rel = r["file"]
    name = r["qual"].split(".")[-1].split("#")[0] if k else None
    mod = os.path.splitext(os.path.basename(rel))[0]
    scored = []
    for t, p in py_files():
        if not TEST_RE.search(t):
            continue
        src = open(p, encoding="utf-8", errors="replace").read()
        s = (3 if os.path.basename(t) in ("test_%s.py" % mod, "%s_test.py" % mod) else 0) + (2 if name and re.search(r"\b" + re.escape(name) + r"\b", src) else 0) + (1 if re.search(r"\b" + re.escape(mod) + r"\b", src[:4000]) else 0)
        if s:
            scored.append((s, t))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    for s, t in scored[:6]:
        print("  " + t)
    if not run:
        return 0
    files = [t for _, t in scored[:2]]
    if not files:
        print("RESULT: NO TESTS FOUND (write a quick python3 -c check instead)")
        return 3
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--no-header", "-rfE"] + files + (["-k", kexpr] if kexpr else [])
    try:
        p = subprocess.run(cmd, cwd=WS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=150)
        out, rc = p.stdout.strip().split("\n"), p.returncode
    except subprocess.TimeoutExpired:
        out, rc = ["TIMEOUT after 150s"], 124
    print("RAN:", " ".join(cmd[2:]))
    keep = [l for l in out if re.search(r"(Error|assert|FAILED|^E )", l)][:25] if len(out) > 60 else out
    print("\n".join(keep + (out[-12:] if len(out) > 60 else [])))
    verdict = {0: "PASSED", 1: "FAILED", 5: "NO TESTS COLLECTED", 124: "TIMEOUT"}.get(rc, "ERROR")
    print("RESULT: %s (pytest exit %d)" % (verdict, rc))
    return 0 if rc == 0 else 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 0
    st = load_state()
    c, args = sys.argv[1], sys.argv[2:]
    rc = 0
    try:
        if c == "find":
            cmd_find(st, " ".join(args) if args else sys.stdin.read())
        elif c == "outline" and args:
            cmd_outline(st, args[0])
        elif c == "show" and args:
            cmd_show(st, args[0])
        elif c == "replace" and args:
            cmd_replace(st, args[0], sys.stdin.read())
        elif c == "insert" and len(args) >= 2 and args[1] in ("after", "before", "into"):
            cmd_insert(st, args[0], args[1], sys.stdin.read())
        elif c == "blocks":
            cmd_blocks(st, sys.stdin.read(), args)
        elif c == "callers" and args:
            cmd_callers(st, args[0])
        elif c == "tests" and args:
            kexpr = args[args.index("-k") + 1] if "-k" in args and args.index("-k") + 1 < len(args) else None
            rc = cmd_tests(st, args[0], "--run" in args, kexpr)
        elif c == "diff":
            out = subprocess.run(["git", "diff", "--stat"], cwd=WS, stdout=subprocess.PIPE, universal_newlines=True).stdout
            d = subprocess.run(["git", "diff"], cwd=WS, stdout=subprocess.PIPE, universal_newlines=True).stdout
            print((out + d[:4000]) or "no changes yet")
        else:
            print(__doc__)
    finally:
        save_state(st)
    return rc


if __name__ == "__main__":
    sys.exit(main())
