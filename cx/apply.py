"""Apply SEARCH/REPLACE blocks read from stdin (or argv[1] file) to files under /workspace.
Block format (repeatable):
FILE: relative/path.py
<<<<<<< SEARCH
old lines
=======
new lines
>>>>>>> REPLACE
Matching: exact, then whitespace-insensitive per line, then fuzzy (difflib, unique best >= 0.82). Empty SEARCH creates a new file.
Each Python edit is compiled; if it fails and the REPLACE text holds a literal backslash-n where a line break was meant,
the edit is retried with real line breaks. An edit that still does not compile is not applied and the error is printed.
Blocks copied from cx show keep its line-number column ("%5d  "). A SEARCH whose lines mostly carry it is located without
it first (ambiguity there fails the block), then as written; REPLACE is tried without it first when SEARCH needed that,
otherwise only if it does not compile. Failed blocks are counted in /tmp/t/apply_failed.json by block and file contents;
the file outlives context compaction, and an identical block that failed before on the same contents gets a note."""
import difflib, hashlib, json, os, re, sys, textwrap
WS = "/workspace"
raw = open(sys.argv[1]).read() if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else sys.stdin.read()
raw = textwrap.dedent(raw.replace("\r\n", "\n"))
end = raw.rfind(">>>>>>> REPLACE")
if end != -1:
    raw = raw[: end + len(">>>>>>> REPLACE")]
pat = re.compile(r"^[ \t]*FILE:[ \t]*(?P<f>[^\n]+)\n[ \t]*<<<<<<< SEARCH[ \t]*\n(?P<s>.*?)\n?[ \t]*=======[ \t]*\n(?P<r>.*?)\n?[ \t]*>>>>>>> REPLACE", re.S | re.M)
blocks = list(pat.finditer(raw))
if not blocks:
    print("ERROR: no blocks found. Use:\nFILE: path\n<<<<<<< SEARCH\n...\n=======\n...\n>>>>>>> REPLACE"); sys.exit(1)


def norm(l):
    return " ".join(l.split())


CXNUM = re.compile(r"^ *\d+  ")
LEDGER = "/tmp/t/apply_failed.json"


def unnumber(block):
    """block without cx show's line-number column, or None when most of its lines do not carry one"""
    ls = [l for l in block.split("\n") if l.strip()]
    if not ls or sum(1 for l in ls if CXNUM.match(l)) < 0.6 * len(ls):
        return None
    return "\n".join(CXNUM.sub("", l, count=1) for l in block.split("\n"))


def locate(text, search):
    if not search.strip():
        return None, "empty"
    n = text.count(search)
    if n == 1:
        i = text.index(search)
        return (i, i + len(search)), "exact"
    if n > 1:
        return None, "SEARCH text occurs %d times; add more lines to make it unique" % n
    lines = text.split("\n")
    sl = search.split("\n")
    while sl and not sl[0].strip():
        sl.pop(0)
    while sl and not sl[-1].strip():
        sl.pop()
    k = len(sl)
    ns = [norm(l) for l in sl]
    hits = [i for i in range(len(lines) - k + 1) if [norm(l) for l in lines[i:i + k]] == ns]
    how = "whitespace"
    if len(hits) == 1:
        i = hits[0]
    elif len(hits) > 1:
        return None, "SEARCH matches %d places (ignoring whitespace); add more lines" % len(hits)
    else:
        best = []
        joined = "\n".join(ns)
        for w in (k, k - 1, k + 1):
            if w <= 0:
                continue
            for j in range(len(lines) - w + 1):
                r = difflib.SequenceMatcher(None, joined, "\n".join(norm(l) for l in lines[j:j + w])).ratio()
                best.append((r, j, w))
        best.sort(reverse=True)
        if not best or best[0][0] < 0.82 or (len(best) > 1 and best[1][0] > best[0][0] - 0.03 and best[1][1] != best[0][1]):
            cand = "\n".join("  lines %d-%d (similarity %.2f)" % (j + 1, j + w, r) for r, j, w in best[:2])
            return None, "SEARCH not found. Closest regions:\n" + cand
        i, k = best[0][1], best[0][2]
        how = "fuzzy"
    start = sum(len(l) + 1 for l in lines[:i])
    stop = start + len("\n".join(lines[i:i + k]))
    return (start, stop), how


def reindent(old_region, new):
    first = next((l for l in old_region.split("\n") if l.strip()), "")
    nfirst = next((l for l in new.split("\n") if l.strip()), "")
    d = len(first) - len(first.lstrip()) - (len(nfirst) - len(nfirst.lstrip()))
    if d <= 0:
        return new
    return "\n".join((" " * d + l) if l.strip() else l for l in new.split("\n"))


def compiles(path, text):
    if not path.endswith(".py"):
        return None
    try:
        compile(text, path, "exec")
        return None
    except SyntaxError as e:
        return "line %s: %s" % (e.lineno, e.msg)


def variants(repl, unnumbered_first=False):
    u = unnumber(repl)
    for r, note in ([(u, " (line numbers removed from REPLACE)"), (repl, "")] if unnumbered_first else [(repl, ""), (u, " (line numbers removed from REPLACE)")]):
        if r is None:
            continue
        yield r, note
        fixed = re.sub(r"\\n(?=[ \t]|$)", "\n", r, flags=re.M)
        if fixed != r:
            yield fixed, note + " (literal \\n in REPLACE turned into line breaks)"
        fixed2 = r.replace("\\n", "\n")
        if fixed2 not in (r, fixed):
            yield fixed2, note + " (all literal \\n in REPLACE turned into line breaks)"


try:
    seen = json.load(open(LEDGER))
except Exception:
    seen = {}


def fail(rel, msg, key):
    print("FAIL %s: %s" % (rel, msg))
    n = seen.get(key, 0)
    seen[key] = n + 1
    if n:
        print("NOTE: you sent this exact edit %d time%s before and it failed. Do not send it again unchanged: "
              "show the current code (python3 /tmp/t/cx.py show %s:START-END) and write the edit from what it prints." % (n, "" if n == 1 else "s", rel))


ok = failed = 0
originals = {}
for b in blocks:
    rel = b.group("f").strip()
    rel = (rel[len("/workspace/"):] if rel.startswith("/workspace/") else rel).lstrip("/")
    path = os.path.join(WS, rel)
    search, repl = b.group("s"), b.group("r")
    state = hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.isfile(path) else "<missing>"
    key = hashlib.md5("\0".join((rel, search, repl, state)).encode()).hexdigest()
    if not search.strip():
        if os.path.exists(path) and open(path).read().strip():
            fail(rel, "empty SEARCH but the file exists; copy lines to replace into SEARCH", key); failed += 1; continue
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        body = repl if repl.endswith("\n") else repl + "\n"
        err = compiles(path, body)
        if err:
            fail(rel, "new file would not compile (%s)" % err, key); failed += 1; continue
        originals.setdefault(path, open(path).read() if os.path.exists(path) else None)
        open(path, "w").write(body)
        print("OK %s: created" % rel); ok += 1; continue
    if not os.path.exists(path):
        fail(rel, "file not found", key); failed += 1; continue
    text = open(path).read()
    u, snote = unnumber(search), ""
    span = None
    if u is not None:
        span, how = locate(text, u)
        if span is not None:
            snote = " (line numbers removed from SEARCH)"
        elif how.startswith(("SEARCH text occurs", "SEARCH matches")):
            fail(rel, how + " (line numbers removed from SEARCH)", key); failed += 1; continue
        else:
            how_u = how
    if span is None:
        span, how = locate(text, search)
        if span is None and u is not None:
            how = how_u + " (tried with the line numbers removed from SEARCH)"
    if span is None:
        fail(rel, how, key); failed += 1; continue
    region = text[span[0]:span[1]]
    err = None
    for cand, note in variants(repl, unnumbered_first=bool(snote)):
        new = cand if how == "exact" else reindent(region, cand)
        new_text = text[:span[0]] + new + text[span[1]:]
        err = compiles(path, new_text)
        if err is None:
            originals.setdefault(path, text)
            open(path, "w").write(new_text)
            print("OK %s: %s match%s%s" % (rel, how, snote, note)); ok += 1
            break
    else:
        fail(rel, "edit not applied, the result would not compile (%s). Check brackets and indentation; use real line breaks, never the two characters \\n." % err, key)
        failed += 1

for path, before in originals.items():
    after = open(path).read()
    d = "".join(difflib.unified_diff((before or "").splitlines(True), after.splitlines(True),
                                     "a/" + os.path.relpath(path, WS), "b/" + os.path.relpath(path, WS), n=2))
    print(d[:2500] + ("\n...diff truncated" if len(d) > 2500 else ""))
try:
    json.dump(seen, open(LEDGER, "w"))
except Exception:
    pass
print("SUMMARY applied=%d failed=%d" % (ok, failed))
sys.exit(1 if failed else 0)
