#!/bin/bash
# Regression tests for the cx issues found in review (run with python3 and python:3.7).
PY=${PY:-python3}
W=/tmp/cx3ws
rm -rf $W /tmp/t/cx_refs.json && mkdir -p $W/pkg $W/tests && cd $W && git init -q && git config user.email t@t && git config user.name t
cat > pkg/mod.py <<'SRC'
import functools

class Box:
    def __init__(self, v):
        self._v = v

    @property
    def value(self):
        return self._v

    @value.setter
    def value(self, v):
        self._v = v

if True:
    def under_if(x):
        return x

try:
    def under_try(x):
        return x
except Exception:
    pass

def with_comment(x):
    y = x
# a column-zero comment inside the function
    return y

def with_docstring(x):
    """Docstring
    spanning lines."""
    s = """multi
line string at end"""
    return s

@functools.lru_cache()
def cached(x):
    return "keep\n literal"
SRC
python3 - <<'GEN' >> pkg/mod.py

print("\ndef long_one(x):")
for i in range(130):
    print("    x = x + %d" % i)
print("    return x")
GEN
cat > tests/test_mod.py <<'SRC'
from pkg.mod import under_if
def test_under_if():
    assert under_if(2) == 2
SRC
git add -A && git commit -qm base
sed "s#WS = \"/workspace\"#WS = \"$W\"#" ~/g4/src/tools/cx.py > /tmp/cx3.py
CX="$PY /tmp/cx3.py"
echo "=== outline (setter separate, if/try visible, 3.7 ends)"; $CX outline pkg/mod.py
echo "=== replace setter by its ref"
R=$($PY -c "import json;s=json.load(open('/tmp/t/cx_refs.json'));print([k for k,v in s['refs'].items() if v['qual']=='Box.value#2'][0])")
printf 'def value(self, v):\n    self._v = int(v)\n' | $CX replace $R | head -3
grep -n "int(v)" pkg/mod.py && grep -n "@value.setter" pkg/mod.py
echo "=== decorator kept when omitted"
printf 'def cached(x):\n    return "keep\\n literal changed"\n' | $CX replace pkg/mod.py:cached | head -2
grep -n -B1 "def cached" pkg/mod.py | head -3
echo "=== kind/name mismatch rejected"
printf 'def other(x):\n    return 1\n' | $CX replace pkg/mod.py:under_if; echo "exit=$?"
echo "=== long function refused"
printf 'def long_one(x):\n    return x\n' | $CX replace pkg/mod.py:long_one; echo "exit=$?"
echo "=== insert into class -> method"
printf 'def double(self):\n    return self._v * 2\n' | $CX insert pkg/mod.py:Box into | head -2
$PY -c "import sys; sys.path.insert(0, '$W'); import importlib, pkg.mod as m; importlib.reload(m); print('method ok:', m.Box(3).double())"
echo "=== literal-\\n repair only in new code; existing literal untouched"
printf 'def under_try(x):\\n    return x + 1\n' | $CX replace pkg/mod.py:under_try | head -2
grep -c 'keep\\n literal changed' pkg/mod.py
echo "=== tests --run reports result and exit code"
$CX tests pkg/mod.py:under_if --run | tail -2; echo "exit=$?"
echo "=== show range"; $CX show pkg/mod.py:1-3
