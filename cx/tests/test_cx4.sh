#!/bin/bash
# Extra regression cases from a verification pass.
PY=${PY:-python3}
W=/tmp/cx4ws
rm -rf $W /tmp/t/cx_refs.json && mkdir -p $W/pkg && cd $W && git init -q && git config user.email t@t && git config user.name t
cat > pkg/m.py <<'SRC'
import functools

class Box(
    object,
):
    """doc"""

    @property
    def value(self):
        return 1

    @value.setter
    def value(self, v):
        self._v = v

async def fetch(x):
    return x

@functools.wraps(
    print,
)
def wrapped(a):
    return call(
        a,
    )

def call(a):
    return a
SRC
git add -A && git commit -qm base
sed "s#WS = \"/workspace\"#WS = \"$W\"#" ~/g4/src/tools/cx.py > /tmp/cx4.py
sed "s#WS = \"/workspace\"#WS = \"$W\"#" ~/g4/src/tools/apply.py > /tmp/apply4.py
CX="$PY /tmp/cx4.py"
$CX outline pkg/m.py
SET=$($PY -c "import json;s=json.load(open('/tmp/t/cx_refs.json'));print([k for k,v in s['refs'].items() if v['qual']=='Box.value#2'][0])")
echo "=== ref drift: insert another value getter BEFORE the setter, then replace setter ref $SET"
printf '@property\ndef other(self):\n    return 2\n' | $CX insert pkg/m.py:Box.value before | head -1
printf 'def value(self, v):\n    self._v = v * 10\n' | $CX replace $SET | head -1
grep -n "v \* 10" pkg/m.py; grep -n -B1 "def value(self, v)" pkg/m.py | head -2
echo "=== async kind check"
printf 'def fetch(x):\n    return x\n' | $CX replace pkg/m.py:fetch; echo "exit=$?"
printf 'async def fetch(x):\n    return x + 1\n' | $CX replace pkg/m.py:fetch | head -1
echo "=== multiline decorator kept + closing paren boundary"
printf 'def wrapped(a):\n    return call(a)\n' | $CX replace pkg/m.py:wrapped | head -1
sed -n '/functools.wraps/,/^def call/p' pkg/m.py
echo "=== insert into multi-line class header"
printf 'def triple(self):\n    return 3\n' | $CX insert pkg/m.py:Box into | head -1
$PY -c "import sys; sys.path.insert(0, '$W'); import pkg.m as m; print('triple', m.Box().triple())"
echo "=== repair keeps string literals"
printf 'def call(a):\n    s = "x\\\\ny"\\n    return a\n' | $CX replace pkg/m.py:call | head -1
grep -n 's = ' pkg/m.py
echo "=== apply rejects bad new file without crashing"
printf 'FILE: pkg/new.py\n<<<<<<< SEARCH\n=======\ndef broken(:\n>>>>>>> REPLACE\n' | $PY /tmp/apply4.py; echo "exit=$?"; ls pkg/new.py 2>&1 | head -1
