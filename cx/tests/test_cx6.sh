#!/bin/bash
# Review pass 3 cases: identical same-name definitions; string content preserved through re-indentation.
PY=${PY:-python3}
W=/tmp/cx6ws
rm -rf $W /tmp/t/cx_refs.json && mkdir -p $W/pkg && cd $W && git init -q && git config user.email t@t && git config user.name t
cat > pkg/m.py <<'SRC'
if True:
    def helper():
        return 1
else:
    def helper():
        return 1

class Item:
    def text(self):
        return "x"
SRC
git add -A && git commit -qm base
sed "s#WS = \"/workspace\"#WS = \"$W\"#" ~/g4/src/tools/cx.py > /tmp/cx6.py
CX="$PY /tmp/cx6.py"
$CX outline pkg/m.py
echo "=== identical definitions: replace the first listed ref, expect line 3 changed"
FIRST=$($PY -c "import json;s=json.load(open('/tmp/t/cx_refs.json'));print([k for k,v in s['refs'].items() if v['qual']=='helper'][0])")
printf 'def helper():\n    return 111\n' | $CX replace $FIRST | head -1
grep -n "return 1" pkg/m.py
echo "=== string content preserved when a method written at column 0 is re-indented"
printf 'def text(self):\n    s = """line one\nline two"""\n    return s\n' | $CX replace Item.text | head -1
$PY -c "import sys; sys.path.insert(0, '$W'); import pkg.m as m; print(repr(m.Item().text()))"
echo "=== same, with repair of a literal separator outside the string"
printf 'def text(self):\n    s = r"""a\nb\\nc"""\\n    return s\n' | $CX replace Item.text | head -1
$PY -c "import sys, importlib; sys.path.insert(0, '$W'); import pkg.m as m; importlib.reload(m); print(repr(m.Item().text()))"
echo "=== Review pass 4 case: identical twins, insert a different helper before the first, replace the saved ref of the original second"
rm -rf $W /tmp/t/cx_refs.json && mkdir -p $W/pkg && cd $W && git init -q && git config user.email t@t && git config user.name t
printf 'if True:\n    def helper():\n        return 1\nelse:\n    def helper():\n        return 1\n' > pkg/m.py && git add -A && git commit -qm base
$CX outline pkg/m.py >/dev/null
FIRST=$($PY -c "import json;s=json.load(open('/tmp/t/cx_refs.json'));print([k for k,v in s['refs'].items() if v['qual']=='helper'][0])")
SECOND=$($PY -c "import json;s=json.load(open('/tmp/t/cx_refs.json'));print([k for k,v in s['refs'].items() if v['qual']=='helper#2'][0])")
printf 'def helper():\n    return 0\n' | $CX insert $FIRST before | head -1
printf 'def helper():\n    return 222\n' | $CX replace $SECOND | head -1
echo "expect return 0, return 1, return 222:"; grep -n "return" pkg/m.py
echo "=== Review pass 5 case: replace a whole class; refs to its old methods become stale, not retargeted"
rm -rf $W /tmp/t/cx_refs.json && mkdir -p $W/pkg && cd $W && git init -q && git config user.email t@t && git config user.name t
printf 'class C:\n    def f_int(self, x: int):\n        return 1\n    def f_str(self, x: str):\n        return 2\n' > pkg/m.py && git add -A && git commit -qm base
$CX outline pkg/m.py >/dev/null
CLS=$($PY -c "import json;s=json.load(open('/tmp/t/cx_refs.json'));print([k for k,v in s['refs'].items() if v['qual']=='C'][0])")
STR=$($PY -c "import json;s=json.load(open('/tmp/t/cx_refs.json'));print([k for k,v in s['refs'].items() if v['qual']=='C.f_str'][0])")
printf 'class C:\n    def f_bytes(self, x: bytes):\n        return 0\n    def f_int(self, x: int):\n        return 1\n    def f_str(self, x: str):\n        return 2\n' | $CX replace $CLS | head -1
printf 'def f_str(self, x: str):\n    return 99\n' | $CX replace $STR; echo "exit=$?"
$CX outline pkg/m.py | head -5
