#!/bin/bash
# Reproduction cases for cx v3 -> v4.
PY=${PY:-python3}
W=/tmp/cx5ws
rm -rf $W /tmp/t/cx_refs.json && mkdir -p $W/pkg && cd $W && git init -q && git config user.email t@t && git config user.name t
cat > pkg/m.py <<'SRC'
FLAG = True
if FLAG:
    def helper():
        return 1
else:
    def helper():
        return 2

class Box(
    object,
): pass

class Item:
    def value(self):
        return 1
SRC
git add -A && git commit -qm base
sed "s#WS = \"/workspace\"#WS = \"$W\"#" ~/g4/src/tools/cx.py > /tmp/cx5.py
CX="$PY /tmp/cx5.py"
$CX outline pkg/m.py
A=$($PY -c "import json;s=json.load(open('/tmp/t/cx_refs.json'));print([k for k,v in s['refs'].items() if v['qual']=='helper'][0])")
B=$($PY -c "import json;s=json.load(open('/tmp/t/cx_refs.json'));print([k for k,v in s['refs'].items() if v['qual']=='helper#2'][0])")
echo "=== 1. renumbering: insert a new helper before $A, then replace old second helper via $B"
printf 'def helper():\n    return 0\n' | $CX insert $A before | head -1
printf 'def helper():\n    return 22\n' | $CX replace $B | head -1
echo "expect: return 0, return 1, return 22 in that order:"; grep -n "return [0-9]" pkg/m.py | head -3
echo "=== 2. repair outside strings only (raw multi-line string keeps its \\n)"
printf 'def value(self):\n    s = r"""a\nb\\nc"""\\n    return s\n' | $CX replace Item.value | head -1
sed -n '/def value/,/return s/p' pkg/m.py
echo "=== 3. insert into class whose body shares the header line"
printf 'def m(self):\n    return 1\n' | $CX insert pkg/m.py:Box into; echo "exit=$?"
echo "=== 4. bare Class.method lookup"
$CX show Item.value | head -2
$PY -c "import ast;ast.parse(open('$W/pkg/m.py').read());print('file parses')"
