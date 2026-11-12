#!/bin/bash
# cx.py integration test on a copy of the rich_4077 snapshot.
set -u
rm -rf /tmp/cxws /tmp/t/cx_refs.json && cp -r ~/g4/snapcache/rich_4077 /tmp/cxws && cd /tmp/cxws && git checkout -q -- . 2>/dev/null
sed 's#WS = "/workspace"#WS = "/tmp/cxws"#' ~/g4/src/tools/cx.py > /tmp/cx_test.py
CX="python3 /tmp/cx_test.py"
echo "=== find"; $CX find "Console.print_json indent highlight" | head -10
echo "=== outline"; $CX outline rich/pretty.py | head -8
echo "=== show by qualname"; $CX show Console.print_json | head -6
REF=$(python3 -c "import json;s=json.load(open('/tmp/t/cx_refs.json'));print([k for k,v in s['refs'].items() if v['qual']=='Console.print_json'][0])")
echo "ref for Console.print_json: $REF"
echo "=== replace (method body, written without class indentation)"
python3 - <<'PY' > /tmp/newdef.txt
import ast
src = open('/tmp/cxws/rich/console.py').read()
t = ast.parse(src)
for n in ast.walk(t):
    if isinstance(n, ast.FunctionDef) and n.name == 'print_json':
        lines = src.split('\n')[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno]
        import textwrap
        body = textwrap.dedent('\n'.join(lines))
        print(body.replace('"""', '"""Marker added by cx test. ', 1))
        break
PY
$CX replace $REF < /tmp/newdef.txt | head -12
python3 -c "import ast;ast.parse(open('/tmp/cxws/rich/console.py').read());print('console.py still parses')"
echo "=== compile guard"
printf 'def print_json(self:\n    pass\n' | $CX replace $REF; echo "exit=$?"
git -C /tmp/cxws diff --stat
echo "=== insert after"
printf 'def _cx_helper():\n    return 42\n' | $CX insert $REF after | head -8
echo "=== callers"; $CX callers $REF | head -5
echo "=== tests"; $CX tests $REF | head -4
echo "=== py3.7"
docker run -i --rm -v /tmp/cxws:/tmp/cxws -v /tmp/cx_test.py:/cx.py python:3.7-slim sh -c "mkdir -p /tmp/t && python /cx.py find 'print_json indent' | head -4 && python /cx.py show rich/console.py:200 | head -3"
