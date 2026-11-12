#!/bin/bash
# cx robustness tests: wrong ref retargeted by name, header in argv, missing END, bare code.
rm -rf /tmp/cxws /tmp/t/cx_refs.json && cp -r ~/g4/snapcache/rich_4077 /tmp/cxws
sed 's#WS = "/workspace"#WS = "/tmp/cxws"#' ~/g4/src/tools/cx.py > /tmp/cx_test.py
CX="python3 /tmp/cx_test.py"
$CX find "print_json" | sed -n 2,3p
echo "=== 1. wrong ref (@1 is print_json in __init__) but code defines FileProxy.isatty-like method fileno -> retarget by name"
$CX outline rich/file_proxy.py | head -8
$CX blocks <<'EOF'
REPLACE @1
def fileno(self) -> int:
    """retarget test"""
    return self.__file.fileno()
END
EOF
git -C /tmp/cxws diff --stat
echo "=== 2. header in argv, no END, bare code"
printf 'def flush(self) -> None:\n    """argv header test"""\n    output = "".join(self.__buffer)\n    if output:\n        self.__console.print(output)\n    del self.__buffer[:]\n' | $CX blocks rich/file_proxy.py 50 | head -4
echo "=== 3. no header at all, code defines a unique name"
printf 'def _ipy_display_hook(*args, **kwargs):\n    """bare code test"""\n    return None\n' | $CX blocks | head -3
git -C /tmp/cxws diff --stat
