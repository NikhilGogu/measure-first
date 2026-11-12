#!/bin/bash
# Unit test for apply.py: indented input, fuzzy whitespace match, syntax guard.
rm -rf /tmp/at && mkdir -p /tmp/at/pkg
printf 'def f(x):\n    if x:\n        return 1\n    return 2\n' > /tmp/at/pkg/m.py
sed 's#WS = "/workspace"#WS = "/tmp/at"#' ~/g4/src/tools/apply.py > /tmp/apply_test.py
python3 /tmp/apply_test.py <<'BLOCK'
   FILE: pkg/m.py
   <<<<<<< SEARCH
   def f(x):
     if x:
           return 1
   =======
   def f(x):
       if x is not None:
           return 1
   >>>>>>> REPLACE
   trailing junk from a broken tool-call delimiter }<tool_call|>
BLOCK
echo "exit=$?"
cat /tmp/at/pkg/m.py
echo "--- syntax guard test"
python3 /tmp/apply_test.py <<'BLOCK'
FILE: pkg/m.py
<<<<<<< SEARCH
    return 2
=======
    return (2
>>>>>>> REPLACE
BLOCK
echo "exit=$?"
cat /tmp/at/pkg/m.py
