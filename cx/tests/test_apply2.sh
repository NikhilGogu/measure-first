#!/bin/bash
# apply.py tests: literal \n repair, indented input, junk after marker, syntax guard, multi-block.
rm -rf /tmp/at && mkdir -p /tmp/at/pkg
printf 'def f(x):\n    params.append({\n        "a": 1,\n        "b": 2,\n    })\n    return params\n' > /tmp/at/pkg/m.py
sed 's#WS = "/workspace"#WS = "/tmp/at"#' ~/g4/src/tools/apply.py > /tmp/apply_test.py
echo "--- literal backslash-n repair"
python3 /tmp/apply_test.py <<'BLOCK'
FILE: pkg/m.py
<<<<<<< SEARCH
        "b": 2,
    })
=======
        "b": 2,\n        "c": 3,\n    })
>>>>>>> REPLACE
junk }<tool_call|>
BLOCK
echo "exit=$?"; cat /tmp/at/pkg/m.py
echo "--- syntax guard keeps file unchanged"
python3 /tmp/apply_test.py <<'BLOCK'
   FILE: pkg/m.py
   <<<<<<< SEARCH
       return params
   =======
       return (params
   >>>>>>> REPLACE
BLOCK
echo "exit=$?"; tail -2 /tmp/at/pkg/m.py
echo "--- python3.7 run"
docker run --rm -v /tmp/at:/tmp/at -v /tmp/apply_test.py:/a.py python:3.7-slim python /a.py <<'BLOCK'
FILE: pkg/m.py
<<<<<<< SEARCH
    return params
=======
    return list(params)
>>>>>>> REPLACE
BLOCK
echo "exit=$?"; tail -1 /tmp/at/pkg/m.py
