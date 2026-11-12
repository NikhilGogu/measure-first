---
name: tools
description: Installs code search, robust editing and test helpers into /tmp/t. Run scripts/install.py once at the start.
---
Run `run_skill_script` with skill_name `tools` and file_path `scripts/install.py` once. Then use, through run_command:
- `python3 /tmp/t/find.py <identifiers or issue summary>`
- `python3 /tmp/t/apply.py <<'EOF'` with SEARCH/REPLACE blocks
- `python3 /tmp/t/t.py <test or source file> [-k expr]`
