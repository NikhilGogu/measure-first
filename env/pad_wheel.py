"""Pad a wheel past the harness's 20 MB large-wheel threshold so it is injected only into repos that import it.
usage: pad_wheel.py WHEEL_PATH"""
import os
import sys
import zipfile

path = sys.argv[1]
name = os.path.basename(path)
dist, ver = name.split("-")[0], name.split("-")[1]
with zipfile.ZipFile(path) as z:
    if any(n.endswith("/PADDING.bin") for n in z.namelist()):
        print("already padded", os.path.getsize(path))
        sys.exit(0)
with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_STORED) as z:
    z.writestr(f"{dist}-{ver}.dist-info/PADDING.bin", os.urandom(21 * 1024 * 1024))
print(name, os.path.getsize(path))
