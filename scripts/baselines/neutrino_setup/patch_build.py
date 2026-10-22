"""Patch upstream neutrino/build.py for hpcc-style hosts (used by install_neutrino.sh).

- detect CUDA by the presence of nvidia-smi: on GPU-less login nodes it exits
  non-zero, which upstream treats as an unhandled CalledProcessError
- look for cuda.h under $CUDA_HOME first
"""
import re
import sys

path, cuda_home = sys.argv[1], sys.argv[2].rstrip("/")
s = open(path).read()

old_check = re.compile(r"def check_command\(cmd: str\):\n    try:\n.*?    except FileNotFoundError:\n        return False", re.S)
new_check = """def check_command(cmd: str):
    # MorphX: only test that the tool exists; nvidia-smi exits non-zero on GPU-less nodes
    import shutil
    return shutil.which(cmd) is not None"""
if old_check.search(s):
    s = old_check.sub(lambda _: new_check, s, count=1)
elif "shutil.which(cmd)" not in s:
    sys.exit("patch_build.py: check_command not found, upstream changed?")

old_path = '"/usr/local/cuda/targets/x86_64-linux/include/", # for x86'
new_path = f'"{cuda_home}/targets/x86_64-linux/include/", # MorphX: $CUDA_HOME\n        ' + old_path
if f"{cuda_home}/targets/x86_64-linux/include/" not in s:
    if old_path not in s:
        sys.exit("patch_build.py: header search path not found, upstream changed?")
    s = s.replace(old_path, new_path)

open(path, "w").write(s)
