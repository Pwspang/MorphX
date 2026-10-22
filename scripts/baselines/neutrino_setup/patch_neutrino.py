"""Patch upstream Neutrino 0.1.0 for hpcc-style hosts (used by install_neutrino.sh).

build.py:
- detect CUDA by the presence of nvidia-smi: on GPU-less login nodes it exits
  non-zero, which upstream treats as an unhandled CalledProcessError
- look for cuda.h under $CUDA_HOME first
cli.py:
- resolve the real driver to its versioned file at run time. preload.so redirects
  every dlopen whose path contains "libcuda.so.1" to the hook, so the real driver
  must not be passed as the SONAME link; and the versioned file differs between
  the login node (where build.py runs) and the GPU nodes.
"""
import os
import re
import sys

src, cuda_home = sys.argv[1], sys.argv[2].rstrip("/")


def patch(path, edits):
    s = open(path).read()
    for old, new, marker in edits:
        if marker in s:
            continue
        if isinstance(old, re.Pattern):
            if not old.search(s):
                sys.exit(f"patch_neutrino.py: pattern not found in {path}, upstream changed?")
            s = old.sub(lambda _: new, s, count=1)
        else:
            if old not in s:
                sys.exit(f"patch_neutrino.py: {old!r} not found in {path}, upstream changed?")
            s = s.replace(old, new, 1)
    open(path, "w").write(s)


new_check = """def check_command(cmd: str):
    # MorphX: only test that the tool exists; nvidia-smi exits non-zero on GPU-less nodes
    import shutil
    return shutil.which(cmd) is not None"""
old_path = '"/usr/local/cuda/targets/x86_64-linux/include/", # for x86'
patch(os.path.join(src, "neutrino", "build.py"), [
    (re.compile(r"def check_command\(cmd: str\):\n    try:\n.*?    except FileNotFoundError:\n        return False", re.S),
     new_check, "shutil.which(cmd)"),
    (old_path, f'"{cuda_home}/targets/x86_64-linux/include/", # MorphX: $CUDA_HOME\n        ' + old_path,
     f"{cuda_home}/targets/x86_64-linux/include/"),
])

old_driver = "    NEUTRINO_REAL_DRIVER: str = args.driver\n"
patch(os.path.join(src, "neutrino", "cli.py"), [
    (old_driver,
     old_driver + "    # MorphX: versioned file of this node's driver; preload.so would redirect a\n"
                  "    # path containing the hook name (libcuda.so.1) back to the hook\n"
                  "    NEUTRINO_REAL_DRIVER = os.path.realpath(NEUTRINO_REAL_DRIVER)\n",
     "NEUTRINO_REAL_DRIVER = os.path.realpath(NEUTRINO_REAL_DRIVER)"),
])
