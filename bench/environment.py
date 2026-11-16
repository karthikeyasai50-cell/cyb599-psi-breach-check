"""Print the hardware and software environment a run was measured on."""

import os
import platform
import sys
from importlib import metadata

cpu = "unknown"
try:
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
except OSError:
    cpu = platform.processor() or "unknown"

mem = "unknown"
try:
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            mem = f"{int(line.split()[1]) / 1024 / 1024:.1f} GB"
            break
except OSError:
    pass

print(f"python      {sys.version.split()[0]} ({platform.platform()})")
print(f"cpu         {cpu}, {os.cpu_count()} logical cores")
print(f"memory      {mem}")
for pkg in ("numpy", "Pyfhel", "openmined_psi", "matplotlib", "pytest"):
    try:
        print(f"{pkg:<11} {metadata.version(pkg)}")
    except metadata.PackageNotFoundError:
        print(f"{pkg:<11} not installed")
